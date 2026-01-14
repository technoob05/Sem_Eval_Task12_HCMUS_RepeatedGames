# -*- coding: utf-8 -*-
"""
================================================================================
Exp23: CF-RAG - Counterfactual Reasoning for RAG
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

📚 PAPER: "Counterfactual Reasoning for Retrieval-Augmented Generation"
   Venue: NeurIPS 2024 Workshop
   Key Idea: Use counterfactual reasoning to avoid "correlation trap"

🔥 NOVELTY:
1. Counterfactual Query Generation
2. Contrastive Evidence Retrieval
3. Correlation vs Causation Discrimination
4. Counterfactual Data Augmentation

GPU: H100 or A100
Time: ~3-4 hours
================================================================================
"""

import json, random, shutil, warnings, re
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp23_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp23_output')
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 512
    
    # ⭐ CF-RAG Settings
    USE_COUNTERFACTUAL_QUERY = True   # Generate "What if NOT X?" queries
    USE_CONTRASTIVE_EVIDENCE = True   # Retrieve both pro and contra evidence
    CF_AUGMENT_PROB = 0.2             # Probability of counterfactual augmentation
    NEGATION_PATTERNS = [
        ('caused', 'did not cause'),
        ('led to', 'did not lead to'),
        ('resulted in', 'did not result in'),
        ('triggered', 'did not trigger'),
        ('before', 'after'),
        ('because', 'despite'),
    ]
    
    # RAG Settings
    CHUNK_SIZE = 400
    TOP_K_CHUNKS = 2
    MAX_CONTEXT = 1400
    
    # Training
    SEED = 42
    EPOCHS = 5
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 1e-5
    PATIENCE = 3
    
    # Loss
    BCE_WEIGHT = 1.0
    CF_WEIGHT = 0.25   # Counterfactual contrastive weight
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*70}")
print(f"Exp23: CF-RAG - Counterfactual Reasoning for RAG")
print(f"{'='*70}")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(config.SEED)

def compute_aer_score(preds, golds):
    scores = []
    for p, g in zip(preds, golds):
        ps = set(x.strip() for x in p.split(',')) if p else set()
        gs = set(x.strip() for x in g.split(',')) if g else set()
        scores.append(1.0 if ps==gs else 0.5 if ps and ps.issubset(gs) else 0.0)
    return sum(scores)/len(scores) if scores else 0.0

def load_questions(split):
    with open(config.DATA_DIR/split/'questions.jsonl','r',encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def load_docs(split):
    with open(config.DATA_DIR/split/'docs.json','r',encoding='utf-8') as f:
        return {d['topic_id']:d for d in json.load(f)}

# ============================================================================
# ⭐ COUNTERFACTUAL QUERY GENERATOR
# ============================================================================
class CounterfactualGenerator:
    """
    Generate counterfactual queries:
    Original: "X caused Y"
    Counterfactual: "What if X did NOT happen? Would Y still occur?"
    """
    
    def __init__(self):
        self.patterns = config.NEGATION_PATTERNS
    
    def generate_counterfactual(self, text):
        """Generate counterfactual version of text."""
        cf_text = text
        for original, negated in self.patterns:
            if original in text.lower():
                cf_text = re.sub(
                    re.escape(original), negated, text, 
                    count=1, flags=re.IGNORECASE
                )
                break
        
        # If no pattern matched, add negation prefix
        if cf_text == text:
            cf_text = f"If NOT: {text}"
        
        return cf_text
    
    def generate_cf_query(self, event, option):
        """Generate counterfactual query for retrieval."""
        return f"What if {option[:100]} did NOT happen? Would {event[:100]} still occur?"

cf_generator = CounterfactualGenerator()

# ============================================================================
# ⭐ CONTRASTIVE EVIDENCE RETRIEVER
# ============================================================================
class ContrastiveEvidenceRetriever:
    """
    Retrieve both supporting AND contradicting evidence.
    Helps model distinguish causation from correlation.
    """
    
    def __init__(self):
        self.model = None
    
    def _load(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except:
                import subprocess
                subprocess.run(['pip', 'install', 'sentence-transformers', '-q'])
                from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def retrieve_contrastive(self, topic_info, event, options, max_chars=1400):
        """Retrieve with both regular and counterfactual queries."""
        if not topic_info:
            return ""
        
        self._load()
        
        topic = topic_info.get('topic', '')
        
        # Build regular query
        regular_query = f"{event} " + " ".join([opt[:50] for opt in options])
        
        # Build counterfactual query
        if config.USE_COUNTERFACTUAL_QUERY:
            cf_query = cf_generator.generate_cf_query(event, options[0])
        else:
            cf_query = regular_query
        
        # Collect chunks
        chunks = []
        for doc in topic_info.get('docs', []):
            title = doc.get('title', '')
            content = doc.get('content', '') or doc.get('snippet', '')
            if content:
                for i in range(0, min(len(content), 1500), config.CHUNK_SIZE - 50):
                    chunk = content[i:i+config.CHUNK_SIZE]
                    if len(chunk) > 50:
                        chunks.append(f"[{title}] {chunk}")
        
        if not chunks:
            return f"Topic: {topic}"
        
        # Regular retrieval
        reg_emb = self.model.encode(regular_query, convert_to_tensor=True)
        chunk_embs = self.model.encode(chunks, convert_to_tensor=True)
        reg_sims = torch.cosine_similarity(reg_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        
        # Get top supporting evidence
        top_support = np.argsort(reg_sims)[-config.TOP_K_CHUNKS:][::-1]
        support_chunks = [chunks[i] for i in top_support]
        
        # Counterfactual retrieval (find potentially contradicting evidence)
        if config.USE_CONTRASTIVE_EVIDENCE:
            cf_emb = self.model.encode(cf_query, convert_to_tensor=True)
            cf_sims = torch.cosine_similarity(cf_emb.unsqueeze(0), chunk_embs).cpu().numpy()
            
            # Find chunks that are similar to CF but different from regular
            contrast_score = cf_sims - 0.5 * reg_sims
            top_contrast = np.argsort(contrast_score)[-1:][::-1]  # Get 1 contrasting chunk
            contrast_chunks = [chunks[i] for i in top_contrast if i not in top_support]
        else:
            contrast_chunks = []
        
        # Assemble context
        context_parts = [f"Topic: {topic}", "\nSupporting Evidence:"]
        context_parts.extend(support_chunks)
        
        if contrast_chunks:
            context_parts.append("\nAlternative Perspective:")
            context_parts.extend(contrast_chunks)
        
        return "\n".join(context_parts)[:max_chars]

retriever = ContrastiveEvidenceRetriever()

# ============================================================================
# ⭐ COUNTERFACTUAL DATA AUGMENTATION
# ============================================================================
class CFAugmenter:
    """Augment training data with counterfactual examples."""
    
    def __init__(self, prob=0.2):
        self.prob = prob
    
    def augment(self, text, labels):
        """Create counterfactual version with swapped semantics."""
        if random.random() > self.prob:
            return text, labels, False
        
        # Apply counterfactual transformation
        cf_text = cf_generator.generate_counterfactual(text)
        return cf_text, labels, True

augmenter = CFAugmenter(config.CF_AUGMENT_PROB)

# ============================================================================
# DATASET
# ============================================================================
class CFRAGDataset(Dataset):
    def __init__(self, questions, docs, tokenizer, max_len=512, is_test=False):
        self.questions = questions
        self.docs = docs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self._cache = {}
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        qid = q.get('id')
        
        if qid not in self._cache:
            topic_info = self.docs.get(q.get('topic_id'), {})
            event = q.get('target_event', '')
            options = [q.get(f'option_{opt}', '') for opt in 'ABCD']
            context = retriever.retrieve_contrastive(topic_info, event, options)
            self._cache[qid] = context
        
        context = self._cache[qid]
        event = q.get('target_event', '')
        
        input_ids_list, attention_mask_list = [], []
        for opt in 'ABCD':
            option_text = q.get(f'option_{opt}', '')
            full_text = f"{context}\n[SEP] Event: {event}\n[SEP] Potential Cause: {option_text}"
            
            # Apply augmentation during training
            if not self.is_test:
                full_text, _, _ = augmenter.augment(full_text, None)
            
            enc = self.tokenizer(full_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
            input_ids_list.append(enc['input_ids'].squeeze(0))
            attention_mask_list.append(enc['attention_mask'].squeeze(0))
        
        result = {
            'id': qid,
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
        }
        
        if not self.is_test:
            golden = q.get('golden_answer', '')
            labels = torch.zeros(4)
            for ans in golden.split(','):
                ans = ans.strip().upper()
                if ans in 'ABCD':
                    labels['ABCD'.index(ans)] = 1.0
            result['labels'] = labels
        
        return result

# ============================================================================
# MODEL WITH COUNTERFACTUAL HEAD
# ============================================================================
class CFRAGModel(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
        
        # Counterfactual projection for contrastive learning
        self.cf_projection = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, input_ids, attention_mask, return_cf_emb=False):
        batch_size = input_ids.size(0)
        flat_ids = input_ids.view(-1, input_ids.size(-1))
        flat_mask = attention_mask.view(-1, attention_mask.size(-1))
        
        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(cls).squeeze(-1).view(batch_size, 4)
        
        if return_cf_emb:
            cf_emb = self.cf_projection(cls).view(batch_size, 4, -1)
            return logits, cf_emb
        return logits

# ============================================================================
# COUNTERFACTUAL CONTRASTIVE LOSS
# ============================================================================
class CFContrastiveLoss(nn.Module):
    """Push correct options away from incorrect in embedding space."""
    
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        loss = 0.0
        count = 0
        
        for i in range(batch_size):
            emb = F.normalize(embeddings[i], dim=-1)
            lbl = labels[i]
            
            pos_mask = lbl == 1
            neg_mask = lbl == 0
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_emb = emb[pos_mask]
                neg_emb = emb[neg_mask]
                
                for p in pos_emb:
                    neg_sim = (p.unsqueeze(0) * neg_emb).sum(dim=-1)
                    triplet_loss = F.relu(neg_sim.max() + self.margin).mean()
                    loss += triplet_loss
                    count += 1
        
        return loss / max(count, 1)

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, loader, optimizer, scheduler, bce_criterion, cf_criterion, scaler):
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        
        with autocast():
            logits, cf_emb = model(input_ids, attention_mask, return_cf_emb=True)
            bce_loss = bce_criterion(logits, labels)
            cf_loss = cf_criterion(cf_emb, labels)
            loss = config.BCE_WEIGHT * bce_loss + config.CF_WEIGHT * cf_loss
            loss = loss / config.GRADIENT_ACCUMULATION
        
        scaler.scale(loss).backward()
        
        if (step + 1) % config.GRADIENT_ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_ids = [], []
    for batch in loader:
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        with autocast():
            logits = model(input_ids, attention_mask)
        all_preds.extend(torch.sigmoid(logits).cpu().numpy())
        all_ids.extend(batch['id'])
    return np.array(all_preds), all_ids

def optimize_threshold(probs, golds):
    opts = list('ABCD')
    best_th, best_sc = 0.5, 0
    for th in np.arange(0.2, 0.7, 0.05):
        ps = [','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]])) for p in probs]
        sc = compute_aer_score(ps, golds)
        if sc > best_sc: best_sc, best_th = sc, th
    return best_th, best_sc

def create_submission(preds, ids, th, out_dir, name):
    opts = list('ABCD')
    results = [{'id': qid, 'answer': ','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]]))} for qid, p in zip(ids, preds)]
    sub = out_dir/'submission'; sub.mkdir(exist_ok=True)
    with open(sub/'submission.jsonl','w') as f:
        for r in results: f.write(json.dumps(r)+'\n')
    shutil.make_archive(str(out_dir/name),'zip',sub)
    return out_dir/f'{name}.zip'

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n[1/5] Loading data...")
    train_q, dev_q, test_q = load_questions('train_data'), load_questions('dev_data'), load_questions('test_data')
    train_docs, dev_docs, test_docs = load_docs('train_data'), load_docs('dev_data'), load_docs('test_data')
    print(f"  Train:{len(train_q)} Dev:{len(dev_q)} Test:{len(test_q)}")
    
    print("\n[2/5] Preparing CF-RAG datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    retriever._load()
    
    train_ds = CFRAGDataset(train_q, train_docs, tokenizer, config.MAX_LENGTH)
    dev_ds = CFRAGDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    test_ds = CFRAGDataset(test_q, test_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    
    print("\n[3/5] Creating CF-RAG model...")
    model = CFRAGModel(config.MODEL_NAME).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    bce_criterion = nn.BCEWithLogitsLoss()
    cf_criterion = CFContrastiveLoss()
    scaler = GradScaler()
    
    print("\n[4/5] Training CF-RAG...")
    dev_gold = [q.get('golden_answer','') for q in dev_q]
    best_score, best_th = 0, 0.5
    patience = 0
    
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        loss = train_epoch(model, train_loader, optimizer, scheduler, bce_criterion, cf_criterion, scaler)
        dev_probs, _ = evaluate(model, dev_loader)
        th, score = optimize_threshold(dev_probs, dev_gold)
        print(f"  Loss: {loss:.4f}, Dev: {score:.4f} (th={th:.2f})")
        
        if score > best_score:
            best_score, best_th = score, th
            patience = 0
            torch.save({'model': model.state_dict(), 'threshold': th}, config.OUTPUT_DIR/'best_model.pt')
            print("  -> New best!")
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print("  Early stopping!")
                break
    
    print("\n[5/5] Generating submission...")
    ckpt = torch.load(config.OUTPUT_DIR/'best_model.pt', weights_only=False)
    model.load_state_dict(ckpt['model'])
    test_probs, test_ids = evaluate(model, test_loader)
    zip_path = create_submission(test_probs, test_ids, ckpt['threshold'], config.OUTPUT_DIR, 'exp23_submission')
    
    print(f"\n{'='*70}")
    print(f"🔥 CF-RAG Complete! Dev: {best_score:.4f}")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
