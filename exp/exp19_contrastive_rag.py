# -*- coding: utf-8 -*-
"""
================================================================================
Exp19: DeBERTa + Contrastive + Multi-Task + RAG-Enhanced Context
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

🔥 IMPROVEMENTS OVER EXP15:
1. ⭐ Fixed Contrastive Loss (no more negative loss!)
2. ⭐ RAG-enhanced context with FULL document content
3. ⭐ Uses document titles for better understanding
4. All exp15 features: Label smoothing, FGM adversarial

GPU: H100 or 2x A100
Time: ~3-4 hours
Expected Score: 0.70+
================================================================================
"""

import json, random, shutil, warnings
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
        OUTPUT_DIR = Path('/kaggle/working/exp19_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp19_output')
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 512  # ⭐ Increased for more context
    
    # ⭐ RAG Settings
    USE_FULL_CONTENT = True
    USE_TITLES = True
    CHUNK_SIZE = 400
    TOP_K_CHUNKS = 2
    MAX_CONTEXT_CHARS = 1500
    
    # Training
    SEED = 42
    EPOCHS = 5
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 1e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    PATIENCE = 3
    
    # Multi-task weights
    BCE_WEIGHT = 1.0
    CONTRASTIVE_WEIGHT = 0.2  # ⭐ Reduced weight
    CONTRASTIVE_MARGIN = 0.3  # ⭐ Margin for triplet loss
    LABEL_SMOOTHING = 0.1
    
    # Adversarial training
    USE_FGM = True
    FGM_EPSILON = 0.5
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*70}")
print(f"Exp19: DeBERTa + Fixed Contrastive + RAG")
print(f"{'='*70}")
print(f"Full Content: {config.USE_FULL_CONTENT} | Titles: {config.USE_TITLES}")
print(f"BCE:{config.BCE_WEIGHT} + Contrastive:{config.CONTRASTIVE_WEIGHT} (margin={config.CONTRASTIVE_MARGIN})")
print(f"Label Smoothing:{config.LABEL_SMOOTHING}, FGM:{config.USE_FGM}")
print(f"{'='*70}")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
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
# ⭐ RAG-ENHANCED CONTEXT
# ============================================================================
class RAGContextBuilder:
    """Build rich context from full document content."""
    
    def __init__(self):
        self.model = None
        self._initialized = False
    
    def _load_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                import subprocess
                subprocess.run(['pip', 'install', 'sentence-transformers', '-q'])
                from sentence_transformers import SentenceTransformer
            
            if not self._initialized:
                print("  Loading embedding model for RAG...")
                self._initialized = True
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def chunk_text(self, text, chunk_size=400):
        """Split text into chunks."""
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                for sep in ['. ', '! ', '? ', '\n']:
                    last_sep = chunk.rfind(sep)
                    if last_sep > chunk_size // 2:
                        chunk = chunk[:last_sep + 1]
                        end = start + last_sep + 1
                        break
            
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end
        
        return chunks
    
    def get_context(self, topic_info, query, max_chars=1500, top_k=2):
        """Get context with semantic retrieval."""
        if not topic_info:
            return ""
        
        self._load_model()
        
        topic_name = topic_info.get('topic', '')
        all_chunks = []
        
        for doc in topic_info.get('docs', []):
            title = doc.get('title', '')
            content = doc.get('content', '') if config.USE_FULL_CONTENT else ''
            snippet = doc.get('snippet', '')
            
            text = content if content else snippet
            if not text:
                continue
            
            chunks = self.chunk_text(text, config.CHUNK_SIZE)
            for chunk in chunks:
                if config.USE_TITLES and title:
                    all_chunks.append(f"[{title}] {chunk}")
                else:
                    all_chunks.append(chunk)
        
        if not all_chunks:
            return f"Topic: {topic_name}"
        
        # Semantic retrieval
        query_emb = self.model.encode(query, convert_to_tensor=True)
        chunk_embs = self.model.encode(all_chunks, convert_to_tensor=True)
        
        sims = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        top_idx = np.argsort(sims)[-top_k:][::-1]
        
        selected = [all_chunks[i] for i in top_idx]
        context = f"Topic: {topic_name}\n\nEvidence:\n" + "\n---\n".join(selected)
        
        return context[:max_chars]

# Global instance
rag_builder = RAGContextBuilder()

# ============================================================================
# DATASET WITH RAG
# ============================================================================
class AERDataset(Dataset):
    def __init__(self, questions, docs, tokenizer, max_len=512, is_test=False):
        self.questions = questions
        self.docs = docs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self._cache = {}
        
    def __len__(self):
        return len(self.questions)
    
    def _get_context(self, q):
        qid = q.get('id')
        if qid in self._cache:
            return self._cache[qid]
        
        topic_info = self.docs.get(q.get('topic_id'), {})
        event = q.get('target_event', '')
        options = ' '.join([q.get(f'option_{opt}', '') for opt in 'ABCD'])
        query = f"{event} {options}"
        
        context = rag_builder.get_context(
            topic_info, query,
            max_chars=config.MAX_CONTEXT_CHARS,
            top_k=config.TOP_K_CHUNKS
        )
        self._cache[qid] = context
        return context
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        context = self._get_context(q)
        event = q.get('target_event', '')
        
        input_ids_list, attention_mask_list = [], []
        for opt in ['A','B','C','D']:
            text = f"{context} [SEP] Event: {event} [SEP] Cause: {q.get(f'option_{opt}','')}"
            enc = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
            input_ids_list.append(enc['input_ids'].squeeze(0))
            attention_mask_list.append(enc['attention_mask'].squeeze(0))
        
        result = {
            'id': q.get('id'),
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
# MODEL WITH CONTRASTIVE HEAD
# ============================================================================
class ContrastiveMultiTaskModel(nn.Module):
    def __init__(self, model_name, hidden_size=256, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        enc_hidden = self.encoder.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(enc_hidden, enc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden // 2, 1)
        )
        
        # Contrastive projection head
        self.projection = nn.Sequential(
            nn.Linear(enc_hidden, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        batch_size = input_ids.size(0)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        
        logits = self.classifier(cls).squeeze(-1).view(batch_size, 4)
        
        if return_embeddings:
            embeddings = self.projection(cls).view(batch_size, 4, -1)
            return logits, embeddings
        return logits

# ============================================================================
# ⭐ FIXED CONTRASTIVE LOSS (Always non-negative!)
# ============================================================================
class ContrastiveLoss(nn.Module):
    """Triplet-style contrastive loss that is always non-negative."""
    
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        embeddings: (batch, 4, hidden)
        labels: (batch, 4) binary
        
        Goal: positive pairs should be closer than negative pairs by margin
        """
        batch_size = embeddings.size(0)
        loss = 0.0
        count = 0
        
        for i in range(batch_size):
            emb = F.normalize(embeddings[i], dim=-1)  # (4, hidden)
            lbl = labels[i]  # (4,)
            
            pos_mask = lbl == 1
            neg_mask = lbl == 0
            
            n_pos = pos_mask.sum().item()
            n_neg = neg_mask.sum().item()
            
            if n_pos > 0 and n_neg > 0:
                pos_emb = emb[pos_mask]  # (n_pos, hidden)
                neg_emb = emb[neg_mask]  # (n_neg, hidden)
                
                # For each positive, compute triplet loss with all negatives
                for p_idx in range(int(n_pos)):
                    anchor = pos_emb[p_idx]  # (hidden,)
                    
                    # Similarity to other positives (if any)
                    if n_pos > 1:
                        other_pos = torch.cat([pos_emb[:p_idx], pos_emb[p_idx+1:]], dim=0)
                        pos_sim = (anchor * other_pos).sum(dim=-1).mean()  # Average similarity to positives
                    else:
                        pos_sim = torch.tensor(1.0, device=emb.device)  # Max similarity
                    
                    # Similarity to negatives
                    neg_sim = (anchor.unsqueeze(0) * neg_emb).sum(dim=-1)  # (n_neg,)
                    
                    # Triplet loss: neg_sim should be < pos_sim - margin
                    # loss = max(0, neg_sim - pos_sim + margin)
                    triplet_loss = F.relu(neg_sim - pos_sim + self.margin).mean()
                    loss += triplet_loss
                    count += 1
        
        return loss / max(count, 1)

# ============================================================================
# FGM ADVERSARIAL TRAINING
# ============================================================================
class FGM:
    def __init__(self, model, epsilon=0.5):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}
    
    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)
    
    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, loader, optimizer, scheduler, bce_criterion, contrastive_criterion, scaler, fgm=None):
    model.train()
    total_loss, total_bce, total_con = 0, 0, 0
    
    pbar = tqdm(loader, desc="Training")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        
        # Apply label smoothing
        smooth_labels = labels * (1 - config.LABEL_SMOOTHING) + config.LABEL_SMOOTHING / 4
        
        with autocast():
            logits, embeddings = model(input_ids, attention_mask, return_embeddings=True)
            bce_loss = bce_criterion(logits, smooth_labels)
            contrastive_loss = contrastive_criterion(embeddings, labels)
            loss = config.BCE_WEIGHT * bce_loss + config.CONTRASTIVE_WEIGHT * contrastive_loss
            loss = loss / config.GRADIENT_ACCUMULATION
        
        scaler.scale(loss).backward()
        
        # FGM adversarial training
        if fgm and (step + 1) % config.GRADIENT_ACCUMULATION == 0:
            fgm.attack()
            with autocast():
                logits_adv, _ = model(input_ids, attention_mask, return_embeddings=True)
                loss_adv = bce_criterion(logits_adv, smooth_labels) / config.GRADIENT_ACCUMULATION
            scaler.scale(loss_adv).backward()
            fgm.restore()
        
        if (step + 1) % config.GRADIENT_ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION
        total_bce += bce_loss.item()
        total_con += contrastive_loss.item()
        pbar.set_postfix({'loss': f'{total_loss/(step+1):.4f}', 'bce': f'{total_bce/(step+1):.4f}', 'con': f'{total_con/(step+1):.4f}'})
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_ids = [], []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        
        with autocast():
            logits = model(input_ids, attention_mask)
        
        all_preds.extend(torch.sigmoid(logits).cpu().numpy())
        all_ids.extend(batch['id'])
    
    return np.array(all_preds), all_ids

def optimize_threshold(probs, golds):
    opts = ['A','B','C','D']
    best_th, best_sc = 0.5, 0
    for th in np.arange(0.20, 0.70, 0.05):
        ps = [','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]])) for p in probs]
        sc = compute_aer_score(ps, golds)
        if sc > best_sc: best_sc, best_th = sc, th
    return best_th, best_sc

def create_predictions(probs, ids, th):
    opts = ['A','B','C','D']
    return [{'id': qid, 'answer': ','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]]))} 
            for qid, p in zip(ids, probs)]

def create_submission(preds, out_dir, name):
    sub = out_dir/'submission'; sub.mkdir(exist_ok=True)
    with open(sub/'submission.jsonl','w') as f:
        for p in preds: f.write(json.dumps(p)+'\n')
    shutil.make_archive(str(out_dir/name),'zip',sub)
    return out_dir/f'{name}.zip'

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n[1/6] Loading data...")
    train_q, dev_q, test_q = load_questions('train_data'), load_questions('dev_data'), load_questions('test_data')
    train_docs, dev_docs, test_docs = load_docs('train_data'), load_docs('dev_data'), load_docs('test_data')
    print(f"  Train:{len(train_q)} Dev:{len(dev_q)} Test:{len(test_q)}")
    
    print("\n[2/6] Preparing datasets with RAG...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Initialize RAG builder
    rag_builder._load_model()
    
    train_ds = AERDataset(train_q, train_docs, tokenizer, config.MAX_LENGTH)
    dev_ds = AERDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    test_ds = AERDataset(test_q, test_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    
    print("\n[3/6] Creating model...")
    model = ContrastiveMultiTaskModel(config.MODEL_NAME).to(config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.WARMUP_RATIO), total_steps)
    
    bce_criterion = nn.BCEWithLogitsLoss()
    contrastive_criterion = ContrastiveLoss(margin=config.CONTRASTIVE_MARGIN)  # ⭐ Fixed!
    scaler = GradScaler()
    fgm = FGM(model, config.FGM_EPSILON) if config.USE_FGM else None
    
    print("\n[4/6] Training...")
    dev_gold = [q.get('golden_answer','') for q in dev_q]
    best_score, best_th = 0, 0.5
    patience = 0
    
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, bce_criterion, contrastive_criterion, scaler, fgm)
        
        dev_probs, _ = evaluate(model, dev_loader)
        th, score = optimize_threshold(dev_probs, dev_gold)
        print(f"  Train Loss: {train_loss:.4f}, Dev Score: {score:.4f} (th={th:.2f})")
        
        if score > best_score:
            best_score, best_th = score, th
            patience = 0
            torch.save({'model': model.state_dict(), 'score': score, 'threshold': th}, config.OUTPUT_DIR/'best_model.pt')
            print(f"  -> New best!")
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print("  Early stopping!")
                break
    
    print(f"\nBest Dev Score: {best_score:.4f}")
    
    print("\n[5/6] Predicting on TEST...")
    ckpt = torch.load(config.OUTPUT_DIR/'best_model.pt', weights_only=False)
    model.load_state_dict(ckpt['model'])
    test_probs, test_ids = evaluate(model, test_loader)
    test_preds = create_predictions(test_probs, test_ids, ckpt['threshold'])
    
    print("\n[6/6] Generating submission...")
    zip_path = create_submission(test_preds, config.OUTPUT_DIR, 'exp19_submission')
    
    # Save results
    with open(config.OUTPUT_DIR/'results.json', 'w') as f:
        json.dump({
            'dev_score': float(best_score),
            'threshold': float(best_th),
            'use_full_content': config.USE_FULL_CONTENT,
            'use_titles': config.USE_TITLES,
            'contrastive_margin': config.CONTRASTIVE_MARGIN,
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🔥 DONE!")
    print(f"  Dev Score: {best_score:.4f}")
    print(f"  Test: {len(test_q)} questions")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
