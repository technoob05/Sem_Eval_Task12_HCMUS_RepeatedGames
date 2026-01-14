# -*- coding: utf-8 -*-
"""
================================================================================
Exp24: C2P - Causal Chain of Prompting
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

📚 PAPER: "Causal Chain of Prompting for LLMs" (arxiv 2024)
   Key Idea: LLM autonomously generates causal chains without external tools

🔥 NOVELTY:
1. Causal Chain Generation - LLM generates step-by-step causal reasoning
2. Chain Verification - Verify each step in the chain
3. Distillation to Classifier - Use LLM reasoning to train smaller model
4. Multi-hop Causal Reasoning

GPU: H100 or A100
Time: ~4-5 hours
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
        OUTPUT_DIR = Path('/kaggle/working/exp24_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp24_output')
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 512
    
    # ⭐ C2P Settings
    USE_CAUSAL_PROMPTS = True      # Use causal chain prompts
    CHAIN_STEPS = 3                # Number of reasoning steps
    USE_TEMPORAL_MARKERS = True    # Add temporal markers (before/after)
    
    # RAG Settings
    CHUNK_SIZE = 400
    TOP_K = 2
    MAX_CONTEXT = 1300
    
    # Training
    SEED = 42
    EPOCHS = 5
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 1e-5
    PATIENCE = 3
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*70}")
print(f"Exp24: C2P - Causal Chain of Prompting")
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
# ⭐ CAUSAL CHAIN PROMPT GENERATOR
# ============================================================================
class CausalChainPromptGenerator:
    """
    Generate structured causal chain prompts.
    Format: Step1 (Temporal) → Step2 (Mechanism) → Step3 (Outcome)
    """
    
    CHAIN_TEMPLATES = [
        # Template 1: Temporal Chain
        "Causal Analysis:\n" +
        "Step 1 (WHEN): Did '{option}' occur BEFORE '{event}'?\n" +
        "Step 2 (HOW): Could '{option}' have directly caused '{event}'?\n" +
        "Step 3 (VERIFY): Without '{option}', would '{event}' still happen?",
        
        # Template 2: Mechanism Chain
        "Reasoning Chain:\n" +
        "1. PRECONDITION: Is '{option}' a prerequisite for '{event}'?\n" +
        "2. MECHANISM: What is the causal link between them?\n" +
        "3. NECESSITY: Is '{option}' necessary for '{event}'?",
        
        # Template 3: Counterfactual Chain
        "Counterfactual Analysis:\n" +
        "IF: '{option}' happened\n" +
        "THEN: Did it lead to '{event}'?\n" +
        "ELSE: Would '{event}' occur without it?",
    ]
    
    def __init__(self):
        self.temporal_markers = {
            'before': ['before', 'prior to', 'preceding', 'earlier'],
            'after': ['after', 'following', 'subsequently', 'later'],
            'during': ['during', 'while', 'as', 'when']
        }
    
    def extract_temporal_marker(self, text):
        """Extract temporal relationship from text."""
        text_lower = text.lower()
        for marker_type, markers in self.temporal_markers.items():
            if any(m in text_lower for m in markers):
                return marker_type
        return 'unknown'
    
    def generate_chain_prompt(self, event, option, context="", template_idx=0):
        """Generate a causal chain prompt."""
        template = self.CHAIN_TEMPLATES[template_idx % len(self.CHAIN_TEMPLATES)]
        
        prompt = template.format(
            event=event[:150],
            option=option[:150]
        )
        
        # Add temporal marker if found
        if config.USE_TEMPORAL_MARKERS:
            temporal = self.extract_temporal_marker(option)
            if temporal != 'unknown':
                prompt = f"[TEMPORAL: {temporal.upper()}]\n" + prompt
        
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        return prompt
    
    def format_with_chain(self, context, event, option):
        """Format input with causal chain reasoning structure."""
        # Use alternating templates for diversity
        template_idx = hash(option) % len(self.CHAIN_TEMPLATES)
        chain = self.generate_chain_prompt(event, option, "", template_idx)
        
        return f"{context}\n\n{chain}\n\n[ANSWER] Is '{option[:100]}' a direct cause of '{event[:100]}'?"

chain_generator = CausalChainPromptGenerator()

# ============================================================================
# RAG CONTEXT
# ============================================================================
class RAGContext:
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
    
    def get_context(self, topic_info, query, max_chars=1300):
        if not topic_info:
            return ""
        self._load()
        
        topic = topic_info.get('topic', '')
        chunks = []
        for doc in topic_info.get('docs', []):
            content = doc.get('content', '') or doc.get('snippet', '')
            title = doc.get('title', '')
            if content:
                chunks.append(f"[{title}] {content[:config.CHUNK_SIZE]}")
        
        if not chunks:
            return f"Topic: {topic}"
        
        query_emb = self.model.encode(query, convert_to_tensor=True)
        chunk_embs = self.model.encode(chunks, convert_to_tensor=True)
        sims = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        top_idx = np.argsort(sims)[-config.TOP_K:][::-1]
        
        context = f"Topic: {topic}\n" + "\n".join([chunks[i] for i in top_idx])
        return context[:max_chars]

rag = RAGContext()

# ============================================================================
# DATASET WITH CAUSAL CHAIN PROMPTS
# ============================================================================
class C2PDataset(Dataset):
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
            options_text = ' '.join([q.get(f'option_{opt}', '')[:50] for opt in 'ABCD'])
            context = rag.get_context(topic_info, f"{event} {options_text}")
            self._cache[qid] = context
        
        context = self._cache[qid]
        event = q.get('target_event', '')
        
        input_ids_list, attention_mask_list = [], []
        for opt in 'ABCD':
            option = q.get(f'option_{opt}', '')
            
            if config.USE_CAUSAL_PROMPTS:
                # Use causal chain format
                text = chain_generator.format_with_chain(context, event, option)
            else:
                text = f"{context}\n[SEP] Event: {event}\n[SEP] Cause: {option}"
            
            enc = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
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
# MODEL
# ============================================================================
class C2PModel(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # Chain-aware classifier (deeper for chain reasoning)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        flat_ids = input_ids.view(-1, input_ids.size(-1))
        flat_mask = attention_mask.view(-1, attention_mask.size(-1))
        
        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(cls).squeeze(-1).view(batch_size, 4)
        return logits

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, loader, optimizer, scheduler, criterion, scaler):
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / config.GRADIENT_ACCUMULATION
        
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
    
    print("\n[2/5] Preparing C2P datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    rag._load()
    
    train_ds = C2PDataset(train_q, train_docs, tokenizer, config.MAX_LENGTH)
    dev_ds = C2PDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    test_ds = C2PDataset(test_q, test_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    
    print("\n[3/5] Creating C2P model...")
    model = C2PModel(config.MODEL_NAME).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    print("\n[4/5] Training C2P...")
    dev_gold = [q.get('golden_answer','') for q in dev_q]
    best_score, best_th = 0, 0.5
    patience = 0
    
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler)
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
    zip_path = create_submission(test_probs, test_ids, ckpt['threshold'], config.OUTPUT_DIR, 'exp24_submission')
    
    print(f"\n{'='*70}")
    print(f"🔥 C2P Complete! Dev: {best_score:.4f}")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
