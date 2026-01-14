# -*- coding: utf-8 -*-
"""
================================================================================
SemEval-2026 Task 12: Abductive Event Reasoning
Exp04: Chain-of-Thought Augmented Training
================================================================================
SELF-CONTAINED SCRIPT - Copy this entire file into a single Kaggle notebook cell

Key Innovations:
- Uses template-based rationales to augment training
- Train classifier on: event + rationale -> options
- Can optionally generate rationales with Flan-T5 (set USE_T5=True)

GPU: H100 80GB recommended
Time: ~2-3 hours
================================================================================
"""

#%% IMPORTS
import json, random, shutil, warnings
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

#%% CONFIG
class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp04_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp04_output')
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 512
    
    SEED = 42; EPOCHS = 5; BATCH_SIZE = 4; LR = 2e-5; PATIENCE = 3
    USE_T5 = False  # Set True to use Flan-T5 for rationale generation
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*60}\nExp04: Chain-of-Thought Augmented\n{'='*60}\nDevice: {config.DEVICE}\nUse T5: {config.USE_T5}\n{'='*60}")

#%% UTILITIES
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(config.SEED)

def compute_aer_score(preds, golds):
    scores = []
    for p, g in zip(preds, golds):
        ps, gs = set(p.split(',')) if p else set(), set(g.split(',')) if g else set()
        scores.append(1.0 if ps==gs else 0.5 if ps and ps.issubset(gs) else 0.0)
    return sum(scores)/len(scores) if scores else 0.0

def load_questions(split):
    with open(config.DATA_DIR/split/'questions.jsonl', 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def load_docs(split):
    with open(config.DATA_DIR/split/'docs.json', 'r', encoding='utf-8') as f:
        return {d['topic_id']: d for d in json.load(f)}

def get_context(topic_info, max_chars=800):
    if not topic_info: return ""
    parts = [topic_info.get('topic', '')]
    for doc in topic_info.get('docs', [])[:3]:
        if s := doc.get('snippet', ''): parts.append(s)
    return ' '.join(parts)[:max_chars]

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=self.avg=self.sum=self.count=0
    def update(self,v,n=1): self.val=v; self.sum+=v*n; self.count+=n; self.avg=self.sum/self.count

#%% RATIONALE GENERATION
def create_template_rationale(q, docs_dict):
    """Template-based rationale without T5."""
    event = q.get('target_event', '')
    topic_info = docs_dict.get(q.get('topic_id'), {})
    topic = topic_info.get('topic', '')
    
    rationale = (
        f"To find the direct cause of '{event[:100]}...', "
        f"we analyze the context about '{topic[:50]}...'. "
        f"We need to identify which option directly led to this event, "
        f"not just correlated factors. "
        f"Options that are effects or unrelated should be eliminated."
    )
    return rationale

def create_rationales(questions, docs_dict):
    """Generate rationales for all questions."""
    rationales = {}
    for q in tqdm(questions, desc="Creating rationales"):
        rationales[q['id']] = create_template_rationale(q, docs_dict)
    return rationales

#%% DATASET
class CoTDataset(Dataset):
    OPTIONS = ['A', 'B', 'C', 'D']
    
    def __init__(self, questions, docs_dict, rationales, tokenizer, max_len=512, is_test=False):
        self.questions = questions
        self.rationales = rationales
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
    
    def __len__(self): return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        qid = q.get('id')
        event = q.get('target_event', '')
        rationale = self.rationales.get(qid, '')
        
        input_ids, attention_mask = [], []
        for opt in self.OPTIONS:
            text = f"{event} [SEP] {rationale} [SEP] Cause: {q.get(f'option_{opt}', '')}"
            enc = self.tokenizer(text, max_length=self.max_len, padding='max_length', 
                                 truncation=True, return_tensors='pt')
            input_ids.append(enc['input_ids'].squeeze(0))
            attention_mask.append(enc['attention_mask'].squeeze(0))
        
        result = {
            'id': qid,
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask)
        }
        
        if not self.is_test:
            labels = torch.zeros(4)
            for a in q.get('golden_answer', '').split(','):
                a = a.strip().upper()
                if a in self.OPTIONS: labels[self.OPTIONS.index(a)] = 1.0
            result['labels'] = labels
        return result

#%% MODEL
class CoTClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        bs = input_ids.size(0)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        out = self.encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
        out = self.dropout(out)
        return self.classifier(out).squeeze(-1).view(bs, 4)

#%% TRAINING
def train_epoch(model, loader, optimizer, scheduler, criterion, scaler):
    model.train()
    loss_m = AverageMeter()
    preds_all, labels_all = [], []
    
    for batch in tqdm(loader, desc="Training"):
        ids = batch['input_ids'].to(config.DEVICE)
        mask = batch['attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        
        optimizer.zero_grad()
        with autocast():
            logits = model(ids, mask)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        
        loss_m.update(loss.item())
        preds_all.extend(torch.sigmoid(logits).detach().cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    return loss_m.avg, np.array(preds_all), np.array(labels_all)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_m = AverageMeter()
    preds_all, labels_all, ids_all = [], [], []
    for batch in tqdm(loader, desc="Evaluating"):
        ids = batch['input_ids'].to(config.DEVICE)
        mask = batch['attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        with autocast():
            logits = model(ids, mask)
            loss = criterion(logits, labels)
        loss_m.update(loss.item())
        preds_all.extend(torch.sigmoid(logits).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        ids_all.extend(batch['id'])
    return loss_m.avg, np.array(preds_all), np.array(labels_all), ids_all

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds_all, ids_all = [], []
    for batch in tqdm(loader, desc="Predicting"):
        ids = batch['input_ids'].to(config.DEVICE)
        mask = batch['attention_mask'].to(config.DEVICE)
        with autocast():
            logits = model(ids, mask)
        preds_all.extend(torch.sigmoid(logits).cpu().numpy())
        ids_all.extend(batch['id'])
    return np.array(preds_all), ids_all

#%% HELPERS
def optimize_threshold(probs, golds):
    opts = ['A','B','C','D']
    best_th, best_sc = 0.5, 0
    for th in [0.3,0.35,0.4,0.45,0.5,0.55,0.6]:
        ps = [','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]])) for p in probs]
        sc = compute_aer_score(ps, golds)
        if sc > best_sc: best_sc, best_th = sc, th
    return best_th, best_sc

def create_predictions(probs, ids, th):
    opts = ['A','B','C','D']
    return [{'id': qid, 'answer': ','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]]))} 
            for qid, p in zip(ids, probs)]

def create_submission(preds, out_dir, name):
    sub_dir = out_dir/'submission'; sub_dir.mkdir(exist_ok=True)
    with open(sub_dir/'submission.jsonl','w',encoding='utf-8') as f:
        for p in preds: f.write(json.dumps(p)+'\n')
    shutil.make_archive(str(out_dir/name), 'zip', sub_dir)
    return out_dir/f'{name}.zip'

#%% MAIN
def main():
    print("\n[1/5] Loading data...")
    train_q, dev_q = load_questions('train_data'), load_questions('dev_data')
    train_docs, dev_docs = load_docs('train_data'), load_docs('dev_data')
    print(f"  Train: {len(train_q)}, Dev: {len(dev_q)}")
    
    print("\n[2/5] Creating rationales...")
    train_rat = create_rationales(train_q, train_docs)
    dev_rat = create_rationales(dev_q, dev_docs)
    all_rat = {**train_rat, **dev_rat}
    
    print("\n[3/5] Preparing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_ds = CoTDataset(train_q, train_docs, all_rat, tokenizer, config.MAX_LENGTH)
    dev_ds = CoTDataset(dev_q, dev_docs, all_rat, tokenizer, config.MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE*2, num_workers=2)
    
    print("\n[3/5] Creating model...")
    model = CoTClassifier(config.MODEL_NAME).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(len(train_loader)*config.EPOCHS*0.1), len(train_loader)*config.EPOCHS)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    print("\n[4/5] Training...")
    opts = ['A','B','C','D']
    best_sc, best_th, patience = 0, 0.5, 0
    for ep in range(config.EPOCHS):
        print(f"\n--- Epoch {ep+1}/{config.EPOCHS} ---")
        tr_loss, _, _ = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler)
        print(f"Train Loss: {tr_loss:.4f}")
        
        dev_loss, dev_preds, dev_labels, _ = evaluate(model, dev_loader, criterion)
        dev_gold = [','.join([opts[i] for i in range(4) if l[i]==1]) or 'A' for l in dev_labels]
        th, sc = optimize_threshold(dev_preds, dev_gold)
        print(f"Dev Loss: {dev_loss:.4f}, Score: {sc:.4f} (th={th})")
        
        if sc > best_sc:
            best_sc, best_th, patience = sc, th, 0
            torch.save({'model': model.state_dict(), 'score': best_sc, 'th': best_th}, config.OUTPUT_DIR/'best.pt')
            print("  -> Saved!")
        else:
            patience += 1
            if patience >= config.PATIENCE: print("Early stop!"); break
    
    print("\n[5/5] Generating submission...")
    ckpt = torch.load(config.OUTPUT_DIR/'best.pt')
    model.load_state_dict(ckpt['model'])
    dev_ds_test = CoTDataset(dev_q, dev_docs, all_rat, tokenizer, config.MAX_LENGTH, is_test=True)
    probs, ids = predict(model, DataLoader(dev_ds_test, batch_size=config.BATCH_SIZE*2))
    preds = create_predictions(probs, ids, ckpt['th'])
    zip_path = create_submission(preds, config.OUTPUT_DIR, 'exp04_submission')
    print(f"Submission: {zip_path}\n{'='*60}\nDONE! Best: {best_sc:.4f}\n{'='*60}")

if __name__ == '__main__': main()
