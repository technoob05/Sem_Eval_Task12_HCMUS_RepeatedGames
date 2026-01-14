# -*- coding: utf-8 -*-
"""
================================================================================
SemEval-2026 Task 12: Abductive Event Reasoning
Exp05: Longformer with Full Document Attention (NOVEL)
================================================================================
SELF-CONTAINED SCRIPT - Copy this entire file into a single Kaggle notebook cell

Key Innovations:
- Full document processing (4096 tokens)
- Global attention on event and option tokens
- Local attention on document content

GPU: H100 80GB recommended
Time: ~4-5 hours
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
from transformers import AutoTokenizer, LongformerModel, LongformerConfig, get_linear_schedule_with_warmup

#%% CONFIG
class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp05_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp05_output')
    
    MODEL_NAME = 'allenai/longformer-base-4096'
    MAX_LENGTH = 4096
    MAX_DOCS = 5
    ATTENTION_WINDOW = 256
    
    SEED = 42; EPOCHS = 5; BATCH_SIZE = 2; LR = 1e-5; PATIENCE = 3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*60}\nExp05: Longformer Full Documents (NOVEL)\n{'='*60}\nDevice: {config.DEVICE}\nMax Length: {config.MAX_LENGTH}\n{'='*60}")

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

def get_full_docs(topic_info, max_docs=5, max_chars=8000):
    if not topic_info: return ""
    parts = [f"Topic: {topic_info.get('topic', '')}"]
    for i, doc in enumerate(topic_info.get('docs', [])[:max_docs]):
        title = doc.get('title', '')
        content = doc.get('content', doc.get('snippet', ''))[:1500]
        parts.append(f"[Doc{i+1}] {title}: {content}")
    return '\n'.join(parts)[:max_chars]

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=self.avg=self.sum=self.count=0
    def update(self,v,n=1): self.val=v; self.sum+=v*n; self.count+=n; self.avg=self.sum/self.count

#%% DATASET
class LongformDataset(Dataset):
    OPTIONS = ['A', 'B', 'C', 'D']
    
    def __init__(self, questions, docs_dict, tokenizer, max_len=4096, is_test=False):
        self.questions = questions
        self.docs_dict = docs_dict
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
    
    def __len__(self): return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        qid = q.get('id')
        event = q.get('target_event', '')
        topic_info = self.docs_dict.get(q.get('topic_id'), {})
        full_docs = get_full_docs(topic_info, config.MAX_DOCS)
        
        input_ids, attention_mask, global_attention = [], [], []
        
        for opt in self.OPTIONS:
            option_text = q.get(f'option_{opt}', '')
            text = f"<s>Event: {event} Option: {option_text}</s> Documents: {full_docs}"
            
            enc = self.tokenizer(text, max_length=self.max_len, padding='max_length',
                                 truncation=True, return_tensors='pt')
            
            ids = enc['input_ids'].squeeze(0)
            mask = enc['attention_mask'].squeeze(0)
            
            # Global attention on first 50 tokens (event + option)
            glob = torch.zeros_like(mask)
            glob[:min(50, mask.sum().item())] = 1
            glob[0] = 1  # CLS always global
            
            input_ids.append(ids)
            attention_mask.append(mask)
            global_attention.append(glob)
        
        result = {
            'id': qid,
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'global_attention_mask': torch.stack(global_attention)
        }
        
        if not self.is_test:
            labels = torch.zeros(4)
            for a in q.get('golden_answer', '').split(','):
                a = a.strip().upper()
                if a in self.OPTIONS: labels[self.OPTIONS.index(a)] = 1.0
            result['labels'] = labels
        return result

#%% MODEL
class LongformerClassifier(nn.Module):
    def __init__(self, model_name, attention_window=256, dropout=0.1):
        super().__init__()
        self.config = LongformerConfig.from_pretrained(model_name)
        self.config.attention_window = [attention_window] * self.config.num_hidden_layers
        self.longformer = LongformerModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, global_attention_mask):
        bs = input_ids.size(0)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        global_attention_mask = global_attention_mask.view(-1, global_attention_mask.size(-1))
        
        out = self.longformer(input_ids, attention_mask, global_attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.classifier(cls).squeeze(-1).view(bs, 4)

#%% TRAINING
def train_epoch(model, loader, optimizer, scheduler, criterion, scaler):
    model.train()
    loss_m = AverageMeter()
    preds_all, labels_all = [], []
    
    for batch in tqdm(loader, desc="Training"):
        ids = batch['input_ids'].to(config.DEVICE)
        mask = batch['attention_mask'].to(config.DEVICE)
        glob = batch['global_attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        
        optimizer.zero_grad()
        with autocast():
            logits = model(ids, mask, glob)
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
        glob = batch['global_attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        with autocast():
            logits = model(ids, mask, glob)
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
        glob = batch['global_attention_mask'].to(config.DEVICE)
        with autocast():
            logits = model(ids, mask, glob)
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
    
    print("\n[2/5] Preparing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_ds = LongformDataset(train_q, train_docs, tokenizer, config.MAX_LENGTH)
    dev_ds = LongformDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, num_workers=2)
    
    print("\n[3/5] Creating model...")
    model = LongformerClassifier(config.MODEL_NAME, config.ATTENTION_WINDOW).to(config.DEVICE)
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
    dev_ds_test = LongformDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    probs, ids = predict(model, DataLoader(dev_ds_test, batch_size=config.BATCH_SIZE))
    preds = create_predictions(probs, ids, ckpt['th'])
    zip_path = create_submission(preds, config.OUTPUT_DIR, 'exp05_submission')
    print(f"Submission: {zip_path}\n{'='*60}\nDONE! Best: {best_sc:.4f}\n{'='*60}")

if __name__ == '__main__': main()
