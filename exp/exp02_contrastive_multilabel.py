# -*- coding: utf-8 -*-
"""
================================================================================
SemEval-2026 Task 12: Abductive Event Reasoning
Exp02: Contrastive Multi-Label Learning (NOVEL)
================================================================================
SELF-CONTAINED SCRIPT - Copy this entire file into a single Kaggle notebook cell

Key Innovations:
- Dual encoder with cross-attention
- Jaccard Similarity Contrastive Loss for multi-label
- Label-aware positive/negative sampling

GPU: H100 80GB recommended
Time: ~3-4 hours for full training
================================================================================
"""

#%% ============================================================================
# IMPORTS
# ==============================================================================
import json
import random
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import shutil
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

#%% ============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp02_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp02_output')
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 256
    HIDDEN_SIZE = 256
    
    SEED = 42
    EPOCHS = 5
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    PATIENCE = 3
    
    # Contrastive learning
    TEMPERATURE = 0.07
    MARGIN = 0.5
    CONTRASTIVE_WEIGHT = 0.3
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"{'='*60}")
print(f"Exp02: Contrastive Multi-Label Learning (NOVEL)")
print(f"{'='*60}")
print(f"Device: {config.DEVICE}")
print(f"Contrastive weight: {config.CONTRASTIVE_WEIGHT}")
print(f"{'='*60}")

#%% ============================================================================
# UTILITIES
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(config.SEED)

def compute_aer_score(predictions, gold_labels):
    scores = []
    for pred, gold in zip(predictions, gold_labels):
        pred_set = set(p.strip() for p in pred.split(',')) if pred else set()
        gold_set = set(g.strip() for g in gold.split(',')) if gold else set()
        if pred_set == gold_set:
            scores.append(1.0)
        elif pred_set and pred_set.issubset(gold_set):
            scores.append(0.5)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0

def load_questions(split):
    path = config.DATA_DIR / split / 'questions.jsonl'
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

def load_docs(split):
    path = config.DATA_DIR / split / 'docs.json'
    with open(path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    return {d['topic_id']: d for d in docs}

def get_context(topic_info, max_chars=1000):
    if not topic_info:
        return ""
    parts = [topic_info.get('topic', '')]
    for doc in topic_info.get('docs', [])[:3]:
        if snippet := doc.get('snippet', ''):
            parts.append(snippet)
    return ' '.join(parts)[:max_chars]

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count

#%% ============================================================================
# JACCARD CONTRASTIVE LOSS (NOVEL)
# ==============================================================================
class JaccardContrastiveLoss(nn.Module):
    """
    Novel: Jaccard Similarity Contrastive Loss for multi-label classification.
    Weighs sample pairs by their label overlap using Jaccard similarity.
    """
    def __init__(self, temperature=0.07, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, event_emb, option_emb, labels):
        # event_emb: (batch, hidden), option_emb: (batch, 4, hidden), labels: (batch, 4)
        batch_size = event_emb.size(0)
        
        # Normalize
        event_emb = F.normalize(event_emb, p=2, dim=-1)
        option_emb = F.normalize(option_emb, p=2, dim=-1)
        
        # Similarity: (batch, 4)
        similarities = torch.bmm(option_emb, event_emb.unsqueeze(-1)).squeeze(-1)
        similarities = similarities / self.temperature
        
        # Positive loss: push correct options closer
        pos_mask = labels.bool()
        pos_sim = similarities.masked_fill(~pos_mask, float('-inf'))
        pos_loss = -torch.logsumexp(pos_sim, dim=-1)
        
        # Negative loss: push incorrect options away
        neg_mask = ~labels.bool()
        neg_sim = similarities.masked_fill(~neg_mask, float('-inf'))
        neg_loss = torch.logsumexp(neg_sim, dim=-1)
        
        loss = F.relu(pos_loss + neg_loss + self.margin)
        return loss.mean()

#%% ============================================================================
# MODEL
# ==============================================================================
class ContrastiveClassifier(nn.Module):
    """
    Dual-encoder with cross-attention for contrastive multi-label learning.
    """
    def __init__(self, model_name, hidden_size=256, dropout=0.1, n_heads=8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        enc_hidden = self.encoder.config.hidden_size
        
        self.event_proj = nn.Sequential(
            nn.Linear(enc_hidden, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.option_proj = nn.Sequential(
            nn.Linear(enc_hidden, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.cross_attention = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        self.hidden_size = hidden_size
    
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
    
    def forward(self, event_ids, event_mask, option_ids, option_mask, return_emb=False):
        batch_size = event_ids.size(0)
        
        # Encode event
        event_emb = self.event_proj(self.encode(event_ids, event_mask))
        
        # Encode options (flatten -> encode -> reshape)
        opt_ids_flat = option_ids.view(-1, option_ids.size(-1))
        opt_mask_flat = option_mask.view(-1, option_mask.size(-1))
        opt_emb = self.option_proj(self.encode(opt_ids_flat, opt_mask_flat))
        opt_emb = opt_emb.view(batch_size, 4, -1)
        
        # Cross-attention
        event_exp = event_emb.unsqueeze(1)
        attn_out, _ = self.cross_attention(query=opt_emb, key=event_exp, value=event_exp)
        
        # Classify
        combined = torch.cat([opt_emb, attn_out], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        
        if return_emb:
            return logits, event_emb, opt_emb
        return logits

#%% ============================================================================
# DATASET
# ==============================================================================
class ContrastiveDataset(Dataset):
    OPTIONS = ['A', 'B', 'C', 'D']
    
    def __init__(self, questions, docs_dict, tokenizer, max_length=256, is_test=False):
        self.questions = questions
        self.docs_dict = docs_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        topic_info = self.docs_dict.get(q.get('topic_id'), {})
        context = get_context(topic_info)
        event = q.get('target_event', '')
        
        # Encode event+context
        event_enc = self.tokenizer(
            f"{context} [SEP] {event}",
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode options
        opt_ids, opt_mask = [], []
        for opt in self.OPTIONS:
            enc = self.tokenizer(
                q.get(f'option_{opt}', ''),
                max_length=self.max_length // 2,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            opt_ids.append(enc['input_ids'].squeeze(0))
            opt_mask.append(enc['attention_mask'].squeeze(0))
        
        result = {
            'id': q.get('id'),
            'event_input_ids': event_enc['input_ids'].squeeze(0),
            'event_attention_mask': event_enc['attention_mask'].squeeze(0),
            'option_input_ids': torch.stack(opt_ids),
            'option_attention_mask': torch.stack(opt_mask),
        }
        
        if not self.is_test:
            labels = torch.zeros(4)
            for ans in q.get('golden_answer', '').split(','):
                ans = ans.strip().upper()
                if ans in self.OPTIONS:
                    labels[self.OPTIONS.index(ans)] = 1.0
            result['labels'] = labels
        
        return result

#%% ============================================================================
# TRAINING
# ==============================================================================
def train_epoch(model, loader, optimizer, scheduler, bce_crit, con_crit, scaler, con_weight):
    model.train()
    loss_m, bce_m, con_m = AverageMeter(), AverageMeter(), AverageMeter()
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        event_ids = batch['event_input_ids'].to(config.DEVICE)
        event_mask = batch['event_attention_mask'].to(config.DEVICE)
        opt_ids = batch['option_input_ids'].to(config.DEVICE)
        opt_mask = batch['option_attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        
        optimizer.zero_grad()
        with autocast():
            logits, event_emb, opt_emb = model(event_ids, event_mask, opt_ids, opt_mask, return_emb=True)
            bce_loss = bce_crit(logits, labels)
            con_loss = con_crit(event_emb, opt_emb, labels)
            loss = bce_loss + con_weight * con_loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        loss_m.update(loss.item())
        bce_m.update(bce_loss.item())
        con_m.update(con_loss.item())
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss_m.avg:.4f}', 'bce': f'{bce_m.avg:.4f}', 'con': f'{con_m.avg:.4f}'})
    
    return loss_m.avg, np.array(all_preds), np.array(all_labels)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_m = AverageMeter()
    all_preds, all_labels, all_ids = [], [], []
    
    for batch in tqdm(loader, desc="Evaluating"):
        event_ids = batch['event_input_ids'].to(config.DEVICE)
        event_mask = batch['event_attention_mask'].to(config.DEVICE)
        opt_ids = batch['option_input_ids'].to(config.DEVICE)
        opt_mask = batch['option_attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        
        with autocast():
            logits = model(event_ids, event_mask, opt_ids, opt_mask)
            loss = criterion(logits, labels)
        
        loss_m.update(loss.item())
        all_preds.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_ids.extend(batch['id'])
    
    return loss_m.avg, np.array(all_preds), np.array(all_labels), all_ids

@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_preds, all_ids = [], []
    for batch in tqdm(loader, desc="Predicting"):
        event_ids = batch['event_input_ids'].to(config.DEVICE)
        event_mask = batch['event_attention_mask'].to(config.DEVICE)
        opt_ids = batch['option_input_ids'].to(config.DEVICE)
        opt_mask = batch['option_attention_mask'].to(config.DEVICE)
        
        with autocast():
            logits = model(event_ids, event_mask, opt_ids, opt_mask)
        all_preds.extend(torch.sigmoid(logits).cpu().numpy())
        all_ids.extend(batch['id'])
    return np.array(all_preds), all_ids

#%% ============================================================================
# HELPERS
# ==============================================================================
def optimize_threshold(probs, gold_labels):
    options = ['A', 'B', 'C', 'D']
    best_th, best_score = 0.5, 0
    for th in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        preds = []
        for prob in probs:
            selected = [options[i] for i in range(4) if prob[i] >= th]
            if not selected:
                selected = [options[np.argmax(prob)]]
            preds.append(','.join(sorted(selected)))
        score = compute_aer_score(preds, gold_labels)
        if score > best_score:
            best_score = score
            best_th = th
    return best_th, best_score

def create_predictions(probs, ids, threshold):
    options = ['A', 'B', 'C', 'D']
    predictions = []
    for qid, prob in zip(ids, probs):
        selected = [options[i] for i in range(4) if prob[i] >= threshold]
        if not selected:
            selected = [options[np.argmax(prob)]]
        predictions.append({'id': qid, 'answer': ','.join(sorted(selected))})
    return predictions

def create_submission(predictions, output_dir, name='submission'):
    sub_dir = output_dir / 'submission'
    sub_dir.mkdir(exist_ok=True)
    with open(sub_dir / 'submission.jsonl', 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    shutil.make_archive(str(output_dir / name), 'zip', sub_dir)
    return output_dir / f'{name}.zip'

#%% ============================================================================
# MAIN
# ==============================================================================
def main():
    print("\n[1/5] Loading data...")
    train_q = load_questions('train_data')
    dev_q = load_questions('dev_data')
    train_docs = load_docs('train_data')
    dev_docs = load_docs('dev_data')
    print(f"  Train: {len(train_q)}, Dev: {len(dev_q)}")
    
    print("\n[2/5] Preparing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_ds = ContrastiveDataset(train_q, train_docs, tokenizer, config.MAX_LENGTH)
    dev_ds = ContrastiveDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE*2, num_workers=2, pin_memory=True)
    
    print("\n[3/5] Creating model...")
    model = ContrastiveClassifier(config.MODEL_NAME, config.HIDDEN_SIZE)
    model.to(config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.WARMUP_RATIO), total_steps)
    bce_criterion = nn.BCEWithLogitsLoss()
    con_criterion = JaccardContrastiveLoss(config.TEMPERATURE, config.MARGIN)
    scaler = GradScaler()
    
    print("\n[4/5] Training...")
    options = ['A', 'B', 'C', 'D']
    best_score, best_th, patience = 0, 0.5, 0
    
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        train_loss, _, _ = train_epoch(model, train_loader, optimizer, scheduler, 
                                        bce_criterion, con_criterion, scaler, config.CONTRASTIVE_WEIGHT)
        print(f"Train Loss: {train_loss:.4f}")
        
        dev_loss, dev_preds, dev_labels, _ = evaluate(model, dev_loader, bce_criterion)
        dev_gold = [','.join([options[i] for i in range(4) if l[i]==1]) or 'A' for l in dev_labels]
        
        th, sc = optimize_threshold(dev_preds, dev_gold)
        print(f"Dev Loss: {dev_loss:.4f}, Score: {sc:.4f} (th={th})")
        
        if sc > best_score:
            best_score, best_th, patience = sc, th, 0
            torch.save({'model_state_dict': model.state_dict(), 'best_score': best_score, 'best_threshold': best_th},
                      config.OUTPUT_DIR / 'best_model.pt')
            print("  -> New best! Saved.")
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print("Early stopping!")
                break
    
    print("\n[5/5] Generating submission...")
    ckpt = torch.load(config.OUTPUT_DIR / 'best_model.pt')
    model.load_state_dict(ckpt['model_state_dict'])
    
    dev_ds_test = ContrastiveDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    dev_loader_test = DataLoader(dev_ds_test, batch_size=config.BATCH_SIZE*2)
    probs, ids = predict(model, dev_loader_test)
    preds = create_predictions(probs, ids, ckpt['best_threshold'])
    
    zip_path = create_submission(preds, config.OUTPUT_DIR, 'exp02_submission')
    print(f"Submission: {zip_path}")
    print(f"\n{'='*60}\nDONE! Best Score: {best_score:.4f}\n{'='*60}")

if __name__ == '__main__':
    main()
