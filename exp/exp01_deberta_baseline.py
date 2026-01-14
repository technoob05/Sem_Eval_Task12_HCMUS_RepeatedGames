# -*- coding: utf-8 -*-
"""
================================================================================
SemEval-2026 Task 12: Abductive Event Reasoning
Exp01: DeBERTa-v3-Large Baseline
================================================================================
SELF-CONTAINED SCRIPT - Copy this entire file into a single Kaggle notebook cell

GPU: H100 80GB recommended
Time: ~2-3 hours for full training
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
from collections import defaultdict
import shutil
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup
)

#%% ============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    # Paths - Auto-detect Kaggle
    IS_KAGGLE = Path('/kaggle/input').exists()
    
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp01_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp01_output')
    
    # Model
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 384
    
    # Training
    SEED = 42
    EPOCHS = 5
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    PATIENCE = 3
    
    # Multi-label
    THRESHOLD = 0.5
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"{'='*60}")
print(f"Exp01: DeBERTa-v3-Large Baseline")
print(f"{'='*60}")
print(f"Device: {config.DEVICE}")
print(f"Data: {config.DATA_DIR}")
print(f"Output: {config.OUTPUT_DIR}")
print(f"{'='*60}")

#%% ============================================================================
# SEED & UTILITIES
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(config.SEED)

def compute_aer_score(predictions, gold_labels):
    """Official AER evaluation metric."""
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

#%% ============================================================================
# DATA LOADING
# ==============================================================================
def load_questions(split):
    path = config.DATA_DIR / split / 'questions.jsonl'
    questions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    return questions

def load_docs(split):
    path = config.DATA_DIR / split / 'docs.json'
    with open(path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    return {d['topic_id']: d for d in docs}

def get_context(topic_info, max_docs=5, max_chars=1500):
    if not topic_info:
        return ""
    parts = [topic_info.get('topic', '')]
    for doc in topic_info.get('docs', [])[:max_docs]:
        snippet = doc.get('snippet', '')
        if snippet:
            parts.append(snippet)
    return ' '.join(parts)[:max_chars]

#%% ============================================================================
# DATASET
# ==============================================================================
class AERDataset(Dataset):
    OPTIONS = ['A', 'B', 'C', 'D']
    
    def __init__(self, questions, docs_dict, tokenizer, max_length=384, is_test=False):
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
        
        input_ids_list = []
        attention_mask_list = []
        
        for opt in self.OPTIONS:
            option_text = q.get(f'option_{opt}', '')
            text = f"{context} [SEP] {event} [SEP] {option_text}"
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids_list.append(encoding['input_ids'].squeeze(0))
            attention_mask_list.append(encoding['attention_mask'].squeeze(0))
        
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
                if ans in self.OPTIONS:
                    labels[self.OPTIONS.index(ans)] = 1.0
            result['labels'] = labels
        
        return result

#%% ============================================================================
# MODEL
# ==============================================================================
class DeBERTaClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        
        return logits.view(batch_size, 4)

#%% ============================================================================
# TRAINING FUNCTIONS
# ==============================================================================
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count

def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler):
    model.train()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        
        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        loss_meter.update(loss.item())
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    return loss_meter.avg, np.array(all_preds), np.array(all_labels)

@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels, all_ids = [], [], []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
        
        loss_meter.update(loss.item())
        all_preds.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_ids.extend(batch['id'])
    
    return loss_meter.avg, np.array(all_preds), np.array(all_labels), all_ids

@torch.no_grad()
def predict(model, dataloader):
    model.eval()
    all_preds, all_ids = [], []
    
    for batch in tqdm(dataloader, desc="Predicting"):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        
        with autocast():
            logits = model(input_ids, attention_mask)
        
        all_preds.extend(torch.sigmoid(logits).cpu().numpy())
        all_ids.extend(batch['id'])
    
    return np.array(all_preds), all_ids

#%% ============================================================================
# THRESHOLD OPTIMIZATION
# ==============================================================================
def optimize_threshold(probs, gold_labels, thresholds=None):
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    
    options = ['A', 'B', 'C', 'D']
    best_th, best_score = 0.5, 0
    
    for th in thresholds:
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

#%% ============================================================================
# SUBMISSION
# ==============================================================================
def create_submission(predictions, output_dir, name='submission'):
    submission_dir = output_dir / 'submission'
    submission_dir.mkdir(exist_ok=True)
    
    with open(submission_dir / 'submission.jsonl', 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    shutil.make_archive(str(output_dir / name), 'zip', submission_dir)
    return output_dir / f'{name}.zip'

#%% ============================================================================
# MAIN TRAINING
# ==============================================================================
def main():
    print("\n[1/5] Loading data...")
    train_questions = load_questions('train_data')
    dev_questions = load_questions('dev_data')
    train_docs = load_docs('train_data')
    dev_docs = load_docs('dev_data')
    print(f"  Train: {len(train_questions)}, Dev: {len(dev_questions)}")
    
    print("\n[2/5] Preparing tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    train_dataset = AERDataset(train_questions, train_docs, tokenizer, config.MAX_LENGTH)
    dev_dataset = AERDataset(dev_questions, dev_docs, tokenizer, config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE * 2, num_workers=2, pin_memory=True)
    
    print("\n[3/5] Creating model...")
    model = DeBERTaClassifier(config.MODEL_NAME)
    model.to(config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.WARMUP_RATIO), total_steps)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    print("\n[4/5] Training...")
    options = ['A', 'B', 'C', 'D']
    best_score, best_threshold = 0, 0.5
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        
        train_loss, _, _ = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler)
        print(f"Train Loss: {train_loss:.4f}")
        
        dev_loss, dev_preds, dev_labels, _ = evaluate(model, dev_loader, criterion)
        
        dev_gold = []
        for label in dev_labels:
            selected = [options[i] for i in range(4) if label[i] == 1]
            dev_gold.append(','.join(selected) if selected else 'A')
        
        best_th, best_sc = optimize_threshold(dev_preds, dev_gold)
        print(f"Dev Loss: {dev_loss:.4f}, Dev Score: {best_sc:.4f} (threshold={best_th})")
        
        if best_sc > best_score:
            best_score = best_sc
            best_threshold = best_th
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_score': best_score,
                'best_threshold': best_threshold
            }, config.OUTPUT_DIR / 'best_model.pt')
            print(f"  -> New best! Saved.")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("Early stopping!")
                break
    
    print(f"\nBest Dev Score: {best_score:.4f}")
    
    print("\n[5/5] Generating TEST submission...")
    checkpoint = torch.load(config.OUTPUT_DIR / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load and predict on TEST data
    test_questions = load_questions('test_data')
    test_docs = load_docs('test_data')
    print(f"  Test questions: {len(test_questions)}")
    
    test_dataset = AERDataset(test_questions, test_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2)
    
    test_probs, test_ids = predict(model, test_loader)
    test_preds = create_predictions(test_probs, test_ids, checkpoint['best_threshold'])
    
    zip_path = create_submission(test_preds, config.OUTPUT_DIR, 'exp01_submission')
    print(f"  Submission saved: {zip_path}")
    
    print("\n" + "="*60)
    print(f"DONE!")
    print(f"  Dev Score: {best_score:.4f}")
    print(f"  Test predictions: {len(test_questions)}")
    print(f"  Threshold: {checkpoint['best_threshold']}")
    print("="*60)

#%% ============================================================================
# RUN
# ==============================================================================
if __name__ == '__main__':
    main()
