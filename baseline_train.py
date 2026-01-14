# -*- coding: utf-8 -*-
"""
SemEval-2026 Task 12: Abductive Event Reasoning
Baseline Model: BERT-based Multiple Choice Classification
============================================================
This baseline uses BERT to classify each option independently.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Paths
    BASE_DIR = Path(r"d:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning")
    DATA_DIR = BASE_DIR / "semeval2026-task12-dataset"
    OUTPUT_DIR = BASE_DIR / "baseline_output"
    
    # Model
    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 256
    
    # Training
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Threshold for multi-label prediction
    THRESHOLD = 0.5

config = Config()
config.OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Device: {config.DEVICE}")

# ============================================================================
# Data Loading
# ============================================================================
def load_questions(split):
    path = config.DATA_DIR / split / "questions.jsonl"
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_docs(split):
    path = config.DATA_DIR / split / "docs.json"
    with open(path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    # Create topic_id -> docs mapping
    return {d['topic_id']: d for d in docs}

# ============================================================================
# Dataset
# ============================================================================
class AERDataset(Dataset):
    def __init__(self, questions, docs_dict, tokenizer, max_length=256, is_test=False):
        self.questions = questions
        self.docs_dict = docs_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.options = ['A', 'B', 'C', 'D']
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        
        # Get context (use topic description as short context)
        topic_id = q.get('topic_id')
        topic_info = self.docs_dict.get(topic_id, {})
        context = topic_info.get('topic', '')
        
        # Event text
        event = q.get('target_event', '')
        
        # Prepare input for each option
        input_ids_list = []
        attention_mask_list = []
        
        for opt in self.options:
            option_text = q.get(f'option_{opt}', '')
            
            # Format: [CLS] context [SEP] event [SEP] option [SEP]
            text = f"Context: {context} Event: {event} Cause: {option_text}"
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids_list.append(encoding['input_ids'].squeeze(0))
            attention_mask_list.append(encoding['attention_mask'].squeeze(0))
        
        input_ids = torch.stack(input_ids_list)  # (4, max_length)
        attention_mask = torch.stack(attention_mask_list)  # (4, max_length)
        
        if self.is_test:
            return {
                'id': q.get('id'),
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            # Parse golden answer
            golden = q.get('golden_answer', '')
            labels = torch.zeros(4)
            for ans in golden.split(','):
                ans = ans.strip().upper()
                if ans in self.options:
                    labels[self.options.index(ans)] = 1.0
            
            return {
                'id': q.get('id'),
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

# ============================================================================
# Model
# ============================================================================
class AERClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        # input_ids: (batch, 4, seq_len)
        batch_size = input_ids.size(0)
        
        # Flatten for BERT
        input_ids = input_ids.view(-1, input_ids.size(-1))  # (batch*4, seq_len)
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch*4, hidden)
        
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # (batch*4, 1)
        
        logits = logits.view(batch_size, 4)  # (batch, 4)
        return logits

# ============================================================================
# Training Functions
# ============================================================================
def compute_score(preds, labels, threshold=0.5):
    """Compute competition metric"""
    scores = []
    for pred, label in zip(preds, labels):
        pred_set = set(np.where(pred >= threshold)[0])
        gold_set = set(np.where(label == 1)[0])
        
        if pred_set == gold_set:
            scores.append(1.0)
        elif pred_set and pred_set.issubset(gold_set):
            scores.append(0.5)
        else:
            scores.append(0.0)
    
    return np.mean(scores)

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    score = compute_score(all_preds, all_labels)
    
    return avg_loss, score

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    score = compute_score(all_preds, all_labels)
    
    return avg_loss, score, all_preds

def predict(model, dataloader, device, threshold=0.5):
    model.eval()
    predictions = []
    options = ['A', 'B', 'C', 'D']
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            ids = batch['id']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            for i, (qid, prob) in enumerate(zip(ids, probs)):
                selected = [options[j] for j in range(4) if prob[j] >= threshold]
                if not selected:
                    # If nothing selected, pick the highest
                    selected = [options[np.argmax(prob)]]
                
                predictions.append({
                    'id': qid,
                    'answer': ','.join(sorted(selected))
                })
    
    return predictions

# ============================================================================
# Main Training Loop
# ============================================================================
def main():
    print("=" * 60)
    print("SemEval-2026 Task 12: Baseline Training")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_questions = load_questions('train_data')
    dev_questions = load_questions('dev_data')
    train_docs = load_docs('train_data')
    dev_docs = load_docs('dev_data')
    
    print(f"Train: {len(train_questions)} questions")
    print(f"Dev: {len(dev_questions)} questions")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create datasets
    train_dataset = AERDataset(train_questions, train_docs, tokenizer, config.MAX_LENGTH)
    dev_dataset = AERDataset(dev_questions, dev_docs, tokenizer, config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE)
    
    # Create model
    print(f"\nCreating model...")
    model = AERClassifier(config.MODEL_NAME)
    model.to(config.DEVICE)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Training
    print(f"\nStarting training for {config.EPOCHS} epochs...")
    best_score = 0
    
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        
        train_loss, train_score = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, config.DEVICE
        )
        print(f"Train Loss: {train_loss:.4f}, Train Score: {train_score:.4f}")
        
        dev_loss, dev_score, _ = eval_epoch(model, dev_loader, criterion, config.DEVICE)
        print(f"Dev Loss: {dev_loss:.4f}, Dev Score: {dev_score:.4f}")
        
        if dev_score > best_score:
            best_score = dev_score
            torch.save(model.state_dict(), config.OUTPUT_DIR / "best_model.pt")
            print(f"  -> New best model saved! Score: {dev_score:.4f}")
    
    print(f"\nBest Dev Score: {best_score:.4f}")
    
    # Load best model and generate submission
    print("\n" + "=" * 60)
    print("Generating Dev Submission...")
    print("=" * 60)
    
    model.load_state_dict(torch.load(config.OUTPUT_DIR / "best_model.pt"))
    
    # Create submission for dev set
    dev_dataset_test = AERDataset(dev_questions, dev_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    dev_loader_test = DataLoader(dev_dataset_test, batch_size=config.BATCH_SIZE)
    
    predictions = predict(model, dev_loader_test, config.DEVICE, threshold=config.THRESHOLD)
    
    # Save submission
    submission_dir = config.OUTPUT_DIR / "submission"
    submission_dir.mkdir(exist_ok=True)
    
    submission_path = submission_dir / "submission.jsonl"
    with open(submission_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    print(f"Saved {len(predictions)} predictions to {submission_path}")
    
    # Create zip
    import shutil
    zip_path = config.OUTPUT_DIR / "submission"
    shutil.make_archive(str(zip_path), 'zip', submission_dir)
    
    print(f"\nSubmission zip created: {config.OUTPUT_DIR / 'submission.zip'}")
    print("\nDone! Upload submission.zip to Codabench.")

if __name__ == "__main__":
    main()
