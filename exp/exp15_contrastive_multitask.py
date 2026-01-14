# -*- coding: utf-8 -*-
"""
================================================================================
Exp15: DeBERTa + Contrastive Learning + Multi-Task (SOTA Training)
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

🔥 INNOVATIONS:
1. DeBERTa-v3-large backbone
2. Multi-task: BCE + Contrastive loss (push correct options close, wrong options far)
3. Label smoothing for better calibration
4. Adversarial training (FGM) for robustness

GPU: H100 or 2x A100
Time: ~2-3 hours
Expected Score: 0.65+
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
        OUTPUT_DIR = Path('/kaggle/working/exp15_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp15_output')
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 384
    
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
    CONTRASTIVE_WEIGHT = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Adversarial training
    USE_FGM = True
    FGM_EPSILON = 0.5
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*70}")
print(f"Exp15: DeBERTa + Contrastive + Multi-Task")
print(f"{'='*70}")
print(f"BCE:{config.BCE_WEIGHT} + Contrastive:{config.CONTRASTIVE_WEIGHT}")
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

def get_context(ti, mx=1200):
    if not ti: return ""
    parts = [ti.get('topic','')]
    for d in ti.get('docs',[])[:3]:
        if s:=d.get('snippet',''): parts.append(s)
    return ' '.join(parts)[:mx]

# ============================================================================
# DATASET
# ============================================================================
class AERDataset(Dataset):
    def __init__(self, questions, docs, tokenizer, max_len=384, is_test=False):
        self.questions = questions
        self.docs = docs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        topic_info = self.docs.get(q.get('topic_id'), {})
        context = get_context(topic_info)
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
        
        # Classification head
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
        
        # Contrastive projection head
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, hidden_size),
            nn.ReLU(),
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
# CONTRASTIVE LOSS
# ============================================================================
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """
        embeddings: (batch, 4, hidden)
        labels: (batch, 4) binary
        Push correct options together, push wrong options away
        """
        batch_size = embeddings.size(0)
        loss = 0.0
        count = 0
        
        for i in range(batch_size):
            emb = F.normalize(embeddings[i], dim=-1)  # (4, hidden)
            lbl = labels[i]  # (4,)
            
            pos_mask = lbl == 1
            neg_mask = lbl == 0
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_emb = emb[pos_mask]  # (n_pos, hidden)
                neg_emb = emb[neg_mask]  # (n_neg, hidden)
                
                # Push positives together
                if pos_emb.size(0) > 1:
                    pos_sim = torch.mm(pos_emb, pos_emb.t()) / self.temperature
                    pos_loss = -torch.log(torch.exp(pos_sim).mean())
                    loss += pos_loss
                    count += 1
                
                # Push positives away from negatives
                for p in pos_emb:
                    neg_sim = torch.mm(p.unsqueeze(0), neg_emb.t()) / self.temperature
                    neg_loss = torch.log(1 + torch.exp(neg_sim).sum())
                    loss += neg_loss
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
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)
    
    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, loader, optimizer, scheduler, bce_criterion, contrastive_criterion, scaler, fgm=None):
    model.train()
    total_loss = 0
    
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
        pbar.set_postfix({'loss': f'{total_loss/(step+1):.4f}'})
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_ids = [], []
    
    for batch in tqdm(loader, desc="Evaluating"):
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
    for th in np.arange(0.25, 0.70, 0.05):
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
    
    print("\n[2/6] Preparing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_ds = AERDataset(train_q, train_docs, tokenizer, config.MAX_LENGTH)
    dev_ds = AERDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    test_ds = AERDataset(test_q, test_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE*2, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE*2, num_workers=2)
    
    print("\n[3/6] Creating model...")
    model = ContrastiveMultiTaskModel(config.MODEL_NAME).to(config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.WARMUP_RATIO), total_steps)
    
    bce_criterion = nn.BCEWithLogitsLoss()
    contrastive_criterion = ContrastiveLoss()
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
    zip_path = create_submission(test_preds, config.OUTPUT_DIR, 'exp15_submission')
    
    print(f"\n{'='*70}")
    print(f"🔥 DONE!")
    print(f"  Dev Score: {best_score:.4f}")
    print(f"  Test: {len(test_q)} questions")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
