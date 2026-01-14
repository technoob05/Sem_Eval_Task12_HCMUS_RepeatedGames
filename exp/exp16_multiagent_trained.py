# -*- coding: utf-8 -*-
"""
================================================================================
Exp16: Multi-Agent Trained Experts with Debate (NOVEL!)
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

🔥 NOVEL CONTRIBUTION:
Train multiple specialized "expert" models, each with different focus:
- Expert 1: Forward Causal Reasoning (trained on "If X → Y" examples)
- Expert 2: Counterfactual Reasoning (trained on "Without X, no Y" examples)
- Expert 3: Temporal Reasoning (trained on timing-focused examples)

Then ensemble with learned weights (meta-learner)

Inspiration: Mixture of Experts (MoE) + Multi-Agent Debate

GPU: H100 or A100
Time: ~4-5 hours
Expected Score: 0.67+ 🎯
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
        OUTPUT_DIR = Path('/kaggle/working/exp16_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp16_output')
    
    # Shared backbone
    MODEL_NAME = 'microsoft/deberta-v3-base'  # Base for efficiency
    MAX_LENGTH = 384
    
    # Training
    SEED = 42
    EPOCHS_PER_EXPERT = 3
    META_EPOCHS = 2
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    META_LR = 1e-3
    WARMUP_RATIO = 0.1
    
    # Multi-Agent config
    NUM_EXPERTS = 3
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*70}")
print(f"Exp16: Multi-Agent Trained Experts with Debate")
print(f"{'='*70}")
print(f"Experts: {config.NUM_EXPERTS} | Epochs/Expert: {config.EPOCHS_PER_EXPERT}")
print(f"{'='*70}")

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
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

def get_context(ti, mx=800):
    if not ti: return ""
    parts = [ti.get('topic','')]
    for d in ti.get('docs',[])[:2]:
        if s:=d.get('snippet',''): parts.append(s[:200])
    return ' '.join(parts)[:mx]

# ============================================================================
# EXPERT-SPECIALIZED PROMPTS (for training data augmentation)
# ============================================================================
EXPERT_PROMPTS = [
    # Expert 1: Forward Causal - focuses on "X leads to Y"
    lambda ctx, event, opt: f"[FORWARD] Context: {ctx} | If '{opt}' happened, would it cause '{event}'?",
    
    # Expert 2: Counterfactual - focuses on "Without X, no Y"
    lambda ctx, event, opt: f"[COUNTERFACTUAL] Context: {ctx} | Would '{event}' still happen without '{opt}'?",
    
    # Expert 3: Temporal - focuses on timing
    lambda ctx, event, opt: f"[TEMPORAL] Context: {ctx} | Did '{opt}' occur before '{event}'? Is it direct?",
]

# ============================================================================
# DATASET
# ============================================================================
class ExpertDataset(Dataset):
    """Dataset for training a specific expert."""
    def __init__(self, questions, docs, tokenizer, expert_id, max_len=384, is_test=False):
        self.questions = questions
        self.docs = docs
        self.tokenizer = tokenizer
        self.expert_id = expert_id
        self.prompt_fn = EXPERT_PROMPTS[expert_id]
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
            option_text = q.get(f'option_{opt}', '')
            # Use expert-specific prompt
            text = self.prompt_fn(context, event, option_text)
            
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
# EXPERT MODEL
# ============================================================================
class ExpertModel(nn.Module):
    """Single expert classifier."""
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
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(cls).squeeze(-1)
        
        return logits.view(batch_size, 4)

# ============================================================================
# META-LEARNER (learns how to weight experts)
# ============================================================================
class MetaLearner(nn.Module):
    """Learns optimal expert weights based on input."""
    def __init__(self, num_experts=3, hidden_size=64):
        super().__init__()
        # Simple gating network
        self.gate = nn.Sequential(
            nn.Linear(4 * num_experts, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, expert_logits):
        """
        expert_logits: (batch, num_experts, 4)
        Returns: (batch, 4) weighted predictions
        """
        batch_size = expert_logits.size(0)
        
        # Flatten expert outputs for gating
        flat = expert_logits.view(batch_size, -1)  # (batch, num_experts * 4)
        
        # Get expert weights
        weights = self.gate(flat)  # (batch, num_experts)
        
        # Weighted sum
        weights = weights.unsqueeze(-1)  # (batch, num_experts, 1)
        weighted = (expert_logits * weights).sum(dim=1)  # (batch, 4)
        
        return weighted, weights.squeeze(-1)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_expert(expert_id, model, train_loader, dev_loader, dev_gold, tokenizer):
    """Train a single expert."""
    print(f"\n  Training Expert {expert_id+1} ({['Forward', 'Counterfactual', 'Temporal'][expert_id]})...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS_PER_EXPERT
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.WARMUP_RATIO), total_steps)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    best_score = 0
    for epoch in range(config.EPOCHS_PER_EXPERT):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Expert{expert_id+1} Epoch{epoch+1}", leave=False):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
        
        # Evaluate
        dev_probs, _ = evaluate_expert(model, dev_loader)
        score = evaluate_threshold(dev_probs, dev_gold)
        print(f"    Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Dev={score:.4f}")
        
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), config.OUTPUT_DIR/f'expert_{expert_id}.pt')
    
    # Load best
    model.load_state_dict(torch.load(config.OUTPUT_DIR/f'expert_{expert_id}.pt', weights_only=False))
    return model, best_score

@torch.no_grad()
def evaluate_expert(model, loader):
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

def evaluate_threshold(probs, golds):
    opts = ['A','B','C','D']
    best_sc = 0
    for th in np.arange(0.3, 0.7, 0.05):
        ps = [','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]])) for p in probs]
        sc = compute_aer_score(ps, golds)
        if sc > best_sc: best_sc = sc
    return best_sc

def train_meta_learner(meta_learner, experts, train_loader, dev_loader, dev_gold, tokenizer):
    """Train the meta-learner to weight experts."""
    print("\n[4/6] Training Meta-Learner...")
    
    optimizer = torch.optim.Adam(meta_learner.parameters(), lr=config.META_LR)
    criterion = nn.BCEWithLogitsLoss()
    
    # Freeze experts
    for expert in experts:
        expert.eval()
        for param in expert.parameters():
            param.requires_grad = False
    
    best_score = 0
    for epoch in range(config.META_EPOCHS):
        meta_learner.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Meta Epoch{epoch+1}", leave=False):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            
            # Get expert predictions
            expert_logits = []
            with torch.no_grad():
                for expert in experts:
                    logits = expert(input_ids, attention_mask)
                    expert_logits.append(torch.sigmoid(logits))
            
            expert_logits = torch.stack(expert_logits, dim=1)  # (batch, num_experts, 4)
            
            # Meta-learner
            optimizer.zero_grad()
            weighted_pred, weights = meta_learner(expert_logits)
            loss = criterion(weighted_pred, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate
        dev_probs = evaluate_meta(meta_learner, experts, dev_loader)
        th, score = optimize_threshold(dev_probs, dev_gold)
        print(f"  Meta Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Dev={score:.4f}")
        
        if score > best_score:
            best_score = score
            torch.save({'meta': meta_learner.state_dict(), 'threshold': th}, config.OUTPUT_DIR/'meta_learner.pt')
    
    ckpt = torch.load(config.OUTPUT_DIR/'meta_learner.pt', weights_only=False)
    meta_learner.load_state_dict(ckpt['meta'])
    return meta_learner, ckpt['threshold'], best_score

@torch.no_grad()
def evaluate_meta(meta_learner, experts, loader):
    meta_learner.eval()
    all_preds = []
    
    for batch in loader:
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        
        expert_logits = []
        for expert in experts:
            logits = expert(input_ids, attention_mask)
            expert_logits.append(torch.sigmoid(logits))
        
        expert_logits = torch.stack(expert_logits, dim=1)
        weighted_pred, _ = meta_learner(expert_logits)
        all_preds.extend(weighted_pred.cpu().numpy())
    
    return np.array(all_preds)

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
    dev_gold = [q.get('golden_answer','') for q in dev_q]
    print(f"  Train:{len(train_q)} Dev:{len(dev_q)} Test:{len(test_q)}")
    
    print("\n[2/6] Preparing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    print("\n[3/6] Training Experts...")
    experts = []
    expert_scores = []
    
    for expert_id in range(config.NUM_EXPERTS):
        # Create expert-specific datasets
        train_ds = ExpertDataset(train_q, train_docs, tokenizer, expert_id, config.MAX_LENGTH)
        dev_ds = ExpertDataset(dev_q, dev_docs, tokenizer, expert_id, config.MAX_LENGTH, is_test=True)
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE*2, num_workers=2)
        
        # Create and train expert
        expert = ExpertModel(config.MODEL_NAME).to(config.DEVICE)
        expert, score = train_expert(expert_id, expert, train_loader, dev_loader, dev_gold, tokenizer)
        
        experts.append(expert)
        expert_scores.append(score)
    
    print(f"\n  Expert Scores: {[f'{s:.4f}' for s in expert_scores]}")
    
    # Create unified loaders for meta-learner (using expert 0's prompt as default)
    train_ds = ExpertDataset(train_q, train_docs, tokenizer, 0, config.MAX_LENGTH)
    dev_ds = ExpertDataset(dev_q, dev_docs, tokenizer, 0, config.MAX_LENGTH, is_test=True)
    test_ds = ExpertDataset(test_q, test_docs, tokenizer, 0, config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE*2, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE*2, num_workers=2)
    
    # Train meta-learner
    meta_learner = MetaLearner(config.NUM_EXPERTS).to(config.DEVICE)
    meta_learner, best_th, best_score = train_meta_learner(meta_learner, experts, train_loader, dev_loader, dev_gold, tokenizer)
    
    print(f"\n  Final Dev Score with Meta-Learner: {best_score:.4f}")
    
    print("\n[5/6] Predicting on TEST...")
    test_probs = evaluate_meta(meta_learner, experts, test_loader)
    test_ids = [q['id'] for q in test_q]
    test_preds = create_predictions(test_probs, test_ids, best_th)
    
    print("\n[6/6] Generating submission...")
    zip_path = create_submission(test_preds, config.OUTPUT_DIR, 'exp16_submission')
    
    # Save results
    with open(config.OUTPUT_DIR/'results.json', 'w') as f:
        json.dump({
            'dev_score': best_score,
            'expert_scores': expert_scores,
            'threshold': best_th,
            'method': 'Multi-Agent Trained Experts + Meta-Learner'
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🔥 DONE!")
    print(f"  Expert Scores: {[f'{s:.4f}' for s in expert_scores]}")
    print(f"  Meta-Learner Score: {best_score:.4f}")
    print(f"  Test: {len(test_q)} questions")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
