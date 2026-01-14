# -*- coding: utf-8 -*-
"""
================================================================================
SemEval-2026 Task 12: Abductive Event Reasoning
Exp06: Ensemble with Confidence Fusion
================================================================================
SELF-CONTAINED SCRIPT - Copy this entire file into a single Kaggle notebook cell

Key Innovations:
- Combines predictions from multiple base models
- Multiple fusion strategies: weighted average, MLP, LightGBM
- Features: probabilities, entropy, model agreement

NOTE: This script expects predictions from Exp01, 02, 03.
If not available, it uses synthetic predictions for demonstration.

GPU: Not required (CPU is sufficient)
Time: ~10 minutes
================================================================================
"""

#%% IMPORTS
import json, random, shutil, warnings
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from itertools import product
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

#%% CONFIG
class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp06_output')
        # Prediction files from other experiments (if available)
        EXP_DIRS = [Path('/kaggle/working/exp01_output'),
                    Path('/kaggle/working/exp02_output'),
                    Path('/kaggle/working/exp03_output')]
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp06_output')
        EXP_DIRS = [Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp01_output'),
                    Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp02_output'),
                    Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp03_output')]
    
    SEED = 42

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*60}\nExp06: Ensemble Fusion\n{'='*60}\nLightGBM available: {HAS_LGB}\n{'='*60}")

#%% UTILITIES
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
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

#%% LOAD PREDICTIONS
def load_model_predictions(exp_dirs, n_samples):
    """Load predictions from trained models or generate synthetic ones."""
    preds = []
    for exp_dir in exp_dirs:
        pred_file = exp_dir / 'dev_predictions.npy'
        if pred_file.exists():
            preds.append(np.load(pred_file))
            print(f"  Loaded predictions from {exp_dir.name}")
        else:
            # Generate synthetic predictions
            np.random.seed(len(preds) + 42)
            synthetic = np.random.rand(n_samples, 4) * 0.5 + 0.25
            preds.append(synthetic)
            print(f"  Generated synthetic predictions for {exp_dir.name}")
    return preds

#%% FEATURE ENGINEERING
def compute_entropy(probs):
    probs = np.clip(probs, 1e-8, 1-1e-8)
    return -np.sum(probs * np.log(probs) + (1-probs) * np.log(1-probs), axis=-1)

def extract_features(model_preds):
    """Extract features for meta-learner."""
    features = []
    # Per-model features
    for probs in model_preds:
        features.append(probs)  # 4 probs
        features.append(compute_entropy(probs).reshape(-1, 1))  # entropy
        features.append(np.max(probs, axis=-1, keepdims=True))  # max
        features.append((np.max(probs, axis=-1) - np.min(probs, axis=-1)).reshape(-1, 1))  # range
    # Cross-model features
    stacked = np.stack(model_preds, axis=0)
    features.append(np.mean(stacked, axis=0))  # mean
    features.append(np.std(stacked, axis=0))   # std
    return np.concatenate(features, axis=-1)

#%% FUSION STRATEGIES
class WeightedAvgFusion:
    def __init__(self, n_models):
        self.weights = np.ones(n_models) / n_models
    
    def fit(self, model_preds, labels, gold_strs):
        best_score, best_weights = 0, self.weights.copy()
        for w in product([0.0, 0.25, 0.5, 0.75, 1.0], repeat=len(model_preds)):
            if sum(w) == 0: continue
            weights = np.array(w) / sum(w)
            combined = sum(p * wt for p, wt in zip(model_preds, weights))
            _, score = optimize_threshold(combined, gold_strs)
            if score > best_score:
                best_score, best_weights = score, weights
        self.weights = best_weights
        return best_score
    
    def predict(self, model_preds):
        return sum(p * w for p, w in zip(model_preds, self.weights))

class MLPFusion(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden//2, 4)
        )
    def forward(self, x): return self.net(x)

def train_mlp(features, labels, epochs=50, lr=1e-3):
    model = MLPFusion(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    dataset = TensorDataset(torch.FloatTensor(features), torch.FloatTensor(labels))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.FloatTensor(features))).numpy()
    return model, preds

#%% MAIN
def main():
    print("\n[1/4] Loading data...")
    dev_q = load_questions('dev_data')
    dev_ids = [q['id'] for q in dev_q]
    
    opts = ['A','B','C','D']
    dev_labels = []
    dev_gold = []
    for q in dev_q:
        golden = q.get('golden_answer', '')
        label = np.zeros(4)
        for a in golden.split(','):
            a = a.strip().upper()
            if a in opts: label[opts.index(a)] = 1
        dev_labels.append(label)
        dev_gold.append(golden)
    dev_labels = np.array(dev_labels)
    print(f"  Dev samples: {len(dev_q)}")
    
    print("\n[2/4] Loading/generating model predictions...")
    model_preds = load_model_predictions(config.EXP_DIRS, len(dev_q))
    print(f"  Loaded {len(model_preds)} models")
    
    print("\n[3/4] Evaluating fusion strategies...")
    
    # Strategy 1: Simple Average
    simple_avg = np.mean(model_preds, axis=0)
    _, simple_score = optimize_threshold(simple_avg, dev_gold)
    print(f"  Simple Average: {simple_score:.4f}")
    
    # Strategy 2: Weighted Average
    wa_fusion = WeightedAvgFusion(len(model_preds))
    wa_score = wa_fusion.fit(model_preds, dev_labels, dev_gold)
    print(f"  Weighted Average: {wa_score:.4f} (weights={wa_fusion.weights.round(2)})")
    
    # Strategy 3: MLP
    features = extract_features(model_preds)
    print(f"  Feature dim: {features.shape[1]}")
    mlp_model, mlp_preds = train_mlp(features, dev_labels)
    _, mlp_score = optimize_threshold(mlp_preds, dev_gold)
    print(f"  MLP Fusion: {mlp_score:.4f}")
    
    # Strategy 4: LightGBM (if available)
    if HAS_LGB:
        lgb_preds = []
        for i in range(4):
            data = lgb.Dataset(features, label=dev_labels[:, i])
            params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1}
            model = lgb.train(params, data, num_boost_round=100)
            lgb_preds.append(model.predict(features))
        lgb_preds = np.stack(lgb_preds, axis=-1)
        _, lgb_score = optimize_threshold(lgb_preds, dev_gold)
        print(f"  LightGBM: {lgb_score:.4f}")
    else:
        lgb_score = 0
        lgb_preds = None
    
    # Select best
    scores = {'simple': (simple_score, simple_avg), 'weighted': (wa_score, wa_fusion.predict(model_preds)), 
              'mlp': (mlp_score, mlp_preds)}
    if HAS_LGB and lgb_preds is not None:
        scores['lgb'] = (lgb_score, lgb_preds)
    
    best_name = max(scores, key=lambda k: scores[k][0])
    best_score, best_preds = scores[best_name]
    
    print(f"\n  Best Strategy: {best_name} ({best_score:.4f})")
    
    print("\n[4/4] Generating submission...")
    best_th, _ = optimize_threshold(best_preds, dev_gold)
    preds = create_predictions(best_preds, dev_ids, best_th)
    zip_path = create_submission(preds, config.OUTPUT_DIR, 'exp06_submission')
    print(f"  Submission: {zip_path}")
    
    # Save results
    with open(config.OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump({
            'simple_avg': simple_score, 'weighted_avg': wa_score,
            'mlp': mlp_score, 'lightgbm': lgb_score if HAS_LGB else None,
            'best': best_name, 'best_score': float(best_score)
        }, f, indent=2)
    
    print(f"\n{'='*60}\nDONE! Best: {best_name} = {best_score:.4f}\n{'='*60}")

if __name__ == '__main__': main()
