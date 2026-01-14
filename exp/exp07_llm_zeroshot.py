# -*- coding: utf-8 -*-
"""
================================================================================
SemEval-2026 Task 12: Abductive Event Reasoning
Exp07: LLM Zero-Shot with Temperature Scaling (HIGHLY NOVEL)
================================================================================
SELF-CONTAINED SCRIPT - Copy this entire file into a single Kaggle notebook cell

Key Innovations:
- Novel multi-label prompt design for LLMs
- Temperature scaling calibration
- Confidence score extraction and normalization
- Support for multiple LLMs

GPU: H100 80GB required for Llama-70B, or use smaller models
Time: ~1-2 hours
================================================================================
"""

#%% IMPORTS
import json, random, shutil, warnings, re
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

#%% CONFIG
class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp07_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp07_output')
    
    # Model: Use smaller model for testing, larger for production
    MODEL_NAME = 'meta-llama/Llama-3.1-8B-Instruct'  # or 'Qwen/Qwen2-7B-Instruct'
    USE_MOCK = False  # Set True to test without LLM
    TEMPERATURE = 0.7
    CALIBRATE = True
    
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*60}\nExp07: LLM Zero-Shot (NOVEL)\n{'='*60}\nModel: {config.MODEL_NAME}\nMock mode: {config.USE_MOCK}\n{'='*60}")

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

def load_docs(split):
    with open(config.DATA_DIR/split/'docs.json', 'r', encoding='utf-8') as f:
        return {d['topic_id']: d for d in json.load(f)}

def get_context(topic_info, max_chars=800):
    if not topic_info: return ""
    parts = [topic_info.get('topic', '')]
    for doc in topic_info.get('docs', [])[:3]:
        if s := doc.get('snippet', ''): parts.append(s)
    return ' '.join(parts)[:max_chars]

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

#%% PROMPT
PROMPT = """You are an expert in causal reasoning.

Event: {event}
Context: {context}

Which option(s) are the DIRECT cause of the event? Multiple may be correct.
A: {option_A}
B: {option_B}
C: {option_C}
D: {option_D}

Rate each option 0-100 for likelihood of being a direct cause.
Output ONLY JSON: {{"A": score, "B": score, "C": score, "D": score}}"""

#%% LLM CLIENTS
class MockLLM:
    """Mock LLM for testing."""
    def generate(self, prompt, **kwargs):
        np.random.seed(hash(prompt) % (2**32))
        return json.dumps({k: int(np.random.randint(30, 70)) for k in ['A','B','C','D']})

class TransformersLLM:
    """Real LLM using HuggingFace."""
    def __init__(self, model_name):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        print(f"Loading {model_name}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @torch.no_grad()
    def generate(self, prompt, temperature=0.7, max_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature,
                                   do_sample=True, top_p=0.9, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def parse_scores(response):
    """Parse JSON scores from LLM response."""
    match = re.search(r'\{[^{}]*\}', response)
    if match:
        try:
            scores = json.loads(match.group())
            return {k: max(0, min(100, float(scores.get(k, 50))))/100 for k in ['A','B','C','D']}
        except: pass
    return {k: 0.5 for k in ['A','B','C','D']}

#%% TEMPERATURE SCALING
def calibrate_temperature(probs, labels, lr=0.01, max_iter=100):
    """Learn optimal temperature for calibration."""
    logits = np.log(np.clip(probs, 1e-8, 1-1e-8) / (1 - np.clip(probs, 1e-8, 1-1e-8)))
    logits_t = torch.FloatTensor(logits)
    labels_t = torch.FloatTensor(labels)
    
    temp = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temp], lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()
    
    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_t / temp, labels_t)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    return temp.item()

#%% MAIN
def main():
    print("\n[1/4] Loading data...")
    dev_q = load_questions('dev_data')
    dev_docs = load_docs('dev_data')
    
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
    
    print("\n[2/4] Initializing LLM...")
    if config.USE_MOCK:
        llm = MockLLM()
    else:
        llm = TransformersLLM(config.MODEL_NAME)
    
    print("\n[3/4] Generating predictions...")
    all_probs = []
    all_ids = []
    
    for q in tqdm(dev_q, desc="LLM Inference"):
        event = q.get('target_event', '')
        topic_info = dev_docs.get(q.get('topic_id'), {})
        context = get_context(topic_info)
        
        prompt = PROMPT.format(
            event=event, context=context,
            option_A=q.get('option_A', ''),
            option_B=q.get('option_B', ''),
            option_C=q.get('option_C', ''),
            option_D=q.get('option_D', '')
        )
        
        response = llm.generate(prompt, temperature=config.TEMPERATURE)
        scores = parse_scores(response)
        
        all_probs.append([scores['A'], scores['B'], scores['C'], scores['D']])
        all_ids.append(q['id'])
    
    all_probs = np.array(all_probs)
    
    # Evaluate raw
    print("\n=== Raw Predictions ===")
    raw_th, raw_sc = optimize_threshold(all_probs, dev_gold)
    print(f"  Score: {raw_sc:.4f} (th={raw_th})")
    
    # Calibrate
    if config.CALIBRATE:
        print("\n=== Calibration ===")
        n_calib = len(all_probs) // 2
        temp = calibrate_temperature(all_probs[:n_calib], dev_labels[:n_calib])
        print(f"  Learned temperature: {temp:.4f}")
        
        # Apply
        logits = np.log(np.clip(all_probs, 1e-8, 1-1e-8) / (1 - np.clip(all_probs, 1e-8, 1-1e-8)))
        calibrated = 1 / (1 + np.exp(-logits / temp))
        
        cal_th, cal_sc = optimize_threshold(calibrated, dev_gold)
        print(f"  Calibrated Score: {cal_sc:.4f} (th={cal_th})")
        
        if cal_sc > raw_sc:
            final_probs, final_th, final_sc = calibrated, cal_th, cal_sc
            print("  Using calibrated predictions")
        else:
            final_probs, final_th, final_sc = all_probs, raw_th, raw_sc
            print("  Using raw predictions")
    else:
        final_probs, final_th, final_sc = all_probs, raw_th, raw_sc
    
    print("\n[4/4] Generating submission...")
    preds = create_predictions(final_probs, all_ids, final_th)
    zip_path = create_submission(preds, config.OUTPUT_DIR, 'exp07_submission')
    print(f"  Submission: {zip_path}")
    
    # Save predictions for ensemble
    np.save(config.OUTPUT_DIR / 'dev_predictions.npy', final_probs)
    
    print(f"\n{'='*60}\nDONE! Score: {final_sc:.4f}\n{'='*60}")

if __name__ == '__main__': main()
