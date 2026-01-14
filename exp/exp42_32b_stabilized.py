# -*- coding: utf-8 -*-
"""
================================================================================
Exp42: 32B STABILIZED - Pooling Fix + Hybrid RAG + BCE (Safe Mode)
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

🎯 GOAL: Recover 0.88+ performance and aim for 0.89+
- Reverts to BCE Loss (4 binary classifiers) from Exp32 (Proven to work)
- Keeps Pooling Fix (Free gain)
- Keeps Hybrid RAG (Better than CausalRAG) but simplified (Top-2 chunks)
- Removes Label Powerset (Too unstable for 32B QLoRA limited training)

⚠️ REQUIREMENTS:
- GPU: H100 80GB (MANDATORY)
"""

import json, random, shutil, warnings, re, gc, subprocess, sys
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
warnings.filterwarnings('ignore')

def install_deps():
    packages = ['bitsandbytes', 'peft', 'accelerate', 'rank_bm25']
    for pkg in packages:
        try:
            __import__(pkg.replace('_', '-'))
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=True)

install_deps()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp42_32b_stabilized_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp42_32b_stabilized_output')
    
    MODEL_NAME = 'Qwen/Qwen2.5-32B-Instruct'
    MAX_LENGTH = 384
    
    # QLoRA (Aggressive for memory)
    LORA_R = 32
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    USE_4BIT = True
    
    # RAG Settings (Simplified Fusion)
    CHUNK_SIZE = 300
    TOP_K_CHUNKS = 2  # Match Exp32
    MAX_CONTEXT = 800 # Match Exp32
    RRF_K = 60
    BM25_WEIGHT = 0.4
    DENSE_WEIGHT = 0.6
    
    CAUSAL_PATTERNS = [
        (r'(.+?) caused (.+)', 'CAUSE'),
        (r'(.+?) led to (.+)', 'CAUSE'),
        (r'(.+?) resulted in (.+)', 'CAUSE'),
        (r'because of (.+?), (.+)', 'CAUSE'),
        (r'after (.+?), (.+)', 'TEMPORAL'),
        (r'(.+?) triggered (.+)', 'CAUSE'),
        (r'(.+?) prompted (.+)', 'CAUSE'),
        (r'following (.+?), (.+)', 'TEMPORAL'),
    ]
    CAUSAL_EDGE_BOOST = 1.5
    MAX_GRAPH_EDGES = 20
    
    SEED = 42
    EPOCHS = 2
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 32
    LEARNING_RATE = 1e-4
    WARMUP_RATIO = 0.1
    PATIENCE = 2
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(config.SEED)

def compute_aer_score(preds, golds):
    scores = []
    for p, g in zip(preds, golds):
        ps = set(x.strip() for x in p.split(',')) if p else set()
        gs = set(x.strip() for x in g.split(',')) if g else set()
        scores.append(1.0 if ps==gs else 0.5 if ps and ps.issubset(gs) else 0.0)
    return sum(scores)/len(scores) if scores else 0.0

def load_data(split):
    with open(config.DATA_DIR/split/'questions.jsonl','r') as f: q = [json.loads(l) for l in f]
    with open(config.DATA_DIR/split/'docs.json','r') as f: d = {x['topic_id']:x for x in json.load(f)}
    return q, d

# --- Utilities ---
class CausalGraphBuilder:
    def __init__(self):
        self.patterns = [(re.compile(p, re.IGNORECASE), t) for p, t in config.CAUSAL_PATTERNS]
    def extract_edges(self, text):
        edges = []
        for sent in re.split(r'[.!?]', text):
            sent=sent.strip()
            if len(sent)<10: continue
            for pat, typ in self.patterns:
                m = pat.search(sent)
                if m:
                    g = m.groups()
                    if len(g)>=2 and len(g[0])>5 and len(g[1])>5:
                        edges.append({'cause':g[0].strip()[:100], 'effect':g[1].strip()[:100], 'type':typ})
        return edges[:config.MAX_GRAPH_EDGES]
    def build(self, info):
        if not info: return {'edges':[]}
        edges = []
        for d in info.get('docs',[]):
            edges.extend(self.extract_edges(d.get('content','') or d.get('snippet','')))
        return {'edges': edges}

graph_builder = CausalGraphBuilder()

class RAGFusionRetriever:
    def __init__(self): self.embed_model = None
    def _load(self):
        if not self.embed_model:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    def tokenize(self, t): return re.sub(r'[^\w\s]', ' ', t.lower()).split()
    def retrieve(self, info, event, options, ax_chars=800):
        if not info: return ""
        self._load()
        from rank_bm25 import BM25Okapi
        
        graph = graph_builder.build(info)
        chunks, metas = [], []
        for d in info.get('docs',[]):
            txt = d.get('content','') or d.get('snippet','')
            for i in range(0, len(txt), config.CHUNK_SIZE-50):
                c = txt[i:i+config.CHUNK_SIZE]
                if len(c)<50: continue
                has_causal = any(e['cause'] in c or e['effect'] in c for e in graph['edges'])
                chunks.append(f"[{d.get('title','')}] {c}")
                metas.append({'has_causal': has_causal})
        
        if not chunks: return f"Topic: {info.get('topic','')}"
        
        # Simple Fusion: Event vs Event+Options
        queries = [event, f"{event} {' '.join(options)}"]
        final_scores = np.zeros(len(chunks))
        
        tokenized_chunks = [self.tokenize(c) for c in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        chunk_embs = self.embed_model.encode(chunks, convert_to_tensor=True)
        
        for q in queries:
            # BM25
            bm25_sc = bm25.get_scores(self.tokenize(q))
            bm25_rnk = np.argsort(np.argsort(-bm25_sc))
            
            # Dense
            q_emb = self.embed_model.encode(q, convert_to_tensor=True)
            dense_sc = torch.cosine_similarity(q_emb.unsqueeze(0), chunk_embs).cpu().numpy()
            for i,m in enumerate(metas):
                if m['has_causal']: dense_sc[i] *= config.CAUSAL_EDGE_BOOST
            dense_rnk = np.argsort(np.argsort(-dense_sc))
            
            # RRF
            for i in range(len(chunks)):
                final_scores[i] += (config.BM25_WEIGHT/(config.RRF_K+bm25_rnk[i])) + (config.DENSE_WEIGHT/(config.RRF_K+dense_rnk[i]))
        
        top_idx = np.argsort(final_scores)[-config.TOP_K_CHUNKS:][::-1]
        selected = [chunks[i] for i in top_idx]
        return f"Topic: {info.get('topic','')}\nEvidence:\n"+"\n".join(selected)

retriever = RAGFusionRetriever()

class Dataset32B(Dataset):
    def __init__(self, qs, docs, tok, max_len=384, test=False):
        self.qs, self.docs, self.tok, self.max_len, self.test = qs, docs, tok, max_len, test
        self.cache = {}
    def __len__(self): return len(self.qs)
    def __getitem__(self, idx):
        q = self.qs[idx]; qid = q['id']
        if qid not in self.cache:
            self.cache[qid] = retriever.retrieve(self.docs.get(q['topic_id']), q['target_event'], [q[f'option_{k}'] for k in 'ABCD'])
        
        ctx = self.cache[qid]
        opts = "\n".join([f"{k}: {q[f'option_{k}']}" for k in 'ABCD'])
        prompt = f"Context:\n{ctx}\nEvent: {q['target_event']}\nOptions:\n{opts}\nWhich option(s) caused the event? Answer A,B,C,D."
        
        enc = self.tok(prompt, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        res = {'input_ids': enc['input_ids'][0], 'attention_mask': enc['attention_mask'][0], 'id': qid}
        
        if not self.test:
            g = q.get('golden_answer','').upper()
            lbl = torch.zeros(4)
            for char in 'ABCD':
                if char in g: lbl['ABCD'.index(char)] = 1.0
            res['labels'] = lbl
        return res

def create_model():
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    conf = LoraConfig(r=config.LORA_R, lora_alpha=config.LORA_ALPHA, target_modules=config.LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, conf)

class Head(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(hdim, hdim//4), nn.GELU(), nn.Dropout(0.1), nn.Linear(hdim//4, 4))
    def forward(self, x, mask):
        # POOLING FIX: Last non-pad
        idx = mask.sum(dim=1) - 1
        idx = idx.clamp(min=0)
        pool = x[torch.arange(x.size(0)), idx]
        return self.h(pool)

class Model32B(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.head = Head(base.config.hidden_size)
    def forward(self, ii, am):
        self.head = self.head.to(ii.device)
        out = self.base(input_ids=ii, attention_mask=am, output_hidden_states=True)
        return self.head(out.hidden_states[-1], am)

def train(model, loader, opt, sched, crit, ep):
    model.train(); total = 0
    for i, b in enumerate(tqdm(loader, desc=f"Ep {ep}")):
        out = model(b['input_ids'].to(config.DEVICE), b['attention_mask'].to(config.DEVICE))
        loss = crit(out, b['labels'].to(config.DEVICE)) / config.GRADIENT_ACCUMULATION
        loss.backward()
        if (i+1)%config.GRADIENT_ACCUMULATION==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); sched.step(); opt.zero_grad()
        total+=loss.item()*config.GRADIENT_ACCUMULATION
    return total/len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds = []; ids = []
    for b in tqdm(loader, desc="Eval"):
        out = model(b['input_ids'].to(config.DEVICE), b['attention_mask'].to(config.DEVICE))
        preds.extend(torch.sigmoid(out).cpu().numpy())
        ids.extend(b['id'])
    return preds, ids

def main():
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    train_q, train_d = load_data('train_data')
    dev_q, dev_d = load_data('dev_data')
    test_q, test_d = load_data('test_data')
    
    retriever._load()
    base = create_model()
    model = Model32B(base)
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if not tok.pad_token: tok.pad_token = tok.eos_token
    
    tr_ds = Dataset32B(train_q, train_d, tok)
    de_ds = Dataset32B(dev_q, dev_d, tok, test=True)
    te_ds = Dataset32B(test_q, test_d, tok, test=True) # NOTE: Using test=True for test loader logic
    
    tr_dl = DataLoader(tr_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    de_dl = DataLoader(de_ds, batch_size=config.BATCH_SIZE)
    te_dl = DataLoader(te_ds, batch_size=config.BATCH_SIZE)
    
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.LEARNING_RATE)
    steps = len(tr_dl)*config.EPOCHS//config.GRADIENT_ACCUMULATION
    sched = get_linear_schedule_with_warmup(opt, int(steps*config.WARMUP_RATIO), steps)
    crit = nn.BCEWithLogitsLoss()
    
    best_sc, best_th = 0, 0.5
    dev_g = [q.get('golden_answer','') for q in dev_q]
    
    for ep in range(1, config.EPOCHS+1):
        l = train(model, tr_dl, opt, sched, crit, ep)
        probs, _ = evaluate(model, de_dl)
        
        # Optimize Threshold
        th_best, sc_best = 0.5, 0
        opts = list('ABCD')
        for th in np.arange(0.15, 0.75, 0.05):
            ps = [','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]])) for p in probs]
            sc = compute_aer_score(ps, dev_g)
            if sc > sc_best: sc_best, th_best = sc, th
            
        print(f"Ep {ep}: Loss={l:.4f} Dev={sc_best:.4f} Th={th_best:.2f}")
        
        if sc_best > best_sc:
            best_sc, best_th = sc_best, th_best
            torch.save({'head': model.head.state_dict(), 'th': best_th}, config.OUTPUT_DIR/'best.pt')
            model.base.save_pretrained(config.OUTPUT_DIR/'lora')

    print(f"Best: {best_sc}")
    ckpt = torch.load(config.OUTPUT_DIR/'best.pt')
    model.head.load_state_dict(ckpt['head'])
    
    te_probs, te_ids = evaluate(model, te_dl)
    opts = list('ABCD')
    res = [{'id':i, 'answer':','.join(sorted([opts[x] for x in range(4) if p[x]>=ckpt['th']] or [opts[np.argmax(p)]]))} for i,p in zip(te_ids, te_probs)]
    
    sub = config.OUTPUT_DIR/'submission'; sub.mkdir(exist_ok=True)
    with open(sub/'submission.jsonl','w') as f:
        for r in res: f.write(json.dumps(r)+'\n')
    shutil.make_archive(str(config.OUTPUT_DIR/'exp42_submission'), 'zip', sub)
    print("Done exp42.")

if __name__=='__main__': main()
