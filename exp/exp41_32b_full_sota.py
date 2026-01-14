# -*- coding: utf-8 -*-
"""
================================================================================
Exp41: 32B FULL SOTA - Massive Model + All Improvements
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

🏆 SOTA COMBINATION SCALED UP:
1. 🧠 **Qwen2.5-32B Model**: 4.5x larger than 7B (Proven 0.88 baseline)
2. ✅ **Fix Pooling**: Last non-pad token (exp34)
3. ✅ **Label Powerset**: 15-class classification for Exact Match (exp35)
4. ✅ **RAG-Fusion**: Multi-Query + Hybrid + RRF Retrieval (exp38)

🎯 TARGET: 0.90+ (SOTA Breaker)

⚠️ HIGHER RESOURCE REQUIREMENTS:
- GPU: H100 80GB (MANDATORY)
- VRAM Usage: ~75GB
- Training Time: ~8 hours

================================================================================
"""

import json, random, shutil, warnings, re, gc, subprocess, sys
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
from itertools import combinations
warnings.filterwarnings('ignore')

def install_deps():
    packages = ['bitsandbytes', 'peft', 'accelerate', 'rank_bm25']
    for pkg in packages:
        try:
            __import__(pkg.replace('_', '-'))
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=True)

install_deps()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ⭐ 15-Class Label Powerset
def generate_label_powerset():
    options = ['A', 'B', 'C', 'D']
    powerset = []
    for r in range(1, len(options) + 1):
        for combo in combinations(options, r):
            powerset.append(','.join(combo))
    return powerset

LABEL_POWERSET = generate_label_powerset()
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABEL_POWERSET)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}
NUM_CLASSES = len(LABEL_POWERSET)

class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp41_32b_full_sota_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp41_32b_full_sota_output')
    
    # ⭐ MASSIVE MODEL
    MODEL_NAME = 'Qwen/Qwen2.5-32B-Instruct'
    MAX_LENGTH = 384  # Shortened to fit VRAM
    
    # ⭐ QLoRA Settings (Aggressive for 32B)
    LORA_R = 32
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    USE_4BIT = True
    
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
    
    # RAG-Fusion Settings (tuned for VRAM)
    CHUNK_SIZE = 300
    TOP_K_CHUNKS = 3  # Reduced from 5 to save context length
    MAX_CONTEXT = 900
    RRF_K = 60
    BM25_WEIGHT = 0.4
    DENSE_WEIGHT = 0.6
    
    # Training
    SEED = 42
    EPOCHS = 2  # Fewer epochs for large model
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 32  # High accumulation
    LEARNING_RATE = 1e-4  # Lower LR
    WARMUP_RATIO = 0.1
    PATIENCE = 2
    
    # Label Powerset
    NUM_CLASSES = 15
    LABEL_SMOOTHING = 0.1
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"{'='*70}")
print(f"Exp41: 32B FULL SOTA COMBINATION")
print(f"{'='*70}")
print(f"🧠 Model: Qwen2.5-32B (H100 Optimized)")
print(f"✅ RAG-Fusion (Multi-Query + Hybrid)")
print(f"✅ Label Powerset (15-Class CE)")
print(f"✅ Fixed Pooling")
print(f"{'='*70}")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(config.SEED)

def compute_aer_score(preds, golds):
    scores = []
    for p, g in zip(preds, golds):
        ps = set(x.strip() for x in p.split(',')) if p else set()
        gs = set(x.strip() for x in g.split(',')) if g else set()
        scores.append(1.0 if ps==gs else 0.5 if ps and ps.issubset(gs) else 0.0)
    return sum(scores)/len(scores) if scores else 0.0

def gold_to_class_idx(golden_answer):
    parts = sorted([x.strip().upper() for x in golden_answer.split(',') if x.strip()])
    normalized = ','.join(parts)
    return LABEL_TO_IDX.get(normalized, 0)

def class_idx_to_answer(idx):
    return IDX_TO_LABEL.get(idx, 'A')

def load_questions(split):
    with open(config.DATA_DIR/split/'questions.jsonl','r',encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def load_docs(split):
    with open(config.DATA_DIR/split/'docs.json','r',encoding='utf-8') as f:
        return {d['topic_id']:d for d in json.load(f)}

class CausalGraphBuilder:
    def __init__(self):
        self.patterns = [(re.compile(p, re.IGNORECASE), t) for p, t in config.CAUSAL_PATTERNS]
    
    def extract_causal_edges(self, text):
        edges = []
        sentences = re.split(r'[.!?]', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10: continue
            for pattern, edge_type in self.patterns:
                match = pattern.search(sent)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        cause, effect = groups[0].strip()[:100], groups[1].strip()[:100]
                        if cause and effect and len(cause) > 5 and len(effect) > 5:
                            edges.append({'cause': cause, 'effect': effect, 'type': edge_type, 'sentence': sent[:200]})
        return edges[:config.MAX_GRAPH_EDGES]
    
    def build_graph(self, topic_info):
        if not topic_info: return {'nodes': set(), 'edges': [], 'sentences': []}
        all_edges = []
        for doc in topic_info.get('docs', []):
            content = doc.get('content', '') or doc.get('snippet', '')
            all_edges.extend(self.extract_causal_edges(content))
        nodes = set()
        for e in all_edges: nodes.add(e['cause']); nodes.add(e['effect'])
        return {'nodes': nodes, 'edges': all_edges}

causal_builder = CausalGraphBuilder()

# ============================================================================
# RAG-FUSION RETRIEVER
# ============================================================================
class RAGFusionRetriever:
    def __init__(self):
        self.embed_model = None
    
    def _load(self):
        if self.embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except:
                subprocess.run(['pip', 'install', 'sentence-transformers', '-q'])
                from sentence_transformers import SentenceTransformer
            print("  Loading embedding model...")
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _tokenize(self, text):
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def _generate_query_variants(self, event, options):
        # Reduced variants slightly for speed/memory if needed, but keeping 6 for fusion quality
        return [
            event,
            f"{event} {options[0][:80]}",
            f"{event} {options[1][:80]}",
            f"{event} {options[2][:80]}",
            f"{event} {options[3][:80]}",
            f"{event} " + " ".join([opt[:40] for opt in options]),
        ]
    
    def _get_hybrid_ranks(self, query, chunks, tokenized_chunks, chunk_meta, graph):
        try:
            from rank_bm25 import BM25Okapi
        except:
            subprocess.run(['pip', 'install', 'rank_bm25', '-q'])
            from rank_bm25 import BM25Okapi
        
        bm25_scores = BM25Okapi(tokenized_chunks).get_scores(self._tokenize(query))
        bm25_ranks = np.argsort(np.argsort(-bm25_scores))
        
        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        chunk_embs = self.embed_model.encode(chunks, convert_to_tensor=True)
        dense_scores = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        
        for i, meta in enumerate(chunk_meta):
            if meta['has_causal']:
                dense_scores[i] *= config.CAUSAL_EDGE_BOOST
        dense_ranks = np.argsort(np.argsort(-dense_scores))
        
        hybrid_scores = np.array([
            config.BM25_WEIGHT / (config.RRF_K + bm25_ranks[i]) +
            config.DENSE_WEIGHT / (config.RRF_K + dense_ranks[i])
            for i in range(len(chunks))
        ])
        return np.argsort(np.argsort(-hybrid_scores))
    
    def retrieve(self, topic_info, event, options, max_chars=1200):
        if not topic_info: return ""
        self._load()
        topic = topic_info.get('topic', '')
        graph = causal_builder.build_graph(topic_info)
        
        chunks, chunk_meta = [], []
        for doc in topic_info.get('docs', []):
            title, content = doc.get('title', ''), doc.get('content', '') or doc.get('snippet', '')
            for i in range(0, len(content), config.CHUNK_SIZE - 50):
                chunk = content[i:i+config.CHUNK_SIZE]
                if len(chunk) < 50: continue
                has_causal = any(e['cause'].lower() in chunk.lower() or e['effect'].lower() in chunk.lower() for e in graph['edges'])
                chunks.append(f"[{title}] {chunk}")
                chunk_meta.append({'has_causal': has_causal})
        
        if not chunks: return f"Topic: {topic}"
        tokenized_chunks = [self._tokenize(c) for c in chunks]
        query_variants = self._generate_query_variants(event, options)
        all_ranks = [self._get_hybrid_ranks(q, chunks, tokenized_chunks, chunk_meta, graph) for q in query_variants]
        final_scores = np.array([sum(1.0 / (config.RRF_K + ranks[i]) for ranks in all_ranks) for i in range(len(chunks))])
        
        top_idx = np.argsort(final_scores)[-config.TOP_K_CHUNKS:][::-1]
        selected = [chunks[i] for i in top_idx]
        
        causal_summary = ""
        if graph['edges']:
            edge_strs = [f"'{e['cause'][:20]}'->{e['effect'][:20]}'" for e in graph['edges'][:2]]
            causal_summary = f"\nCausal: {'; '.join(edge_strs)}"
        context = f"Topic: {topic}{causal_summary}\n\nEvidence:\n" + "\n".join(selected)
        return context[:max_chars]

retriever = RAGFusionRetriever()

# ============================================================================
# DATASET
# ============================================================================
class LabelPowersetDataset(Dataset):
    def __init__(self, questions, docs, tokenizer, max_len=384, is_test=False):
        self.questions, self.docs, self.tokenizer = questions, docs, tokenizer
        self.max_len, self.is_test = max_len, is_test
        self._cache = {}
        
    def __len__(self): return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]; qid = q.get('id')
        if qid not in self._cache:
            topic_info = self.docs.get(q.get('topic_id'), {})
            event = q.get('target_event', '')
            options = [q.get(f'option_{opt}', '') for opt in 'ABCD']
            self._cache[qid] = retriever.retrieve(topic_info, event, options)
        
        context = self._cache[qid]
        event = q.get('target_event', '')
        options_text = "\n".join([f"{opt}: {q.get(f'option_{opt}', '')[:80]}" for opt in 'ABCD'])
        
        prompt = f"Context:\n{context}\n\nEvent: {event}\n\nOptions:\n{options_text}\n\nWhich option(s) best explain the cause of the event? Answer with the letter(s) only (e.g., A or A,B)."
        enc = self.tokenizer(prompt, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        result = {'id': qid, 'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0)}
        
        if not self.is_test:
            label_idx = gold_to_class_idx(q.get('golden_answer', ''))
            result['labels'] = torch.tensor(label_idx, dtype=torch.long)
        return result

# ============================================================================
# MODEL & TRAINING
# ============================================================================
def create_model():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except:
        subprocess.run(['pip', 'install', 'peft', 'bitsandbytes', 'accelerate', '-q'])
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(r=config.LORA_R, lora_alpha=config.LORA_ALPHA, target_modules=config.LORA_TARGET_MODULES, lora_dropout=config.LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, lora_config)

class LabelPowersetHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4), nn.GELU(), nn.Dropout(0.1), # Smaller intermediate for 32B memory
            nn.Linear(hidden_size // 4, config.NUM_CLASSES)
        )
    def forward(self, hidden_states, attention_mask):
        idx = attention_mask.sum(dim=1) - 1
        idx = idx.clamp(min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return self.classifier(hidden_states[batch_indices, idx])

class CausalRAG32B(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.classifier = LabelPowersetHead(hidden_size)
    def forward(self, input_ids, attention_mask):
        self.classifier = self.classifier.to(input_ids.device)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return self.classifier(outputs.hidden_states[-1], attention_mask)

def train_epoch(model, loader, optimizer, scheduler, criterion, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    for step, batch in enumerate(pbar):
        input_ids, attention_mask, labels = batch['input_ids'].to(config.DEVICE), batch['attention_mask'].to(config.DEVICE), batch['labels'].to(config.DEVICE)
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels) / config.GRADIENT_ACCUMULATION
        loss.backward()
        if (step + 1) % config.GRADIENT_ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION
        pbar.set_postfix({'loss': f'{total_loss/(step+1):.4f}'})
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_ids = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids, attention_mask = batch['input_ids'].to(config.DEVICE), batch['attention_mask'].to(config.DEVICE)
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_ids.extend(batch['id'])
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return [class_idx_to_answer(idx) for idx in all_preds], all_ids

def create_submission(answers, ids, out_dir, name):
    results = [{'id': qid, 'answer': ans} for qid, ans in zip(ids, answers)]
    sub = out_dir/'submission'; sub.mkdir(exist_ok=True)
    with open(sub/'submission.jsonl','w') as f:
        for r in results: f.write(json.dumps(r)+'\n')
    shutil.make_archive(str(out_dir/name),'zip',sub)
    return out_dir/f'{name}.zip'

def main():
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    print("\n[1/6] Loading data...")
    train_q, dev_q, test_q = load_questions('train_data'), load_questions('dev_data'), load_questions('test_data')
    train_docs, dev_docs, test_docs = load_docs('train_data'), load_docs('dev_data'), load_docs('test_data')
    
    print("\n[2/6] Initializing RAG-Fusion Retriever...")
    retriever._load()
    
    print("\n[3/6] Creating 32B model with QLoRA...")
    base_model = create_model()
    model = CausalRAG32B(base_model, base_model.config.hidden_size)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    print("\n[4/6] Preparing datasets...")
    train_ds = LabelPowersetDataset(train_q, train_docs, tokenizer, config.MAX_LENGTH)
    dev_ds = LabelPowersetDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    test_ds = LabelPowersetDataset(test_q, test_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, num_workers=0)
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.WARMUP_RATIO), total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    print("\n[5/6] Training 32B SOTA Model...")
    print("Warning: This may take ~8 hours on H100.")
    dev_gold = [q.get('golden_answer','') for q in dev_q]
    best_score = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, epoch)
        dev_answers, _ = evaluate(model, dev_loader)
        score = compute_aer_score(dev_answers, dev_gold)
        print(f"  Epoch {epoch}: Loss={train_loss:.4f}, Dev={score:.4f}")
        
        if score > best_score:
            best_score = score
            model.base_model.save_pretrained(config.OUTPUT_DIR/'lora_weights')
            torch.save({'classifier': model.classifier.state_dict(), 'score': score}, config.OUTPUT_DIR/'best_model.pt')
            print(f"  -> New best! 🎉")
    
    print(f"\n  Best Dev Score: {best_score:.4f}")
    
    print("\n[6/6] Predicting on TEST...")
    ckpt = torch.load(config.OUTPUT_DIR/'best_model.pt', map_location='cpu', weights_only=False)
    model.classifier.load_state_dict(ckpt['classifier'])
    test_answers, test_ids = evaluate(model, test_loader)
    zip_path = create_submission(test_answers, test_ids, config.OUTPUT_DIR, 'exp41_32b_full_sota_submission')
    
    with open(config.OUTPUT_DIR/'results.json', 'w') as f:
        json.dump({'experiment': 'exp41_32b_full_sota', 'dev_score': float(best_score), 'model': config.MODEL_NAME}, f, indent=2)
    print(f"\n🏆 Exp41 32B Complete! SOTA Combination of 32B + RAG-Fusion + Label Powerset + Pooling Fix. Submission: {zip_path}")

if __name__ == '__main__':
    main()
