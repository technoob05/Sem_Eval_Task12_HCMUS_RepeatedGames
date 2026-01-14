# -*- coding: utf-8 -*-
"""
================================================================================
Exp39: SELF-REFLECTIVE RAG - Confidence-Based Retrieval Gating
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

🎯 KEY INSIGHT: "Easy" questions don't need evidence (can introduce noise)
✅ SOLUTION: "Gate" retrieval based on model confidence (Self-RAG style)

Logic (Inference only):
1. Predict answer WITHOUT context (Zero-shot)
2. Calculate Confidence (Entropy of probabilities)
3. If Confidence is HIGH (Entropy < Threshold):
   -> Return Zero-shot answer (Skip retrieval)
4. If Confidence is LOW (Entropy >= Threshold):
   -> Retrieve evidence (RAG-Fusion)
   -> Predict with context

Includes: RAG-Fusion + Pooling fix

GPU: H100 80GB (required)
================================================================================
"""

import json, random, shutil, warnings, re, gc, subprocess, sys, copy
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
            print(f"  Installing {pkg}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=True)

install_deps()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp39_self_rag_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp39_self_rag_output')
    
    MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
    MAX_LENGTH = 512
    
    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
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
    
    # RAG Settings
    CHUNK_SIZE = 350
    TOP_K_CHUNKS = 5
    MAX_CONTEXT = 1200
    RRF_K = 60
    BM25_WEIGHT = 0.4
    DENSE_WEIGHT = 0.6
    
    # ⭐ Self-RAG Settings
    ENTROPY_THRESHOLD = 0.6  # If entropy > 0.6, consider "unsure" -> Retrieve
    # Max entropy for 4 classes is ln(4) ≈ 1.38
    # Uniform dist [0.25]*4 -> entropy 1.38
    # Certainty [0.9, 0.03, 0.03, 0.04] -> entropy ~0.45
    
    SEED = 42
    EPOCHS = 3
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 16
    LEARNING_RATE = 2e-4
    WARMUP_RATIO = 0.1
    PATIENCE = 2
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"{'='*70}")
print(f"Exp39: SELF-REFLECTIVE RAG (Gated Retrieval)")
print(f"{'='*70}")
print(f"✅ Gated Retrieval Strategy:")
print(f"   1. Low Entropy (Sure) -> Use Zero-shot answer")
print(f"   2. High Entropy (Unsure) -> Use RAG-Fusion answer")
print(f"✅ Entropy Threshold: {config.ENTROPY_THRESHOLD}")
print(f"✅ Includes pooling fix")
print(f"Model: {config.MODEL_NAME}")
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

def load_questions(split):
    with open(config.DATA_DIR/split/'questions.jsonl','r',encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def load_docs(split):
    with open(config.DATA_DIR/split/'docs.json','r',encoding='utf-8') as f:
        return {d['topic_id']:d for d in json.load(f)}

# ============================================================================
# CAUSAL GRAPH BUILDER
# ============================================================================
class CausalGraphBuilder:
    def __init__(self):
        self.patterns = [(re.compile(p, re.IGNORECASE), t) for p, t in config.CAUSAL_PATTERNS]
    
    def extract_causal_edges(self, text):
        edges = []
        sentences = re.split(r'[.!?]', text)
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            
            for pattern, edge_type in self.patterns:
                match = pattern.search(sent)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        cause = groups[0].strip()[:100]
                        effect = groups[1].strip()[:100]
                        if cause and effect and len(cause) > 5 and len(effect) > 5:
                            edges.append({
                                'cause': cause,
                                'effect': effect,
                                'type': edge_type,
                                'sentence': sent[:200]
                            })
        
        return edges[:config.MAX_GRAPH_EDGES]
    
    def build_graph(self, topic_info):
        if not topic_info:
            return {'nodes': set(), 'edges': [], 'sentences': []}
        
        all_edges = []
        for doc in topic_info.get('docs', []):
            content = doc.get('content', '') or doc.get('snippet', '')
            edges = self.extract_causal_edges(content)
            all_edges.extend(edges)
        
        nodes = set()
        for e in all_edges:
            nodes.add(e['cause'])
            nodes.add(e['effect'])
        
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
            title = doc.get('title', '')
            content = doc.get('content', '') or doc.get('snippet', '')
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
class CausalRAGDataset(Dataset):
    def __init__(self, questions, docs, tokenizer, max_len=512, is_test=False, mode='rag'):
        self.questions = questions
        self.docs = docs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.mode = mode  # 'rag' or 'zero_shot'
        self._cache = {}
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        qid = q.get('id')
        
        context = ""
        # ⭐ Only do retrieval if mode is 'rag'
        if self.mode == 'rag':
            if qid not in self._cache:
                topic_info = self.docs.get(q.get('topic_id'), {})
                event = q.get('target_event', '')
                options = [q.get(f'option_{opt}', '') for opt in 'ABCD']
                context = retriever.retrieve(topic_info, event, options)
                self._cache[qid] = context
            context = self._cache[qid]
        
        event = q.get('target_event', '')
        options_text = "\n".join([f"{opt}: {q.get(f'option_{opt}', '')}" for opt in 'ABCD'])
        
        prompt = f"""Context:
{context}

Event: {event}

Options:
{options_text}

Which option(s) best explain the cause of the event? Answer with the letter(s) only (e.g., A or A,B)."""

        enc = self.tokenizer(
            prompt, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        result = {
            'id': qid,
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
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
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA, target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 4)
        )
    def forward(self, hidden_states, attention_mask):
        idx = attention_mask.sum(dim=1) - 1
        idx = idx.clamp(min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, idx]
        return self.classifier(last_hidden)

class CausalRAG7B(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.classifier = ClassificationHead(hidden_size)
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
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels) / config.GRADIENT_ACCUMULATION
        loss.backward()
        if (step + 1) % config.GRADIENT_ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION
        pbar.set_postfix({'loss': f'{total_loss/(step+1):.4f}'})
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_ids = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        logits = model(input_ids, attention_mask)
        all_preds.extend(torch.sigmoid(logits).cpu().numpy())
        all_ids.extend(batch['id'])
    return np.array(all_preds), all_ids

def optimize_threshold(probs, golds):
    opts = list('ABCD')
    best_th, best_sc = 0.5, 0
    for th in np.arange(0.15, 0.75, 0.025):
        ps = [','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]])) for p in probs]
        sc = compute_aer_score(ps, golds)
        if sc > best_sc: best_sc, best_th = sc, th
    return best_th, best_sc

# ⭐ SELF-RAG GATED PREDICTION
def predict_with_gating(model, questions, docs, tokenizer):
    """Run Self-RAG prediction pipeline"""
    model.eval()
    
    # 1. Create Zero-shot Dataset (no retrieval)
    zero_shot_ds = CausalRAGDataset(questions, docs, tokenizer, config.MAX_LENGTH, is_test=True, mode='zero_shot')
    zero_shot_loader = DataLoader(zero_shot_ds, batch_size=config.BATCH_SIZE, num_workers=0)
    
    print("  Running Step 1: Zero-shot prediction + Confidence Check...")
    zero_probs, ids = evaluate(model, zero_shot_loader)
    
    # 2. Check Entropy
    # entropy = -sum(p * log(p))
    # Normalize probs to sum to 1 for entropy calc (softmax behavior approximation)
    # But we have sigmoids. Let's treat them as independent.
    # High entropy means "uncertainty".
    # For multi-label, maybe use average distance from 0.5?
    # Or simply: if max prob < 0.7, retrieve.
    # Let's use standard entropy on normalized logits if treating as distribution, or just raw uncertainty.
    # Let's calculate entropy of the normalized distribution across 4 classes
    
    final_probs = np.zeros_like(zero_probs)
    rag_indices = []
    
    for i, probs in enumerate(zero_probs):
        # Normalize to probability distribution for entropy
        p_norm = probs / (probs.sum() + 1e-9)
        entropy = -np.sum(p_norm * np.log(p_norm + 1e-9))
        
        # Check gating condition
        if entropy < config.ENTROPY_THRESHOLD:
            # High confidence (Low entropy) -> Keep Zero-shot
            final_probs[i] = probs
        else:
            # Low confidence -> Mark for RAG
            rag_indices.append(i)
    
    print(f"  Step 1 Stats: {len(zero_probs)-len(rag_indices)} kept Zero-shot, {len(rag_indices)} need RAG")
    
    if rag_indices:
        # 3. Create RAG Dataset for hard questions
        rag_questions = [questions[i] for i in rag_indices]
        rag_ds = CausalRAGDataset(rag_questions, docs, tokenizer, config.MAX_LENGTH, is_test=True, mode='rag')
        rag_loader = DataLoader(rag_ds, batch_size=config.BATCH_SIZE, num_workers=0)
        
        print("  Running Step 2: RAG prediction for hard samples...")
        rag_probs, _ = evaluate(model, rag_loader)
        
        # 4. Merge results
        for idx_in_rag, original_idx in enumerate(rag_indices):
            final_probs[original_idx] = rag_probs[idx_in_rag]
            
    return final_probs, ids

def create_submission(probs, ids, th, out_dir, name):
    opts = list('ABCD')
    results = [{'id': qid, 'answer': ','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]]))} for qid, p in zip(ids, probs)]
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
    
    print("\n[3/6] Creating model with QLoRA...")
    base_model = create_model()
    model = CausalRAG7B(base_model, base_model.config.hidden_size)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    print("\n[4/6] Preparing datasets (Training uses full RAG)...")
    train_ds = CausalRAGDataset(train_q, train_docs, tokenizer, config.MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.WARMUP_RATIO), total_steps)
    criterion = nn.BCEWithLogitsLoss()
    
    print("\n[5/6] Training...")
    dev_gold = [q.get('golden_answer','') for q in dev_q]
    best_score, best_th = 0, 0.5
    
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, epoch)
        
        # Validation using Self-RAG gating
        print("  Evaluating Dev with Self-RAG Gating...")
        dev_probs, _ = predict_with_gating(model, dev_q, dev_docs, tokenizer)
        th, score = optimize_threshold(dev_probs, dev_gold)
        print(f"  Epoch {epoch}: Loss={train_loss:.4f}, Dev={score:.4f} (th={th:.3f})")
        
        if score > best_score:
            best_score, best_th = score, th
            model.base_model.save_pretrained(config.OUTPUT_DIR/'lora_weights')
            torch.save({'classifier': model.classifier.state_dict(), 'score': score, 'threshold': th}, config.OUTPUT_DIR/'best_model.pt')
            print(f"  -> New best! 🎉")
    
    print(f"\n  Best Dev Score: {best_score:.4f}")
    
    print("\n[6/6] Predicting on TEST with Self-RAG Gating...")
    ckpt = torch.load(config.OUTPUT_DIR/'best_model.pt', map_location='cpu', weights_only=False)
    model.classifier.load_state_dict(ckpt['classifier'])
    
    test_probs, test_ids = predict_with_gating(model, test_q, test_docs, tokenizer)
    zip_path = create_submission(test_probs, test_ids, ckpt['threshold'], config.OUTPUT_DIR, 'exp39_self_rag_submission')
    
    with open(config.OUTPUT_DIR/'results.json', 'w') as f:
        json.dump({'experiment': 'exp39_self_rag', 'dev_score': float(best_score), 'model': config.MODEL_NAME}, f, indent=2)
    
    print(f"\n🏆 Exp39 Complete! Submission: {zip_path}")

if __name__ == '__main__':
    main()
