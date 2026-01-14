# -*- coding: utf-8 -*-
"""
================================================================================
Exp32 32B: CausalRAG + Qwen2.5-32B QLoRA Fine-tuning
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

🎯 TARGET: 0.90+ (from 0.86)
🏆 MODEL: Qwen2.5-32B-Instruct with QLoRA (4-bit)

💡 WHY THIS WORKS:
- Qwen2.5-32B is 4.5x larger than 7B (32B vs 7B params)
- Larger models = better reasoning capability
- QLoRA with aggressive settings to fit in H100 80GB

⚠️ REQUIREMENTS:
- GPU: H100 80GB (REQUIRED - will OOM on smaller GPUs)
- Time: ~6-8 hours
- VRAM: ~70-75GB with QLoRA

================================================================================
"""

import json, random, shutil, warnings, re, gc, subprocess, sys
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
warnings.filterwarnings('ignore')

# ⭐ Install required packages for QLoRA
def install_qlora_deps():
    packages = ['bitsandbytes', 'peft', 'accelerate', 'sentence-transformers', 'rank_bm25']
    for pkg in packages:
        try:
            if pkg == 'rank_bm25':
                __import__('rank_bm25')
            else:
                __import__(pkg)
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=True)

install_qlora_deps()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp32_32b_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp32_32b_output')
    
    # ⭐ LARGEST MODEL: Qwen2.5-32B with QLoRA
    MODEL_NAME = 'Qwen/Qwen2.5-32B-Instruct'
    MAX_LENGTH = 384  # Shorter for larger model
    
    # ⭐ QLoRA Settings - Expanded for better learning
    LORA_R = 32
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
    USE_4BIT = True
    USE_GRAD_CHECKPOINT = True  # Added for memory efficiency
    
    # CausalRAG Settings
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
    
    # RAG Settings - Hybrid (Dense + BM25)
    CHUNK_SIZE = 300 
    TOP_K_CHUNKS = 3  # Slightly more for 32B
    MAX_CONTEXT = 1000 
    BM25_WEIGHT = 0.3 # 30% BM25, 70% Dense
    DENSE_WEIGHT = 0.7
    
    # Training - Very conservative for huge model
    SEED = 42
    EPOCHS = 2  # Fewer epochs
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 32  # Higher accumulation
    LEARNING_RATE = 1e-4  # Lower LR for stability
    WARMUP_RATIO = 0.1
    PATIENCE = 2
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"{'='*70}")
print(f"Exp32 32B: CausalRAG + Qwen2.5-32B QLoRA")
print(f"{'='*70}")
print(f"Model: {config.MODEL_NAME}")
print(f"LoRA Rank: {config.LORA_R}")
print(f"4-bit Quantization: {config.USE_4BIT}")
print(f"Max Length: {config.MAX_LENGTH}")
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
# CAUSAL RETRIEVER
# ============================================================================
class CausalAwareRetriever:
    def __init__(self):
        self.embed_model = None
    
    def _load(self):
        if self.embed_model is None:
            from sentence_transformers import SentenceTransformer
            print("  Loading embedding model...")
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _tokenize(self, text):
        return re.sub(r'[^\w\s]', ' ', text.lower()).split()

    def retrieve(self, topic_info, event, options, max_chars=1000):
        if not topic_info:
            return ""
        
        self._load()
        from rank_bm25 import BM25Okapi
        
        topic = topic_info.get('topic', '')
        query = f"{event} " + " ".join([opt[:50] for opt in options])
        
        graph = causal_builder.build_graph(topic_info)
        
        chunks = []
        chunk_meta = []
        
        for doc in topic_info.get('docs', []):
            title = doc.get('title', '')
            content = doc.get('content', '') or doc.get('snippet', '')
            
            for i in range(0, len(content), config.CHUNK_SIZE - 50):
                chunk = content[i:i+config.CHUNK_SIZE]
                if len(chunk) < 50:
                    continue
                
                has_causal = any(
                    e['cause'].lower() in chunk.lower() or e['effect'].lower() in chunk.lower()
                    for e in graph['edges']
                )
                
                chunks.append(f"[{title}] {chunk}")
                chunk_meta.append({'has_causal': has_causal})
        
        if not chunks:
            return f"Topic: {topic}"
        
        # 1. BM25 Scores
        tokenized_chunks = [self._tokenize(c) for c in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = bm25.get_scores(self._tokenize(query))
        # Normalize BM25
        if max(bm25_scores) > 0:
            bm25_scores = bm25_scores / max(bm25_scores)
            
        # 2. Dense Scores
        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        chunk_embs = self.embed_model.encode(chunks, convert_to_tensor=True)
        dense_scores = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        
        # 3. Hybrid Combination
        combined_scores = (config.BM25_WEIGHT * bm25_scores) + (config.DENSE_WEIGHT * dense_scores)
        
        for i, meta in enumerate(chunk_meta):
            if meta['has_causal']:
                combined_scores[i] *= config.CAUSAL_EDGE_BOOST
        
        top_idx = np.argsort(combined_scores)[-config.TOP_K_CHUNKS:][::-1]
        selected = [chunks[i] for i in top_idx]
        
        causal_summary = ""
        if graph['edges']:
            edge_strs = [f"'{e['cause'][:15]}'→'{e['effect'][:15]}'" for e in graph['edges'][:2]]
            causal_summary = f"\nCausal: {'; '.join(edge_strs)}"
        
        context = f"Topic: {topic}{causal_summary}\n\nEvidence:\n" + "\n".join(selected)
        return context[:max_chars]

retriever = CausalAwareRetriever()

# ============================================================================
# DATASET
# ============================================================================
class CausalRAGDataset(Dataset):
    def __init__(self, questions, docs, tokenizer, max_len=384, is_test=False):
        self.questions = questions
        self.docs = docs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self._cache = {}
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        qid = q.get('id')
        
        if qid not in self._cache:
            topic_info = self.docs.get(q.get('topic_id'), {})
            event = q.get('target_event', '')
            options = [q.get(f'option_{opt}', '') for opt in 'ABCD']
            context = retriever.retrieve(topic_info, event, options)
            self._cache[qid] = context
        
        context = self._cache[qid]
        event = q.get('target_event', '')
        
        # Shorter prompt for 32B
        options_text = "\n".join([f"{opt}: {q.get(f'option_{opt}', '')[:80]}" for opt in 'ABCD'])
        
        prompt = f"""Context:
{context}

Event: {event[:150]}

Options:
{options_text}

Which option(s) best explain the cause? Answer with letter(s) only (A, B, C, D or combinations like A,B)."""

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
# MODEL WITH QLORA
# ============================================================================
def create_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    print(f"  Loading {config.MODEL_NAME} with 4-bit quantization...")
    print(f"  ⚠️ This requires ~70GB VRAM - H100 80GB recommended!")
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config - Smaller for 32B
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    if config.USE_GRAD_CHECKPOINT:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        
    model.print_trainable_parameters()
    
    return model

class ClassificationHead(nn.Module):
    """Classification head on top of LLM"""
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),  # Smaller intermediate
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 4)  # 4 options
        )
    
    def forward(self, hidden_states):
        # Use last token's hidden state, cast to float32
        last_hidden = hidden_states[:, -1, :].float()
        return self.classifier(last_hidden)

class CausalRAG32B(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.classifier = ClassificationHead(hidden_size)
        self._classifier_moved = False
    
    def forward(self, input_ids, attention_mask):
        if not self._classifier_moved:
            self.classifier = self.classifier.to(input_ids.device)
            self._classifier_moved = True
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        logits = self.classifier(hidden_states)
        return logits

# ============================================================================
# TRAINING
# ============================================================================
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
        loss = criterion(logits, labels)
        loss = loss / config.GRADIENT_ACCUMULATION
        
        loss.backward()
        
        if (step + 1) % config.GRADIENT_ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Lower grad clip
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Memory cleanup every accumulation cycle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
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
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return np.array(all_preds), all_ids

def optimize_threshold(probs, golds):
    opts = list('ABCD')
    best_th, best_sc = 0.5, 0
    for th in np.arange(0.15, 0.75, 0.025):
        ps = [','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]])) for p in probs]
        sc = compute_aer_score(ps, golds)
        if sc > best_sc: best_sc, best_th = sc, th
    return best_th, best_sc

def create_submission(probs, ids, th, out_dir, name):
    opts = list('ABCD')
    results = [{'id': qid, 'answer': ','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]]))} for qid, p in zip(ids, probs)]
    sub = out_dir/'submission'; sub.mkdir(exist_ok=True)
    with open(sub/'submission.jsonl','w') as f:
        for r in results: f.write(json.dumps(r)+'\n')
    shutil.make_archive(str(out_dir/name),'zip',sub)
    return out_dir/f'{name}.zip'

# ============================================================================
# MAIN
# ============================================================================
def main():
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    
    print("\n[1/6] Loading data...")
    train_q, dev_q, test_q = load_questions('train_data'), load_questions('dev_data'), load_questions('test_data')
    train_docs, dev_docs, test_docs = load_docs('train_data'), load_docs('dev_data'), load_docs('test_data')
    print(f"  Train:{len(train_q)} Dev:{len(dev_q)} Test:{len(test_q)}")
    
    print("\n[2/6] Initializing Causal Retriever...")
    retriever._load()
    
    print("\n[3/6] Creating 32B model with QLoRA...")
    base_model = create_model()
    
    # Get hidden size
    hidden_size = base_model.config.hidden_size
    print(f"  Hidden size: {hidden_size}")
    
    # Create full model with classification head
    model = CausalRAG32B(base_model, hidden_size)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\n[4/6] Preparing datasets...")
    train_ds = CausalRAGDataset(train_q, train_docs, tokenizer, config.MAX_LENGTH)
    dev_ds = CausalRAGDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    test_ds = CausalRAGDataset(test_q, test_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, num_workers=0)
    
    # Only train classifier and LoRA weights
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.WARMUP_RATIO), total_steps)
    criterion = nn.BCEWithLogitsLoss()
    
    print("\n[5/6] Training CausalRAG 32B...")
    print(f"  ⚠️ This will take ~6-8 hours on H100 80GB!")
    dev_gold = [q.get('golden_answer','') for q in dev_q]
    best_score, best_th = 0, 0.5
    patience = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, epoch)
        
        dev_probs, _ = evaluate(model, dev_loader)
        th, score = optimize_threshold(dev_probs, dev_gold)
        print(f"  Epoch {epoch}: Loss={train_loss:.4f}, Dev={score:.4f} (th={th:.3f})")
        
        if score > best_score:
            best_score, best_th = score, th
            patience = 0
            # Save LoRA weights and classifier
            model.base_model.save_pretrained(config.OUTPUT_DIR/'lora_weights')
            torch.save({
                'classifier': model.classifier.state_dict(),
                'score': score,
                'threshold': th,
                'epoch': epoch
            }, config.OUTPUT_DIR/'best_model.pt')
            print(f"  -> New best! 🎉")
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print("  Early stopping!")
                break
    
    print(f"\n  Best Dev Score: {best_score:.4f}")
    
    # ⭐ DEV SET FINE-TUNING (1 Epoch)
    print("\n[5.5/6] Fine-tuning on DEV set (1 epoch)...")
    print("  Reloading best weights for fine-tuning...")
    
    # Reload LoRA
    model.base_model.load_adapter(config.OUTPUT_DIR/'lora_weights', 'finetune', is_trainable=True)
    model.base_model.set_adapter('finetune')
    
    # Reload Classifier
    ckpt = torch.load(config.OUTPUT_DIR/'best_model.pt', map_location='cpu', weights_only=False)
    model.classifier.load_state_dict(ckpt['classifier'])
    
    # Train 1 epoch on Dev with lower LR
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LEARNING_RATE * 0.5, weight_decay=0.01)
    # Re-calc steps for just 1 epoch on dev
    dev_steps = len(dev_loader) // config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(optimizer, dev_steps // 10, dev_steps)
    
    dev_loss = train_epoch(model, dev_loader, optimizer, scheduler, criterion, epoch=config.EPOCHS + 1)
    print(f"  Dev Finetune Loss: {dev_loss:.4f}")
    
    # Save as the final best model
    model.base_model.save_pretrained(config.OUTPUT_DIR/'lora_weights') # Overwrite default or save new? logic suggests overwrite for simplicity in inference
    torch.save({
        'classifier': model.classifier.state_dict(),
        'score': best_score, # Keep valid dev score
        'threshold': float(best_th),
        'epoch': 'dev_finetune'
    }, config.OUTPUT_DIR/'best_model.pt')
    
    print("\n[6/6] Predicting on TEST (using Dev-trained weights)...")
    # No need to reload, model is already in memory with dev-trained weights!
    # model.classifier.load_state_dict(ckpt['classifier']) # SKIP THIS
    
    test_probs, test_ids = evaluate(model, test_loader)
    zip_path = create_submission(test_probs, test_ids, ckpt['threshold'], config.OUTPUT_DIR, 'exp32_32b_submission')
    
    with open(config.OUTPUT_DIR/'results.json', 'w') as f:
        json.dump({
            'dev_score': float(best_score),
            'threshold': float(best_th),
            'model': config.MODEL_NAME,
            'lora_r': config.LORA_R,
            'baseline_exp22_7b': 0.86
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🏆 CausalRAG 32B Complete!")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  LoRA Rank: {config.LORA_R}")
    print(f"  Best Dev: {best_score:.4f}")
    print(f"  Baseline exp22_7b: 0.86 | Change: {best_score-0.86:+.4f}")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()