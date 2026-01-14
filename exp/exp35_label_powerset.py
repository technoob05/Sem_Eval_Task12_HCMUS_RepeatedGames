# -*- coding: utf-8 -*-
"""
================================================================================
Exp35: LABEL POWERSET - 15-Class Multi-class Classification
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

🎯 KEY INSIGHT: AER metric rewards EXACT SET MATCH = 1.0, SUBSET = 0.5
📊 BCE treats each option independently -> misses correlations
✅ SOLUTION: 15-class classification (all possible answer combinations)

Classes: A, B, C, D, A+B, A+C, A+D, B+C, B+D, C+D, A+B+C, A+B+D, A+C+D, B+C+D, A+B+C+D

Includes: Fix pooling from exp34

GPU: H100 80GB (required)
================================================================================
"""

import json, random, shutil, warnings, re, gc, subprocess, sys
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
from itertools import combinations
warnings.filterwarnings('ignore')

def install_qlora_deps():
    packages = ['bitsandbytes', 'peft', 'accelerate']
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=True)

install_qlora_deps()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ⭐ Generate all 15 possible answer combinations (label powerset)
def generate_label_powerset():
    """Generate all non-empty subsets of {A, B, C, D} -> 15 classes"""
    options = ['A', 'B', 'C', 'D']
    powerset = []
    for r in range(1, len(options) + 1):
        for combo in combinations(options, r):
            powerset.append(','.join(combo))
    return powerset

LABEL_POWERSET = generate_label_powerset()  # ['A', 'B', 'C', 'D', 'A,B', 'A,C', ...]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABEL_POWERSET)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}
NUM_CLASSES = len(LABEL_POWERSET)  # 15

print(f"Label Powerset ({NUM_CLASSES} classes): {LABEL_POWERSET}")

class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp35_label_powerset_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp35_label_powerset_output')
    
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
    
    CHUNK_SIZE = 350
    TOP_K_CHUNKS = 3
    MAX_CONTEXT = 1200
    
    SEED = 42
    EPOCHS = 3
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 16
    LEARNING_RATE = 2e-4
    WARMUP_RATIO = 0.1
    PATIENCE = 2
    
    # ⭐ Label Powerset specific
    NUM_CLASSES = NUM_CLASSES
    LABEL_SMOOTHING = 0.1
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"{'='*70}")
print(f"Exp35: LABEL POWERSET - 15-Class Classification")
print(f"{'='*70}")
print(f"✅ 15 classes for all possible answer combinations")
print(f"✅ CrossEntropy loss learns joint distribution")
print(f"✅ Includes pooling fix from exp34")
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

def gold_to_class_idx(golden_answer):
    """Convert golden answer string to class index"""
    # Normalize: sort and join
    parts = sorted([x.strip().upper() for x in golden_answer.split(',') if x.strip()])
    normalized = ','.join(parts)
    return LABEL_TO_IDX.get(normalized, 0)  # Default to 'A' if unknown

def class_idx_to_answer(idx):
    """Convert class index back to answer string"""
    return IDX_TO_LABEL.get(idx, 'A')

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
            try:
                from sentence_transformers import SentenceTransformer
            except:
                subprocess.run(['pip', 'install', 'sentence-transformers', '-q'])
                from sentence_transformers import SentenceTransformer
            
            print("  Loading embedding model...")
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def retrieve(self, topic_info, event, options, max_chars=1200):
        if not topic_info:
            return ""
        
        self._load()
        
        topic = topic_info.get('topic', '')
        query = f"{event} " + " ".join([opt[:60] for opt in options])
        
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
        
        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        chunk_embs = self.embed_model.encode(chunks, convert_to_tensor=True)
        sims = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        
        for i, meta in enumerate(chunk_meta):
            if meta['has_causal']:
                sims[i] *= config.CAUSAL_EDGE_BOOST
        
        top_idx = np.argsort(sims)[-config.TOP_K_CHUNKS:][::-1]
        selected = [chunks[i] for i in top_idx]
        
        causal_summary = ""
        if graph['edges']:
            edge_strs = [f"'{e['cause'][:20]}'->{e['effect'][:20]}'" for e in graph['edges'][:2]]
            causal_summary = f"\nCausal: {'; '.join(edge_strs)}"
        
        context = f"Topic: {topic}{causal_summary}\n\nEvidence:\n" + "\n".join(selected)
        return context[:max_chars]

retriever = CausalAwareRetriever()

# ============================================================================
# DATASET - Label Powerset
# ============================================================================
class LabelPowersetDataset(Dataset):
    def __init__(self, questions, docs, tokenizer, max_len=512, is_test=False):
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
            # ⭐ Convert to class index instead of binary vector
            label_idx = gold_to_class_idx(golden)
            result['labels'] = torch.tensor(label_idx, dtype=torch.long)
        
        return result

# ============================================================================
# MODEL WITH LABEL POWERSET HEAD
# ============================================================================
def create_model():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except:
        subprocess.run(['pip', 'install', 'peft', 'bitsandbytes', 'accelerate', '-q'])
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    print(f"  Loading {config.MODEL_NAME} with 4-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

# ⭐ 15-Class Classification Head
class LabelPowersetHead(nn.Module):
    """15-class classification head for label powerset"""
    def __init__(self, hidden_size, num_classes=15):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)  # 15 classes
        )
    
    def forward(self, hidden_states, attention_mask):
        # ⭐ Fix: Get last NON-PAD token
        idx = attention_mask.sum(dim=1) - 1
        idx = idx.clamp(min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, idx]
        
        return self.classifier(last_hidden)

class CausalRAGPowerset(nn.Module):
    def __init__(self, base_model, hidden_size, num_classes=15):
        super().__init__()
        self.base_model = base_model
        self.classifier = LabelPowersetHead(hidden_size, num_classes)
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
        logits = self.classifier(hidden_states, attention_mask)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION
        pbar.set_postfix({'loss': f'{total_loss/(step+1):.4f}'})
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_ids, all_probs = [], [], []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_ids.extend(batch['id'])
    
    # Convert class indices to answer strings
    answers = [class_idx_to_answer(idx) for idx in all_preds]
    return answers, all_ids, np.array(all_probs)

def create_submission(answers, ids, out_dir, name):
    results = [{'id': qid, 'answer': ans} for qid, ans in zip(ids, answers)]
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
    
    # Analyze label distribution
    label_dist = defaultdict(int)
    for q in train_q:
        golden = q.get('golden_answer', '')
        label_dist[gold_to_class_idx(golden)] += 1
    print(f"  Label distribution: {dict(sorted(label_dist.items()))}")
    
    print("\n[2/6] Initializing Causal Retriever...")
    retriever._load()
    
    print("\n[3/6] Creating model with Label Powerset Head...")
    base_model = create_model()
    
    hidden_size = base_model.config.hidden_size
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num classes: {NUM_CLASSES}")
    
    model = CausalRAGPowerset(base_model, hidden_size, NUM_CLASSES)
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\n[4/6] Preparing datasets...")
    train_ds = LabelPowersetDataset(train_q, train_docs, tokenizer, config.MAX_LENGTH)
    dev_ds = LabelPowersetDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    test_ds = LabelPowersetDataset(test_q, test_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, num_workers=0)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.WARMUP_RATIO), total_steps)
    
    # ⭐ CrossEntropyLoss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    print("\n[5/6] Training with Label Powerset...")
    dev_gold = [q.get('golden_answer','') for q in dev_q]
    best_score = 0
    patience = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, epoch)
        
        dev_answers, _, _ = evaluate(model, dev_loader)
        score = compute_aer_score(dev_answers, dev_gold)
        print(f"  Epoch {epoch}: Loss={train_loss:.4f}, Dev={score:.4f}")
        
        if score > best_score:
            best_score = score
            patience = 0
            model.base_model.save_pretrained(config.OUTPUT_DIR/'lora_weights')
            torch.save({
                'classifier': model.classifier.state_dict(),
                'score': score,
                'epoch': epoch
            }, config.OUTPUT_DIR/'best_model.pt')
            print(f"  -> New best! 🎉")
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print("  Early stopping!")
                break
    
    print(f"\n  Best Dev Score: {best_score:.4f}")
    
    print("\n[6/6] Predicting on TEST...")
    ckpt = torch.load(config.OUTPUT_DIR/'best_model.pt', map_location='cpu', weights_only=False)
    model.classifier.load_state_dict(ckpt['classifier'])
    
    test_answers, test_ids, _ = evaluate(model, test_loader)
    zip_path = create_submission(test_answers, test_ids, config.OUTPUT_DIR, 'exp35_label_powerset_submission')
    
    with open(config.OUTPUT_DIR/'results.json', 'w') as f:
        json.dump({
            'experiment': 'exp35_label_powerset',
            'method': '15_class_classification',
            'dev_score': float(best_score),
            'model': config.MODEL_NAME,
            'num_classes': NUM_CLASSES,
            'label_smoothing': config.LABEL_SMOOTHING,
            'baseline_exp22': 0.78
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🏆 Exp35: LABEL POWERSET Complete!")
    print(f"  📊 15-class classification instead of BCE")
    print(f"  ✅ Includes pooling fix")
    print(f"  Best Dev: {best_score:.4f}")
    print(f"  Baseline exp22: 0.78 | Change: {best_score-0.78:+.4f}")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
