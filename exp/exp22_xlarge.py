# -*- coding: utf-8 -*-
"""
================================================================================
Exp22 XLarge: CausalRAG + Bigger Model ONLY
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

🎯 TARGET: 0.80+ (from 0.78)
🏆 SINGLE CHANGE: DeBERTa-v2-xlarge (900M) instead of v3-large (435M)

⚠️ NO OTHER CHANGES - Same as exp22 that got 0.78
   - Same BCE loss (no multi-task)
   - Same training loop (no FGM, no SWA, no TTA)
   - Same hyperparameters
   - Just bigger model + gradient checkpointing for memory

GPU: H100/A100 80GB
Time: ~4-5 hours
================================================================================
"""

import json, random, shutil, warnings, re
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
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
        OUTPUT_DIR = Path('/kaggle/working/exp22xl_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp22xl_output')
    
    # ⭐ ONLY CHANGE: Bigger model
    MODEL_NAME = 'microsoft/deberta-v2-xlarge'  # 900M params (was v3-large 435M)
    MAX_LENGTH = 384  # Reduced for memory (was 512)
    
    # ⭐ Memory optimization for larger model
    GRADIENT_CHECKPOINTING = True
    
    # CausalRAG Settings - SAME AS EXP22
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
    
    # RAG Settings - SAME AS EXP22
    CHUNK_SIZE = 350
    TOP_K_CHUNKS = 3
    MAX_CONTEXT = 1500
    
    # Training - SAME AS EXP22 (just smaller batch for memory)
    SEED = 42
    EPOCHS = 5
    BATCH_SIZE = 2  # Reduced from 4 for memory
    GRADIENT_ACCUMULATION = 8  # Increased to compensate
    LEARNING_RATE = 1e-5
    PATIENCE = 3
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*70}")
print(f"Exp22 XLarge: CausalRAG + Bigger Model")
print(f"{'='*70}")
print(f"Model: {config.MODEL_NAME}")
print(f"Gradient Checkpointing: {config.GRADIENT_CHECKPOINTING}")
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
# CAUSAL GRAPH BUILDER - SAME AS EXP22
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
# CAUSAL RETRIEVER - SAME AS EXP22
# ============================================================================
class CausalAwareRetriever:
    def __init__(self):
        self.embed_model = None
    
    def _load(self):
        if self.embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except:
                import subprocess
                subprocess.run(['pip', 'install', 'sentence-transformers', '-q'])
                from sentence_transformers import SentenceTransformer
            
            print("  Loading embedding model...")
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def retrieve(self, topic_info, event, options, max_chars=1500):
        if not topic_info:
            return "", []
        
        self._load()
        
        topic = topic_info.get('topic', '')
        query = f"{event} " + " ".join([opt[:60] for opt in options])
        
        # Build causal graph
        graph = causal_builder.build_graph(topic_info)
        
        # Collect chunks
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
                chunk_meta.append({'has_causal': has_causal, 'title': title})
        
        if not chunks:
            return f"Topic: {topic}", []
        
        # Semantic similarity
        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        chunk_embs = self.embed_model.encode(chunks, convert_to_tensor=True)
        sims = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        
        # Causal boost
        for i, meta in enumerate(chunk_meta):
            if meta['has_causal']:
                sims[i] *= config.CAUSAL_EDGE_BOOST
        
        # Select top chunks
        top_idx = np.argsort(sims)[-config.TOP_K_CHUNKS:][::-1]
        selected = [chunks[i] for i in top_idx]
        
        # Causal graph summary
        causal_summary = ""
        if graph['edges']:
            edge_strs = [f"'{e['cause'][:25]}' → '{e['effect'][:25]}'" for e in graph['edges'][:3]]
            causal_summary = f"\nCausal: {'; '.join(edge_strs)}"
        
        context = f"Topic: {topic}{causal_summary}\n\nEvidence:\n" + "\n---\n".join(selected)
        return context[:max_chars], graph['edges']

retriever = CausalAwareRetriever()

# ============================================================================
# DATASET - SAME AS EXP22
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
            context, _ = retriever.retrieve(topic_info, event, options)
            self._cache[qid] = context
        
        context = self._cache[qid]
        event = q.get('target_event', '')
        
        input_ids_list, attention_mask_list = [], []
        for opt in 'ABCD':
            option = q.get(f'option_{opt}', '')
            text = f"{context}\n[SEP] Event: {event}\n[SEP] Cause: {option}"
            enc = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
            input_ids_list.append(enc['input_ids'].squeeze(0))
            attention_mask_list.append(enc['attention_mask'].squeeze(0))
        
        result = {
            'id': qid,
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
# MODEL - SAME AS EXP22 BUT WITH GRADIENT CHECKPOINTING
# ============================================================================
class CausalRAGModel(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Enable gradient checkpointing for memory
        if config.GRADIENT_CHECKPOINTING:
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        flat_ids = input_ids.view(-1, input_ids.size(-1))
        flat_mask = attention_mask.view(-1, attention_mask.size(-1))
        
        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        
        logits = self.classifier(cls).squeeze(-1).view(batch_size, 4)
        return logits

# ============================================================================
# TRAINING - SAME AS EXP22
# ============================================================================
def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / config.GRADIENT_ACCUMULATION
        
        scaler.scale(loss).backward()
        
        if (step + 1) % config.GRADIENT_ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
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

def create_submission(probs, ids, th, out_dir, name):
    opts = list('ABCD')
    results = [{'id': qid, 'answer': ','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]]))} for qid, p in zip(ids, probs)]
    sub = out_dir/'submission'; sub.mkdir(exist_ok=True)
    with open(sub/'submission.jsonl','w') as f:
        for r in results: f.write(json.dumps(r)+'\n')
    shutil.make_archive(str(out_dir/name),'zip',sub)
    return out_dir/f'{name}.zip'

# ============================================================================
# MAIN - SAME AS EXP22
# ============================================================================
def main():
    print("\n[1/6] Loading data...")
    train_q, dev_q, test_q = load_questions('train_data'), load_questions('dev_data'), load_questions('test_data')
    train_docs, dev_docs, test_docs = load_docs('train_data'), load_docs('dev_data'), load_docs('test_data')
    print(f"  Train:{len(train_q)} Dev:{len(dev_q)} Test:{len(test_q)}")
    
    print("\n[2/6] Initializing Causal Retriever...")
    retriever._load()
    
    print("\n[3/6] Preparing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    train_ds = CausalRAGDataset(train_q, train_docs, tokenizer, config.MAX_LENGTH)
    dev_ds = CausalRAGDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    test_ds = CausalRAGDataset(test_q, test_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    
    print("\n[4/6] Creating model...")
    print(f"  Loading {config.MODEL_NAME}...")
    model = CausalRAGModel(config.MODEL_NAME).to(config.DEVICE)
    print(f"  Gradient Checkpointing: {config.GRADIENT_CHECKPOINTING}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    print("\n[5/6] Training CausalRAG XLarge...")
    dev_gold = [q.get('golden_answer','') for q in dev_q]
    best_score, best_th = 0, 0.5
    patience = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, epoch)
        
        dev_probs, _ = evaluate(model, dev_loader)
        th, score = optimize_threshold(dev_probs, dev_gold)
        print(f"  Epoch {epoch}: Loss={train_loss:.4f}, Dev={score:.4f} (th={th:.3f})")
        
        if score > best_score:
            best_score, best_th = score, th
            patience = 0
            torch.save({
                'model': model.state_dict(),
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
    
    print("\n[6/6] Predicting on TEST...")
    ckpt = torch.load(config.OUTPUT_DIR/'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(config.DEVICE)
    
    test_probs, test_ids = evaluate(model, test_loader)
    zip_path = create_submission(test_probs, test_ids, ckpt['threshold'], config.OUTPUT_DIR, 'exp22xl_submission')
    
    with open(config.OUTPUT_DIR/'results.json', 'w') as f:
        json.dump({
            'dev_score': float(best_score),
            'threshold': float(best_th),
            'model': config.MODEL_NAME,
            'baseline_exp22': 0.78
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🏆 CausalRAG XLarge Complete!")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Best Dev: {best_score:.4f}")
    print(f"  Baseline exp22: 0.78 | Change: {best_score-0.78:+.4f}")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
