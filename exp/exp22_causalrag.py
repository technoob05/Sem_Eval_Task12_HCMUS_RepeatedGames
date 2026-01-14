# -*- coding: utf-8 -*-
"""
================================================================================
Exp22: CausalRAG - Causal Graph Enhanced Retrieval
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

📚 PAPER: "CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation"
   Venue: ACL 2025 Findings
   Key Idea: Build causal graph from evidence, retrieve along causal paths

🔥 NOVELTY:
1. Causal Graph Construction from documents
2. Causal Path-based Retrieval (not just similarity)
3. Cause-Effect Edge Scoring
4. Graph-Aware Context Assembly

GPU: H100 or A100
Time: ~3-4 hours
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
        OUTPUT_DIR = Path('/kaggle/working/exp22_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp22_output')
    
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 512
    
    # ⭐ CausalRAG Settings
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
    CAUSAL_EDGE_BOOST = 1.5   # Boost for causal edges in retrieval
    MAX_GRAPH_EDGES = 20
    
    # RAG Settings  
    CHUNK_SIZE = 350
    TOP_K_CHUNKS = 3
    MAX_CONTEXT = 1500
    
    # Training
    SEED = 42
    EPOCHS = 5
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 1e-5
    PATIENCE = 3
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*70}")
print(f"Exp22: CausalRAG - Causal Graph Enhanced Retrieval")
print(f"{'='*70}")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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
# ⭐ CAUSAL GRAPH BUILDER
# ============================================================================
class CausalGraphBuilder:
    """
    Extract causal relationships from text to build a causal graph.
    Nodes: Events/Entities
    Edges: Causal relationships (cause, effect, temporal)
    """
    
    def __init__(self):
        self.patterns = [(re.compile(p, re.IGNORECASE), t) for p, t in config.CAUSAL_PATTERNS]
    
    def extract_causal_edges(self, text):
        """Extract causal edges from text."""
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
        """Build causal graph from all documents in a topic."""
        if not topic_info:
            return {'nodes': set(), 'edges': [], 'sentences': []}
        
        all_edges = []
        all_sentences = []
        
        for doc in topic_info.get('docs', []):
            content = doc.get('content', '') or doc.get('snippet', '')
            edges = self.extract_causal_edges(content)
            all_edges.extend(edges)
            all_sentences.extend([e['sentence'] for e in edges])
        
        # Extract unique nodes
        nodes = set()
        for e in all_edges:
            nodes.add(e['cause'])
            nodes.add(e['effect'])
        
        return {
            'nodes': nodes,
            'edges': all_edges,
            'sentences': all_sentences
        }

causal_builder = CausalGraphBuilder()

# ============================================================================
# ⭐ CAUSAL-AWARE RETRIEVER
# ============================================================================
class CausalAwareRetriever:
    """
    Retrieve content based on both semantic similarity AND causal relevance.
    Boosts chunks that contain causal relationships matching the query.
    """
    
    def __init__(self):
        self.model = None
    
    def _load(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except:
                import subprocess
                subprocess.run(['pip', 'install', 'sentence-transformers', '-q'])
                from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def retrieve(self, topic_info, event, options, max_chars=1500):
        """Retrieve with causal graph awareness."""
        if not topic_info:
            return "", []
        
        self._load()
        
        topic = topic_info.get('topic', '')
        query = f"{event} " + " ".join([opt[:50] for opt in options])
        
        # Build causal graph
        graph = causal_builder.build_graph(topic_info)
        
        # Collect chunks
        chunks = []
        chunk_meta = []
        
        for doc in topic_info.get('docs', []):
            title = doc.get('title', '')
            content = doc.get('content', '') or doc.get('snippet', '')
            
            # Split into chunks
            for i in range(0, len(content), config.CHUNK_SIZE - 50):
                chunk = content[i:i+config.CHUNK_SIZE]
                if len(chunk) < 50:
                    continue
                
                # Check if chunk contains causal edges
                has_causal = any(
                    e['cause'].lower() in chunk.lower() or e['effect'].lower() in chunk.lower()
                    for e in graph['edges']
                )
                
                chunks.append(f"[{title}] {chunk}")
                chunk_meta.append({'has_causal': has_causal, 'title': title})
        
        if not chunks:
            return f"Topic: {topic}", []
        
        # Semantic similarity
        query_emb = self.model.encode(query, convert_to_tensor=True)
        chunk_embs = self.model.encode(chunks, convert_to_tensor=True)
        sims = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        
        # ⭐ Boost chunks with causal edges
        for i, meta in enumerate(chunk_meta):
            if meta['has_causal']:
                sims[i] *= config.CAUSAL_EDGE_BOOST
        
        # Get top chunks
        top_idx = np.argsort(sims)[-config.TOP_K_CHUNKS:][::-1]
        selected = [chunks[i] for i in top_idx]
        
        # Add causal graph summary
        causal_summary = ""
        if graph['edges']:
            edge_strs = [f"'{e['cause'][:30]}' → '{e['effect'][:30]}'" for e in graph['edges'][:3]]
            causal_summary = f"\nCausal Relations: {'; '.join(edge_strs)}"
        
        context = f"Topic: {topic}{causal_summary}\n\nEvidence:\n" + "\n---\n".join(selected)
        return context[:max_chars], graph['edges']

retriever = CausalAwareRetriever()

# ============================================================================
# DATASET
# ============================================================================
class CausalRAGDataset(Dataset):
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
            context, _ = retriever.retrieve(topic_info, event, options)
            self._cache[qid] = context
        
        context = self._cache[qid]
        event = q.get('target_event', '')
        
        input_ids_list, attention_mask_list = [], []
        for opt in 'ABCD':
            text = f"{context}\n[SEP] Event: {event}\n[SEP] Cause: {q.get(f'option_{opt}', '')}"
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
# MODEL
# ============================================================================
class CausalRAGModel(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
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
# TRAINING & EVALUATION
# ============================================================================
def train_epoch(model, loader, optimizer, scheduler, criterion, scaler):
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        
        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / config.GRADIENT_ACCUMULATION
        
        scaler.scale(loss).backward()
        
        if (step + 1) % config.GRADIENT_ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
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

def optimize_threshold(probs, golds):
    opts = ['A','B','C','D']
    best_th, best_sc = 0.5, 0
    for th in np.arange(0.20, 0.70, 0.05):
        ps = [','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]])) for p in probs]
        sc = compute_aer_score(ps, golds)
        if sc > best_sc: best_sc, best_th = sc, th
    return best_th, best_sc

def create_predictions(probs, ids, th):
    opts = ['A','B','C','D']
    return [{'id': qid, 'answer': ','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]]))} for qid, p in zip(ids, probs)]

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
    print("\n[1/5] Loading data...")
    train_q, dev_q, test_q = load_questions('train_data'), load_questions('dev_data'), load_questions('test_data')
    train_docs, dev_docs, test_docs = load_docs('train_data'), load_docs('dev_data'), load_docs('test_data')
    print(f"  Train:{len(train_q)} Dev:{len(dev_q)} Test:{len(test_q)}")
    
    print("\n[2/5] Preparing CausalRAG datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    retriever._load()
    
    train_ds = CausalRAGDataset(train_q, train_docs, tokenizer, config.MAX_LENGTH)
    dev_ds = CausalRAGDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    test_ds = CausalRAGDataset(test_q, test_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    
    print("\n[3/5] Creating model...")
    model = CausalRAGModel(config.MODEL_NAME).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    print("\n[4/5] Training CausalRAG...")
    dev_gold = [q.get('golden_answer','') for q in dev_q]
    best_score, best_th = 0, 0.5
    patience = 0
    
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler)
        dev_probs, _ = evaluate(model, dev_loader)
        th, score = optimize_threshold(dev_probs, dev_gold)
        print(f"  Loss: {loss:.4f}, Dev: {score:.4f} (th={th:.2f})")
        
        if score > best_score:
            best_score, best_th = score, th
            patience = 0
            torch.save({'model': model.state_dict(), 'threshold': th}, config.OUTPUT_DIR/'best_model.pt')
            print("  -> New best!")
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print("  Early stopping!")
                break
    
    print("\n[5/5] Generating submission...")
    ckpt = torch.load(config.OUTPUT_DIR/'best_model.pt', weights_only=False)
    model.load_state_dict(ckpt['model'])
    test_probs, test_ids = evaluate(model, test_loader)
    preds = create_predictions(test_probs, test_ids, ckpt['threshold'])
    zip_path = create_submission(preds, config.OUTPUT_DIR, 'exp22_submission')
    
    print(f"\n{'='*70}")
    print(f"🔥 CausalRAG Complete! Dev: {best_score:.4f}")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
