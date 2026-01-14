# -*- coding: utf-8 -*-
"""
================================================================================
Exp22 V2: CausalRAG SOTA - Maximum Performance
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

🎯 TARGET: 0.85+ (from 0.78)
🏆 UPGRADES:
1. ⭐ Uses BOTH train_data + dev_data for training (+22% data)
2. ⭐ DeBERTa-v2-xlarge (900M params)
3. ⭐ Cross-encoder reranking
4. ⭐ Multi-task loss (BCE + Contrastive + Focal)
5. ⭐ FGM Adversarial + TTA + SWA
6. ⭐ Gradient Checkpointing + FP16

GPU: H100/A100 80GB
Time: ~5-6 hours
================================================================================
"""

import json, random, shutil, warnings, re, gc
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import get_cosine_schedule_with_warmup

# ============================================================================
# CONFIG - MAXIMUM PERFORMANCE
# ============================================================================
class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp22v2_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp22v2_output')
    
    # ⭐ BIGGER MODEL
    MODEL_NAME = 'microsoft/deberta-v2-xlarge'  # 900M params
    MAX_LENGTH = 384
    
    # ⭐ MEMORY OPTIMIZATION
    GRADIENT_CHECKPOINTING = True
    USE_FP16 = True
    
    # ⭐ USE DEV DATA FOR TRAINING
    USE_DEV_FOR_TRAIN = True  # Adds 400 more samples!
    
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
    CAUSAL_EDGE_BOOST = 1.5
    MAX_GRAPH_EDGES = 20
    
    # ⭐ ADVANCED RAG
    TOP_K_DOCS = 4
    CHUNK_SIZE = 350
    TOP_K_CHUNKS = 3
    MAX_CONTEXT = 1800
    USE_RERANKING = True
    
    # Training
    SEED = 42
    EPOCHS = 6
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 16
    LEARNING_RATE = 5e-6
    MIN_LR = 1e-7
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    PATIENCE = 3
    
    # ⭐ ADVANCED TRAINING
    USE_SWA = True
    SWA_START_EPOCH = 4
    USE_FGM = True
    FGM_EPSILON = 0.3
    
    # ⭐ MULTI-TASK LOSS
    BCE_WEIGHT = 1.0
    CONTRASTIVE_WEIGHT = 0.15
    FOCAL_WEIGHT = 0.1
    CONTRASTIVE_MARGIN = 0.35
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.08
    
    # ⭐ TTA
    USE_TTA = True
    TTA_ROUNDS = 3
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"{'='*70}")
print(f"🚀 Exp22 V2: CausalRAG SOTA - Target 0.85+")
print(f"{'='*70}")
print(f"Model: {config.MODEL_NAME}")
print(f"Use dev for training: {config.USE_DEV_FOR_TRAIN}")
print(f"Gradient Checkpointing: {config.GRADIENT_CHECKPOINTING}")
print(f"MAX_LENGTH: {config.MAX_LENGTH}")
print(f"SWA: {config.USE_SWA}, FGM: {config.USE_FGM}, TTA: {config.USE_TTA}")
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
# ⭐ CAUSAL GRAPH BUILDER
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
# ⭐ ADVANCED CAUSAL RETRIEVER WITH RERANKING
# ============================================================================
class AdvancedCausalRetriever:
    def __init__(self):
        self.embed_model = None
        self.rerank_model = None
    
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
            
            if config.USE_RERANKING:
                print("  Loading reranking model...")
                try:
                    from sentence_transformers import CrossEncoder
                    self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                except Exception as e:
                    print(f"  Warning: Could not load reranker: {e}")
                    self.rerank_model = None
    
    def retrieve(self, topic_info, event, options, max_chars=1800):
        if not topic_info:
            return "", []
        
        self._load()
        
        topic = topic_info.get('topic', '')
        query = f"{event} " + " ".join([opt[:60] for opt in options])
        
        graph = causal_builder.build_graph(topic_info)
        
        # Collect chunks with full content
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
        
        # Reranking
        if config.USE_RERANKING and self.rerank_model:
            top_k_for_rerank = min(8, len(chunks))
            top_idx = np.argsort(sims)[-top_k_for_rerank:][::-1]
            
            pairs = [(query, chunks[i]) for i in top_idx]
            rerank_scores = self.rerank_model.predict(pairs)
            
            reranked = sorted(zip(top_idx, rerank_scores), key=lambda x: x[1], reverse=True)
            final_idx = [x[0] for x in reranked[:config.TOP_K_CHUNKS]]
        else:
            final_idx = np.argsort(sims)[-config.TOP_K_CHUNKS:][::-1]
        
        selected = [chunks[i] for i in final_idx]
        
        # Causal graph summary
        causal_summary = ""
        if graph['edges']:
            edge_strs = [f"'{e['cause'][:25]}' → '{e['effect'][:25]}'" for e in graph['edges'][:3]]
            causal_summary = f"\nCausal: {'; '.join(edge_strs)}"
        
        context = f"Topic: {topic}{causal_summary}\n\nEvidence:\n" + "\n---\n".join(selected)
        return context[:max_chars], graph['edges']

retriever = AdvancedCausalRetriever()

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
# ⭐ MODEL WITH GRADIENT CHECKPOINTING
# ============================================================================
class CausalRAGModelV2(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
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
        
        self.projection = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        batch_size = input_ids.size(0)
        flat_ids = input_ids.view(-1, input_ids.size(-1))
        flat_mask = attention_mask.view(-1, attention_mask.size(-1))
        
        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        
        logits = self.classifier(cls).squeeze(-1).view(batch_size, 4)
        
        if return_embeddings:
            embeddings = self.projection(cls).view(batch_size, 4, -1)
            return logits, embeddings
        return logits

# ============================================================================
# LOSSES
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal = ((1 - p_t) ** self.gamma) * ce
        return focal.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.35):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        loss = 0.0
        count = 0
        
        for i in range(batch_size):
            emb = F.normalize(embeddings[i], dim=-1)
            lbl = labels[i]
            
            pos_mask = lbl == 1
            neg_mask = lbl == 0
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_emb = emb[pos_mask]
                neg_emb = emb[neg_mask]
                
                for p in pos_emb:
                    neg_sim = (p.unsqueeze(0) * neg_emb).sum(dim=-1)
                    triplet = F.relu(neg_sim - 0.5 + self.margin).mean()
                    loss += triplet
                    count += 1
        
        return loss / max(count, 1)

# ============================================================================
# FGM
# ============================================================================
class FGM:
    def __init__(self, model, epsilon=0.3):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}
    
    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'word_embeddings' in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    param.data.add_(self.epsilon * param.grad / norm)
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, loader, optimizer, scheduler, criterions, fgm, scaler, epoch):
    model.train()
    bce_crit, con_crit, focal_crit = criterions
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)
        
        smooth_labels = labels * (1 - config.LABEL_SMOOTHING) + config.LABEL_SMOOTHING / 4
        
        with torch.cuda.amp.autocast(enabled=config.USE_FP16):
            logits, emb = model(input_ids, attention_mask, return_embeddings=True)
            bce_loss = bce_crit(logits, smooth_labels)
            con_loss = con_crit(emb, labels)
            focal_loss = focal_crit(logits, labels)
            
            loss = (config.BCE_WEIGHT * bce_loss + 
                    config.CONTRASTIVE_WEIGHT * con_loss + 
                    config.FOCAL_WEIGHT * focal_loss)
            loss = loss / config.GRADIENT_ACCUMULATION
        
        scaler.scale(loss).backward()
        
        # FGM adversarial - do before the regular unscale
        already_unscaled = False
        if config.USE_FGM and fgm and (step + 1) % config.GRADIENT_ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            already_unscaled = True
            fgm.attack()
            with torch.cuda.amp.autocast(enabled=config.USE_FP16):
                logits_adv = model(input_ids, attention_mask)
                loss_adv = bce_crit(logits_adv, smooth_labels) / config.GRADIENT_ACCUMULATION
            scaler.scale(loss_adv).backward()
            fgm.restore()
        
        if (step + 1) % config.GRADIENT_ACCUMULATION == 0:
            if not already_unscaled:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION
        pbar.set_postfix({'loss': f'{total_loss/(step+1):.4f}'})
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, use_tta=False):
    model.eval()
    all_preds, all_ids = [], []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        
        if use_tta and config.USE_TTA:
            preds = []
            for _ in range(config.TTA_ROUNDS):
                model.train()
                with torch.no_grad():
                    logits = model(input_ids, attention_mask)
                    preds.append(torch.sigmoid(logits).cpu().numpy())
            model.eval()
            avg_preds = np.mean(preds, axis=0)
            all_preds.extend(avg_preds)
        else:
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

def create_predictions(probs, ids, th):
    opts = list('ABCD')
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
    print("\n[1/7] Loading data...")
    train_q = load_questions('train_data')
    dev_q = load_questions('dev_data')
    test_q = load_questions('test_data')
    
    train_docs = load_docs('train_data')
    dev_docs = load_docs('dev_data')
    test_docs = load_docs('test_data')
    
    # ⭐ MERGE TRAIN + DEV FOR TRAINING
    if config.USE_DEV_FOR_TRAIN:
        combined_q = train_q + dev_q
        combined_docs = {**train_docs, **dev_docs}
        print(f"  Train: {len(train_q)} + Dev: {len(dev_q)} = {len(combined_q)} samples!")
        # Use a portion of original train as validation
        val_size = int(len(train_q) * 0.15)
        val_q = train_q[-val_size:]
        train_q_final = combined_q[:-val_size] if val_size > 0 else combined_q
    else:
        train_q_final = train_q
        val_q = dev_q
        combined_docs = train_docs
    
    print(f"  Final Train: {len(train_q_final)}, Val: {len(val_q)}, Test: {len(test_q)}")
    
    print("\n[2/7] Initializing Advanced Retriever...")
    retriever._load()
    
    print("\n[3/7] Preparing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    train_ds = CausalRAGDataset(train_q_final, combined_docs, tokenizer, config.MAX_LENGTH)
    val_ds = CausalRAGDataset(val_q, combined_docs, tokenizer, config.MAX_LENGTH, is_test=False)
    test_ds = CausalRAGDataset(test_q, test_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    
    print("\n[4/7] Creating model...")
    print(f"  Loading {config.MODEL_NAME}...")
    model = CausalRAGModelV2(config.MODEL_NAME).to(config.DEVICE)
    print(f"  Gradient Checkpointing: {config.GRADIENT_CHECKPOINTING}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * config.EPOCHS // config.GRADIENT_ACCUMULATION
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * config.WARMUP_RATIO), total_steps)
    
    bce_crit = nn.BCEWithLogitsLoss()
    con_crit = ContrastiveLoss(config.CONTRASTIVE_MARGIN)
    focal_crit = FocalLoss(config.FOCAL_GAMMA)
    criterions = (bce_crit, con_crit, focal_crit)
    
    fgm = FGM(model, config.FGM_EPSILON) if config.USE_FGM else None
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_FP16)
    
    # SWA setup
    swa_model = None
    if config.USE_SWA:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=config.MIN_LR)
    
    print("\n[5/7] Training CausalRAG V2...")
    val_gold = [q.get('golden_answer','') for q in val_q]
    best_score, best_th = 0, 0.5
    patience = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterions, fgm, scaler, epoch)
        
        if config.USE_SWA and epoch >= config.SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        val_probs, _ = evaluate(model, val_loader, use_tta=True)
        th, score = optimize_threshold(val_probs, val_gold)
        print(f"  Epoch {epoch}: Loss={train_loss:.4f}, Val={score:.4f} (th={th:.3f})")
        
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
    
    # SWA evaluation
    swa_score = 0
    if config.USE_SWA and swa_model:
        print("\n  Updating SWA BatchNorm...")
        try:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=config.DEVICE)
            swa_probs, _ = evaluate(swa_model, val_loader, use_tta=True)
            swa_th, swa_score = optimize_threshold(swa_probs, val_gold)
            print(f"  SWA Val Score: {swa_score:.4f}")
            
            if swa_score > best_score:
                best_score, best_th = swa_score, swa_th
                torch.save({
                    'model': swa_model.module.state_dict(),
                    'score': swa_score,
                    'threshold': swa_th,
                    'epoch': -1
                }, config.OUTPUT_DIR/'best_model.pt')
                print(f"  -> SWA is better! 🎉")
        except Exception as e:
            print(f"  SWA update failed: {e}")
    
    print(f"\n  Best Val Score: {best_score:.4f}")
    
    print("\n[6/7] Predicting on TEST...")
    ckpt = torch.load(config.OUTPUT_DIR/'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(config.DEVICE)
    
    test_probs, test_ids = evaluate(model, test_loader, use_tta=True)
    preds = create_predictions(test_probs, test_ids, ckpt['threshold'])
    
    print("\n[7/7] Generating submission...")
    zip_path = create_submission(preds, config.OUTPUT_DIR, 'exp22v2_submission')
    
    with open(config.OUTPUT_DIR/'results.json', 'w') as f:
        json.dump({
            'val_score': float(best_score),
            'threshold': float(best_th),
            'model': config.MODEL_NAME,
            'use_dev_for_train': config.USE_DEV_FOR_TRAIN,
            'techniques': ['CausalRAG', 'CrossEncoder', 'FGM', 'SWA', 'TTA', 'MultiTaskLoss']
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🏆 CausalRAG V2 Complete!")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Best Val: {best_score:.4f}")
    print(f"  Baseline: 0.78 | Improvement: +{best_score-0.78:.4f}")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
