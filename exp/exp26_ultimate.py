# -*- coding: utf-8 -*-
"""
================================================================================
Exp26: ULTIMATE - Maximum Performance for Top 1 Leaderboard
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

🎯 TARGET: 0.89+ (Current Top 1)
🏆 CURRENT BEST: 0.78 (exp22 CausalRAG)

🔥 FIXED FOR MEMORY:
1. ⭐ MODEL: DeBERTa-v2-xlarge (900M) - fits on GPU
2. ⭐ Gradient Checkpointing: Enabled
3. ⭐ R-Drop: Disabled (too memory intensive)
4. ⭐ MAX_LENGTH: 384 (reduced from 512)
5. ⭐ All other techniques: CausalRAG + ICCL + TTA + SWA + FGM

GPU: H100/A100 80GB or 2xT4
Time: ~4-5 hours
================================================================================
"""

import json, random, shutil, warnings, os, re, gc
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup

# ============================================================================
# TPU/GPU DETECTION
# ============================================================================
IS_TPU = 'TPU_NAME' in os.environ or 'COLAB_TPU_ADDR' in os.environ

if IS_TPU:
    try:
        import torch_xla.core.xla_model as xm
        DEVICE = xm.xla_device()
        print(f"🚀 Running on TPU: {DEVICE}")
    except:
        IS_TPU = False
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {DEVICE}")

# Clear memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================================
# CONFIG - MEMORY OPTIMIZED
# ============================================================================
class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp26_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp26_output')
    
    # ⭐ FIXED: Use xlarge instead of xxlarge
    MODEL_NAME = 'microsoft/deberta-v2-xlarge'  # 900M params (fits on 80GB)
    MAX_LENGTH = 384  # Reduced from 512
    
    # ⭐ MEMORY OPTIMIZATION
    GRADIENT_CHECKPOINTING = True
    USE_FP16 = True  # Mixed precision
    
    # ⭐ ADVANCED RAG
    USE_MULTI_STAGE_RAG = True
    STAGE1_TOP_DOCS = 4  # Reduced from 5
    STAGE2_CHUNK_SIZE = 300
    STAGE2_TOP_CHUNKS = 3  # Reduced from 4
    USE_RERANKING = True
    MAX_CONTEXT = 1500  # Reduced from 2000
    
    # ⭐ CAUSAL PATTERNS (from CausalRAG)
    CAUSAL_PATTERNS = [
        r'(.+?) caused (.+)', r'(.+?) led to (.+)', r'(.+?) resulted in (.+)',
        r'because of (.+?), (.+)', r'(.+?) triggered (.+)', r'(.+?) prompted (.+)',
        r'after (.+?), (.+)', r'following (.+?), (.+)',
    ]
    CAUSAL_BOOST = 1.3
    
    # ⭐ COUNTERFACTUAL (from CF-RAG)
    USE_COUNTERFACTUAL = True
    CF_AUGMENT_PROB = 0.1
    
    # ⭐ CAUSAL CHAIN PROMPTS (from C2P)
    USE_CHAIN_PROMPTS = True
    
    # ⭐ IN-CONTEXT DEMOS (from ICCL)
    USE_ICCL = True
    NUM_DEMOS = 2
    
    # Training - MEMORY OPTIMIZED
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
    
    # ⭐ ADVANCED TRAINING (Memory-friendly options only)
    USE_SWA = True
    SWA_START_EPOCH = 4
    USE_RDROP = False  # ❌ DISABLED - requires 2x memory
    USE_FGM = True
    FGM_EPSILON = 0.3
    
    # ⭐ MULTI-TASK LOSS
    BCE_WEIGHT = 1.0
    CONTRASTIVE_WEIGHT = 0.15
    FOCAL_WEIGHT = 0.1
    CONTRASTIVE_MARGIN = 0.35
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.08
    
    # ⭐ TEST-TIME AUGMENTATION
    USE_TTA = True
    TTA_ROUNDS = 3

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"{'='*70}")
print(f"🏆 Exp26: ULTIMATE - Target Top 1 (0.89+)")
print(f"{'='*70}")
print(f"Model: {config.MODEL_NAME}")
print(f"Gradient Checkpointing: {config.GRADIENT_CHECKPOINTING}")
print(f"MAX_LENGTH: {config.MAX_LENGTH}")
print(f"Multi-Stage RAG: {config.USE_MULTI_STAGE_RAG}")
print(f"ICCL + CausalRAG + CF-RAG + C2P: Combined")
print(f"SWA: {config.USE_SWA}, FGM: {config.USE_FGM}")
print(f"TTA: {config.USE_TTA}, R-Drop: {config.USE_RDROP}")
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
# ⭐ ADVANCED MULTI-STAGE RAG WITH RERANKING
# ============================================================================
class AdvancedRAG:
    def __init__(self):
        self.embed_model = None
        self.rerank_model = None
        self.causal_patterns = [re.compile(p, re.IGNORECASE) for p in config.CAUSAL_PATTERNS]
        self._demo_embs = None
        self._demo_questions = None
    
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
    
    def has_causal(self, text):
        return any(p.search(text) for p in self.causal_patterns)
    
    def chunk_document(self, text, chunk_size=300):
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        
        for sent in sentences:
            if len(current) + len(sent) <= chunk_size:
                current += " " + sent
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sent
        if current.strip():
            chunks.append(current.strip())
        
        return chunks if chunks else [text[:chunk_size]]
    
    def retrieve(self, topic_info, event, options, max_chars=1500):
        if not topic_info:
            return "", {}
        
        self._load()
        
        topic = topic_info.get('topic', '')
        query = f"{event} " + " ".join([opt[:60] for opt in options])
        
        docs = topic_info.get('docs', [])
        if not docs:
            return f"Topic: {topic}", {}
        
        # Stage 1: Document scoring
        doc_texts = []
        for doc in docs:
            title = doc.get('title', '')
            snippet = doc.get('snippet', '')
            content = doc.get('content', '')[:400]
            doc_texts.append(f"{title} {snippet} {content}")
        
        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        doc_embs = self.embed_model.encode(doc_texts, convert_to_tensor=True)
        doc_scores = torch.cosine_similarity(query_emb.unsqueeze(0), doc_embs).cpu().numpy()
        
        top_doc_idx = np.argsort(doc_scores)[-config.STAGE1_TOP_DOCS:][::-1]
        
        # Stage 2: Chunk retrieval
        all_chunks = []
        chunk_meta = []
        
        for idx in top_doc_idx:
            doc = docs[idx]
            title = doc.get('title', '')
            content = doc.get('content', '') or doc.get('snippet', '')
            
            chunks = self.chunk_document(content, config.STAGE2_CHUNK_SIZE)
            for chunk in chunks:
                has_causal = self.has_causal(chunk)
                all_chunks.append(f"[{title}] {chunk}")
                chunk_meta.append({
                    'title': title,
                    'doc_score': float(doc_scores[idx]),
                    'has_causal': has_causal
                })
        
        if not all_chunks:
            return f"Topic: {topic}", {}
        
        # Semantic scoring
        chunk_embs = self.embed_model.encode(all_chunks, convert_to_tensor=True)
        chunk_scores = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        
        # Causal boost
        for i, meta in enumerate(chunk_meta):
            if meta['has_causal']:
                chunk_scores[i] *= config.CAUSAL_BOOST
        
        # Reranking
        if config.USE_RERANKING and self.rerank_model:
            top_k_for_rerank = min(8, len(all_chunks))
            top_idx = np.argsort(chunk_scores)[-top_k_for_rerank:][::-1]
            
            pairs = [(query, all_chunks[i]) for i in top_idx]
            rerank_scores = self.rerank_model.predict(pairs)
            
            reranked = sorted(zip(top_idx, rerank_scores), key=lambda x: x[1], reverse=True)
            final_idx = [x[0] for x in reranked[:config.STAGE2_TOP_CHUNKS]]
        else:
            final_idx = np.argsort(chunk_scores)[-config.STAGE2_TOP_CHUNKS:][::-1]
        
        selected = [all_chunks[i] for i in final_idx]
        
        # Build causal summary
        causal_edges = []
        for i in final_idx:
            if chunk_meta[i]['has_causal']:
                for pattern in self.causal_patterns:
                    match = pattern.search(all_chunks[i])
                    if match and len(match.groups()) >= 2:
                        causal_edges.append(f"'{match.group(1)[:25]}' → '{match.group(2)[:25]}'")
                        break
        
        causal_summary = f"\nCausal: {'; '.join(causal_edges[:2])}" if causal_edges else ""
        
        context = f"Topic: {topic}{causal_summary}\n\nEvidence:\n" + "\n---\n".join(selected)
        return context[:max_chars], {'causal_edges': causal_edges}
    
    def build_demo_index(self, train_questions):
        print("  Building ICCL demo index...")
        texts = [f"{q.get('target_event', '')} {q.get('option_A', '')[:40]}" for q in train_questions]
        self._demo_embs = self.embed_model.encode(texts, convert_to_tensor=True)
        self._demo_questions = train_questions
    
    def get_demos(self, event, options, num=2):
        if self._demo_embs is None:
            return []
        
        query = f"{event} {options[0][:40]}"
        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        sims = torch.cosine_similarity(query_emb.unsqueeze(0), self._demo_embs).cpu().numpy()
        
        top_idx = np.argsort(sims)[-num*2:][::-1]
        
        demos = []
        for idx in top_idx:
            q = self._demo_questions[idx]
            demos.append({
                'event': q.get('target_event', '')[:60],
                'options': {o: q.get(f'option_{o}', '')[:30] for o in 'ABCD'},
                'answer': q.get('golden_answer', '')
            })
            if len(demos) >= num:
                break
        return demos
    
    def format_demos(self, demos):
        if not demos:
            return ""
        lines = ["Examples:"]
        for i, d in enumerate(demos, 1):
            lines.append(f"E{i}: {d['event']} → {d['answer']}")
        return "\n".join(lines) + "\n\n"

rag = AdvancedRAG()

# ============================================================================
# ⭐ CAUSAL CHAIN PROMPT (C2P)
# ============================================================================
def generate_chain_prompt(event, option):
    return f"Chain: Did '{option[:60]}' cause '{event[:60]}'?"

# ============================================================================
# DATASET
# ============================================================================
class UltimateDataset(Dataset):
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
            context, _ = rag.retrieve(topic_info, event, options)
            self._cache[qid] = context
        
        context = self._cache[qid]
        event = q.get('target_event', '')
        golden = q.get('golden_answer', '') if not self.is_test else ''
        
        # ICCL demos
        if config.USE_ICCL and not self.is_test and rag._demo_embs is not None:
            options = [q.get(f'option_{opt}', '') for opt in 'ABCD']
            demos = rag.get_demos(event, options, config.NUM_DEMOS)
            demo_text = rag.format_demos(demos)
        else:
            demo_text = ""
        
        input_ids_list, attention_mask_list = [], []
        for opt in 'ABCD':
            option = q.get(f'option_{opt}', '')
            
            parts = []
            if demo_text:
                parts.append(demo_text)
            parts.append(context)
            
            if config.USE_CHAIN_PROMPTS:
                parts.append(generate_chain_prompt(event, option))
            
            parts.append(f"[SEP] Event: {event}")
            parts.append(f"[SEP] Cause: {option}")
            
            text = "\n".join(parts)
            
            # CF augmentation
            if config.USE_COUNTERFACTUAL and not self.is_test and random.random() < config.CF_AUGMENT_PROB:
                text = text.replace("caused", "did not cause")
            
            enc = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
            input_ids_list.append(enc['input_ids'].squeeze(0))
            attention_mask_list.append(enc['attention_mask'].squeeze(0))
        
        result = {
            'id': qid,
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
        }
        
        if not self.is_test:
            labels = torch.zeros(4)
            for ans in golden.split(','):
                ans = ans.strip().upper()
                if ans in 'ABCD':
                    labels['ABCD'.index(ans)] = 1.0
            result['labels'] = labels
        
        return result

# ============================================================================
# ⭐ ULTIMATE MODEL WITH GRADIENT CHECKPOINTING
# ============================================================================
class UltimateModel(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Enable gradient checkpointing with use_reentrant=False for autocast compatibility
        if config.GRADIENT_CHECKPOINTING:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
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
# FGM ADVERSARIAL
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
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        smooth_labels = labels * (1 - config.LABEL_SMOOTHING) + config.LABEL_SMOOTHING / 4
        
        # Mixed precision forward
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
            
            if IS_TPU:
                xm.mark_step()
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION
        pbar.set_postfix({'loss': f'{total_loss/(step+1):.4f}'})
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, use_tta=False):
    model.eval()
    all_preds, all_ids = [], []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        
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
        
        if IS_TPU:
            xm.mark_step()
    
    return np.array(all_preds), all_ids

def optimize_threshold(probs, golds):
    opts = list('ABCD')
    best_th, best_sc = 0.5, 0
    for th in np.arange(0.15, 0.75, 0.025):
        ps = [','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]])) for p in probs]
        sc = compute_aer_score(ps, golds)
        if sc > best_sc: best_sc, best_th = sc, th
    return best_th, best_sc

def create_submission(preds, ids, th, out_dir, name):
    opts = list('ABCD')
    results = [{'id': qid, 'answer': ','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]]))} for qid, p in zip(ids, preds)]
    sub = out_dir/'submission'; sub.mkdir(exist_ok=True)
    with open(sub/'submission.jsonl','w') as f:
        for r in results: f.write(json.dumps(r)+'\n')
    shutil.make_archive(str(out_dir/name),'zip',sub)
    return out_dir/f'{name}.zip'

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n[1/7] Loading data...")
    train_q, dev_q, test_q = load_questions('train_data'), load_questions('dev_data'), load_questions('test_data')
    train_docs, dev_docs, test_docs = load_docs('train_data'), load_docs('dev_data'), load_docs('test_data')
    print(f"  Train:{len(train_q)} Dev:{len(dev_q)} Test:{len(test_q)}")
    
    print("\n[2/7] Initializing Advanced RAG...")
    rag._load()
    if config.USE_ICCL:
        rag.build_demo_index(train_q)
    
    print("\n[3/7] Preparing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    train_ds = UltimateDataset(train_q, train_docs, tokenizer, config.MAX_LENGTH)
    dev_ds = UltimateDataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    test_ds = UltimateDataset(test_q, test_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
    
    print("\n[4/7] Creating ULTIMATE model...")
    print(f"  Loading {config.MODEL_NAME}...")
    model = UltimateModel(config.MODEL_NAME).to(DEVICE)
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
    
    print("\n[5/7] Training ULTIMATE...")
    dev_gold = [q.get('golden_answer','') for q in dev_q]
    best_score, best_th = 0, 0.5
    patience = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterions, fgm, scaler, epoch)
        
        # Update SWA
        if config.USE_SWA and epoch >= config.SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # Evaluate
        dev_probs, _ = evaluate(model, dev_loader, use_tta=True)
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
    
    # SWA model evaluation
    swa_score = 0
    if config.USE_SWA and swa_model:
        print("\n  Updating SWA BatchNorm...")
        try:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)
            swa_probs, _ = evaluate(swa_model, dev_loader, use_tta=True)
            swa_th, swa_score = optimize_threshold(swa_probs, dev_gold)
            print(f"  SWA Dev Score: {swa_score:.4f}")
            
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
    
    print(f"\n  Best Dev Score: {best_score:.4f}")
    
    print("\n[6/7] Predicting on TEST with TTA...")
    ckpt = torch.load(config.OUTPUT_DIR/'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(DEVICE)
    
    test_probs, test_ids = evaluate(model, test_loader, use_tta=True)
    
    print("\n[7/7] Generating submission...")
    zip_path = create_submission(test_probs, test_ids, ckpt['threshold'], config.OUTPUT_DIR, 'exp26_submission')
    
    with open(config.OUTPUT_DIR/'results.json', 'w') as f:
        json.dump({
            'dev_score': float(best_score),
            'threshold': float(best_th),
            'model': config.MODEL_NAME,
            'techniques': ['CausalRAG', 'ICCL', 'CF-RAG', 'C2P', 'SWA', 'FGM', 'TTA', 'Focal']
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🏆 ULTIMATE Training Complete!")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Best Dev: {best_score:.4f}")
    print(f"  Target: 0.89 | Gap: {0.89 - best_score:.4f}")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
