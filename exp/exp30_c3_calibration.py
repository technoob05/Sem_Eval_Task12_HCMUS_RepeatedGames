# -*- coding: utf-8 -*-
"""
================================================================================
Exp30: C3 - Causal-Aware Confidence Calibration
================================================================================
🎯 NOVEL CONTRIBUTION: First to combine LLM internal states + causal structure
   for confidence estimation in reasoning tasks

📊 METHOD: Calibration network that learns from BOTH:
   - LLM hidden states (uncertainty within model)
   - Causal graph topology (uncertainty in evidence)

🏆 TARGET: 0.89+ (from 0.86) by rejecting low-confidence predictions

REQUIREMENTS: Trained exp22_7b model (LoRA weights + classifier)
GPU: H100 80GB
Time: ~2-3 hours (calibration training + inference)
================================================================================
"""

import json, random, shutil, warnings, re, gc, subprocess, sys
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
warnings.filterwarnings('ignore')

def install_deps():
    packages = ['bitsandbytes', 'peft', 'accelerate', 'sentence-transformers']
    for pkg in packages:
        try: __import__(pkg)
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=True)
install_deps()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        MODEL_DIR = Path('/kaggle/input/basline-semeval-task12/exp22_7b_output')
        OUTPUT_DIR = Path('/kaggle/working/exp30_c3_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        MODEL_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp22_7b_output')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp30_c3_output')
    
    MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
    MAX_LENGTH = 512
    CHUNK_SIZE = 350
    TOP_K_CHUNKS = 3
    MAX_CONTEXT = 1200
    
    # Causal patterns
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
    
    # C3 Calibration settings
    CALIBRATION_EPOCHS = 10
    CALIBRATION_LR = 1e-3
    CALIBRATION_BATCH = 8
    CONFIDENCE_THRESHOLD = 0.3  # Lower threshold - only reject very uncertain ones
    
    SEED = 42
    BATCH_SIZE = 1
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

def load_questions(split):
    with open(config.DATA_DIR/split/'questions.jsonl','r',encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def load_docs(split):
    with open(config.DATA_DIR/split/'docs.json','r',encoding='utf-8') as f:
        return {d['topic_id']:d for d in json.load(f)}

# ============================================================================
# CAUSAL GRAPH + RETRIEVER (Same as exp22)
# ============================================================================
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
                        if len(cause) > 5 and len(effect) > 5:
                            edges.append({'cause': cause, 'effect': effect, 'type': edge_type})
        return edges[:config.MAX_GRAPH_EDGES]
    
    def build_graph(self, topic_info):
        if not topic_info: return {'nodes': set(), 'edges': []}
        all_edges = []
        for doc in topic_info.get('docs', []):
            content = doc.get('content', '') or doc.get('snippet', '')
            all_edges.extend(self.extract_causal_edges(content))
        nodes = set()
        for e in all_edges: nodes.add(e['cause']); nodes.add(e['effect'])
        return {'nodes': nodes, 'edges': all_edges}

causal_builder = CausalGraphBuilder()

class CausalAwareRetriever:
    def __init__(self): self.embed_model = None
    
    def _load(self):
        if self.embed_model is None:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def retrieve(self, topic_info, event, options):
        if not topic_info: return "", {}, []
        self._load()
        
        topic = topic_info.get('topic', '')
        query = f"{event} " + " ".join([opt[:60] for opt in options])
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
        
        if not chunks: return f"Topic: {topic}", graph, [0.0, 0.0, 0.0]
        
        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        chunk_embs = self.embed_model.encode(chunks, convert_to_tensor=True)
        sims = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        for i, meta in enumerate(chunk_meta):
            if meta['has_causal']: sims[i] *= config.CAUSAL_EDGE_BOOST
        
        top_idx = np.argsort(sims)[-config.TOP_K_CHUNKS:][::-1]
        selected = [chunks[i] for i in top_idx]
        selected_sims = [sims[i] for i in top_idx]
        
        context = f"Topic: {topic}\n\nEvidence:\n" + "\n".join(selected)
        return context[:config.MAX_CONTEXT], graph, selected_sims

retriever = CausalAwareRetriever()

# ============================================================================
# ⭐ NOVELTY: C3 - CAUSAL-AWARE CONFIDENCE CALIBRATION
# ============================================================================
class CausalConfidenceFeatureExtractor:
    """Extract features from BOTH LLM hidden states AND causal graph"""
    
    @staticmethod
    def extract_llm_features(hidden_states, logits):
        """Features from LLM internal states"""
        # hidden_states: [batch, seq_len, hidden_dim]
        # logits: [batch, 4]
        
        probs = torch.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1)[0]
        
        # Entropy (uncertainty)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        # Hidden state statistics (last token)
        last_hidden = hidden_states[:, -1, :]
        hidden_mean = last_hidden.mean(dim=-1)
        hidden_std = last_hidden.std(dim=-1)
        
        # L2 norm (activation magnitude)
        hidden_norm = torch.norm(last_hidden, p=2, dim=-1)
        
        return torch.stack([max_prob, entropy, hidden_mean, hidden_std, hidden_norm], dim=1)
    
    @staticmethod
    def extract_causal_features(graph, retrieval_sims):
        """Features from causal graph structure"""
        # Graph connectivity
        num_nodes = len(graph.get('nodes', set()))
        num_edges = len(graph.get('edges', []))
        edge_density = num_edges / max(num_nodes, 1)
        
        # Retrieval quality
        avg_sim = np.mean(retrieval_sims) if retrieval_sims else 0.0
        max_sim = np.max(retrieval_sims) if retrieval_sims else 0.0
        
        return torch.tensor([num_nodes, num_edges, edge_density, avg_sim, max_sim], dtype=torch.float32)

class C3CalibrationNetwork(nn.Module):
    """
    Learns to calibrate confidence using BOTH:
    - LLM internal uncertainty (5 features)
    - Causal graph quality (5 features)
    """
    def __init__(self):
        super().__init__()
        self.calibrator = nn.Sequential(
            nn.Linear(10, 32),  # 5 LLM + 5 Causal = 10 features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),  # Output: calibrated confidence [0,1]
            nn.Sigmoid()
        )
    
    def forward(self, llm_features, causal_features):
        # llm_features: [batch, 5]
        # causal_features: [batch, 5]
        combined = torch.cat([llm_features, causal_features], dim=1)  # [batch, 10]
        confidence = self.calibrator(combined)  # [batch, 1]
        return confidence.squeeze(-1)  # [batch]

# ============================================================================
# MODEL COMPONENTS
# ============================================================================
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 4)
        )
    def forward(self, hidden_states):
        # Cast to float32 for classifier (base model outputs bfloat16)
        return self.classifier(hidden_states[:, -1, :].float())

class CausalRAG7B(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.classifier = ClassificationHead(hidden_size)
        self._classifier_moved = False
    
    def forward(self, input_ids, attention_mask, return_hidden=False):
        if not self._classifier_moved:
            self.classifier = self.classifier.to(input_ids.device)
            self._classifier_moved = True
        
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        logits = self.classifier(hidden_states)
        
        if return_hidden:
            return logits, hidden_states
        return logits

# ============================================================================
# DATASET
# ============================================================================
class C3Dataset(Dataset):
    def __init__(self, questions, docs, tokenizer, max_len=512, is_test=False):
        self.questions, self.docs, self.tokenizer = questions, docs, tokenizer
        self.max_len, self.is_test = max_len, is_test
        self._cache = {}
    
    def __len__(self): return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        qid = q.get('id')
        
        if qid not in self._cache:
            topic_info = self.docs.get(q.get('topic_id'), {})
            event = q.get('target_event', '')
            options = [q.get(f'option_{opt}', '') for opt in 'ABCD']
            context, graph, retrieval_sims = retriever.retrieve(topic_info, event, options)
            self._cache[qid] = (context, graph, retrieval_sims)
        
        context, graph, retrieval_sims = self._cache[qid]
        options_text = "\n".join([f"{opt}: {q.get(f'option_{opt}', '')}" for opt in 'ABCD'])
        prompt = f"Context:\n{context}\n\nEvent: {q.get('target_event','')}\n\nOptions:\n{options_text}\n\nWhich option(s) best explain the cause of the event? Answer with the letter(s) only (e.g., A or A,B)."
        
        enc = self.tokenizer(prompt, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        
        result = {
            'id': qid,
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'graph_nodes': len(graph.get('nodes', set())),
            'graph_edges': len(graph.get('edges', [])),
            'retrieval_sims': retrieval_sims
        }
        
        if not self.is_test:
            golden = q.get('golden_answer', '')
            labels = torch.zeros(4)
            for ans in golden.split(','):
                if ans.strip().upper() in 'ABCD':
                    labels['ABCD'.index(ans.strip().upper())] = 1.0
            result['labels'] = labels
            
            # For calibration: correctness label (1 if any answer is correct)
            result['is_correct'] = 1.0 if labels.sum() > 0 else 0.0
        
        return result

# ============================================================================
# MAIN TRAINING + INFERENCE
# ============================================================================
def main():
    print(f"\n{'='*70}")
    print("Exp30: C3 - Causal-Aware Confidence Calibration")
    print(f"{'='*70}\n")
    
    print("[1/6] Loading data...")
    train_q, dev_q, test_q = load_questions('train_data'), load_questions('dev_data'), load_questions('test_data')
    train_docs, dev_docs, test_docs = load_docs('train_data'), load_docs('dev_data'), load_docs('test_data')
    retriever._load()
    
    print("\n[2/6] Loading trained Qwen-7B model...")
    print(f"  MODEL_DIR: {config.MODEL_DIR}")
    print(f"  Looking for LoRA: {config.MODEL_DIR/'lora_weights'}")
    print(f"  Looking for classifier: {config.MODEL_DIR/'best_model.pt'}")
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    
    # Load LoRA weights
    lora_path = config.MODEL_DIR/'lora_weights'
    if lora_path.exists():
        print(f"  ✅ Loading LoRA weights from {lora_path}")
        base_model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        print(f"  ⚠️ WARNING: LoRA weights not found at {lora_path}!")
    
    model = CausalRAG7B(base_model, base_model.config.hidden_size)
    
    # Load classifier checkpoint
    ckpt_path = config.MODEL_DIR/'best_model.pt'
    if ckpt_path.exists():
        print(f"  ✅ Loading classifier from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.classifier.load_state_dict(ckpt['classifier'])
        print(f"  ✅ Model loaded successfully!")
    else:
        print(f"  ⚠️ WARNING: Classifier checkpoint not found at {ckpt_path}!")
        print(f"  ⚠️ Using untrained classifier - results may be poor!")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    print("\n[3/6] Extracting features for calibration training...")
    dev_ds = C3Dataset(dev_q, dev_docs, tokenizer, config.MAX_LENGTH)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Collect features + labels
    all_llm_feats, all_causal_feats, all_correctness = [], [], []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Extracting dev features"):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            
            logits, hidden_states = model(input_ids, attention_mask, return_hidden=True)
            
            # LLM features
            llm_feats = CausalConfidenceFeatureExtractor.extract_llm_features(hidden_states, logits)
            
            # Causal features
            graph_nodes = batch['graph_nodes'].float()
            graph_edges = batch['graph_edges'].float()
            edge_density = graph_edges / torch.clamp(graph_nodes, min=1.0)
            
            retrieval_sims = batch['retrieval_sims']
            avg_sim = sum(retrieval_sims[0]) / len(retrieval_sims[0]) if len(retrieval_sims[0]) > 0 else 0.0
            max_sim = max(retrieval_sims[0]) if len(retrieval_sims[0]) > 0 else 0.0
            
            causal_feats = torch.tensor([[graph_nodes[0], graph_edges[0], edge_density[0], avg_sim, max_sim]], dtype=torch.float32)
            
            # Correctness (for training calibrator)
            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).float().cpu()
            labels = batch['labels']
            is_correct = (predicted == labels).all(dim=1).float()
            
            all_llm_feats.append(llm_feats.cpu())
            all_causal_feats.append(causal_feats)
            all_correctness.append(is_correct)
    
    all_llm_feats = torch.cat(all_llm_feats, dim=0)
    all_causal_feats = torch.cat(all_causal_feats, dim=0)
    all_correctness = torch.cat(all_correctness, dim=0)
    
    print(f"\n[4/6] Training C3 calibration network...")
    calibrator = C3CalibrationNetwork().to(config.DEVICE)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=config.CALIBRATION_LR)
    criterion = nn.BCELoss()
    
    # Training loop
    calibrator.train()
    for epoch in range(config.CALIBRATION_EPOCHS):
        perm = torch.randperm(len(all_llm_feats))
        total_loss = 0
        
        for i in range(0, len(all_llm_feats), config.CALIBRATION_BATCH):
            idx = perm[i:i+config.CALIBRATION_BATCH]
            llm_batch = all_llm_feats[idx].to(config.DEVICE)
            causal_batch = all_causal_feats[idx].to(config.DEVICE)
            target = all_correctness[idx].to(config.DEVICE)
            
            optimizer.zero_grad()
            pred_confidence = calibrator(llm_batch, causal_batch)
            loss = criterion(pred_confidence, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"  Epoch {epoch+1}/{config.CALIBRATION_EPOCHS}: Loss={total_loss/(len(all_llm_feats)//config.CALIBRATION_BATCH):.4f}")
    
    # Save calibrator
    torch.save(calibrator.state_dict(), config.OUTPUT_DIR/'calibrator.pt')
    
    print(f"\n[5/6] Inference on TEST with confidence filtering...")
    test_ds = C3Dataset(test_q, test_docs, tokenizer, config.MAX_LENGTH, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model.eval()
    calibrator.eval()
    
    all_preds, all_confidences, all_ids = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test inference"):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            
            logits, hidden_states = model(input_ids, attention_mask, return_hidden=True)
            llm_feats = CausalConfidenceFeatureExtractor.extract_llm_features(hidden_states, logits).to(config.DEVICE)
            
            # Causal features
            graph_nodes = batch['graph_nodes'].float().to(config.DEVICE)
            graph_edges = batch['graph_edges'].float().to(config.DEVICE)
            edge_density = graph_edges / torch.clamp(graph_nodes, min=1.0)
            retrieval_sims = batch['retrieval_sims']
            avg_sim = sum(retrieval_sims[0]) / len(retrieval_sims[0]) if len(retrieval_sims[0]) > 0 else 0.0
            max_sim = max(retrieval_sims[0]) if len(retrieval_sims[0]) > 0 else 0.0
            causal_feats = torch.tensor([[graph_nodes[0], graph_edges[0], edge_density[0], avg_sim, max_sim]], dtype=torch.float32).to(config.DEVICE)
            
            # Get calibrated confidence
            confidence = calibrator(llm_feats, causal_feats).cpu().item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs[0])
            all_confidences.append(confidence)
            all_ids.append(batch['id'][0])
    
    print(f"\n[6/6] Creating submission with confidence threshold={config.CONFIDENCE_THRESHOLD}...")
    opts = list('ABCD')
    results = []
    rejected_count = 0
    
    for qid, p, conf in zip(all_ids, all_preds, all_confidences):
        if conf < config.CONFIDENCE_THRESHOLD:
            # Low confidence → use argmax as safe fallback (NOT default 'A')
            ans = opts[np.argmax(p)]
            rejected_count += 1
        else:
            # High confidence → use model prediction with threshold
            ans = [opts[i] for i in range(4) if p[i] >= 0.5]
            if not ans: ans = [opts[np.argmax(p)]]
            ans = ','.join(sorted(ans))
        
        results.append({'id': qid, 'answer': ans})
    
    with open(config.OUTPUT_DIR/'submission.jsonl', 'w') as f:
        for r in results: f.write(json.dumps(r)+'\n')
    
    shutil.make_archive(str(config.OUTPUT_DIR/'exp30_c3_submission'), 'zip', config.OUTPUT_DIR)
    
    print(f"\n{'='*70}")
    print(f"🏆 C3 Calibration Complete!")
    print(f"  Rejected {rejected_count}/{len(all_ids)} low-confidence predictions")
    print(f"  Avg confidence: {np.mean(all_confidences):.3f}")
    print(f"  Submission: {config.OUTPUT_DIR/'exp30_c3_submission.zip'}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
