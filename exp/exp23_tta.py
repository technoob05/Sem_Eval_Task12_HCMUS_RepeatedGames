# -*- coding: utf-8 -*-
"""
================================================================================
Exp23 TTA: Test Time Augmentation for CausalRAG 7B
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

🎯 GOAL: Boost 0.86 score to 0.88+ using TTA
🔧 METHOD: Shuffle evidence chunks 5 times & average logits
REQUIREMENT: Must have exp22_7b_output/best_model.pt
================================================================================
"""

import json, random, shutil, warnings, re, gc, subprocess, sys
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
warnings.filterwarnings('ignore')

# ⭐ Install required packages
def install_deps():
    packages = ['bitsandbytes', 'peft', 'accelerate', 'sentence-transformers']
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=True)

install_deps()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        MODEL_DIR = Path('/kaggle/working/exp22_7b_output')  # Assumes previous output is available or uploaded
        OUTPUT_DIR = Path('/kaggle/working/exp23_tta_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        MODEL_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp22_7b_output')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp23_tta_output')
    
    # Must match exp22_7b
    MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
    MAX_LENGTH = 512
    CHUNK_SIZE = 350
    TOP_K_CHUNKS = 3
    MAX_CONTEXT = 1200
    
    # TTA Settings
    TTA_N = 5  # Number of augmentations
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1

    # Causal Patterns
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

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# UTILS & CLASSES (Copied from exp22_7b)
# ============================================================================
def load_questions(split):
    with open(config.DATA_DIR/split/'questions.jsonl','r',encoding='utf-8') as f: return [json.loads(l) for l in f]

def load_docs(split):
    with open(config.DATA_DIR/split/'docs.json','r',encoding='utf-8') as f: return {d['topic_id']:d for d in json.load(f)}

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

class CausalAwareRetriever:
    def __init__(self): self.embed_model = None
    
    def _load(self):
        if self.embed_model is None:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def retrieve(self, topic_info, event, options, max_chars=1200):
        if not topic_info: return ""
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
        
        if not chunks: return f"Topic: {topic}"
        
        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        chunk_embs = self.embed_model.encode(chunks, convert_to_tensor=True)
        sims = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        for i, meta in enumerate(chunk_meta):
            if meta['has_causal']: sims[i] *= config.CAUSAL_EDGE_BOOST
        
        selected = [chunks[i] for i in np.argsort(sims)[-config.TOP_K_CHUNKS:][::-1]]
        
        return selected, topic, graph

retriever = CausalAwareRetriever()

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_size//2, 4))
    def forward(self, hidden_states): return self.classifier(hidden_states[:, -1, :])

class CausalRAG7B(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.classifier = ClassificationHead(hidden_size)
        self._classifier_moved = False
    
    def forward(self, input_ids, attention_mask):
        if not self._classifier_moved:
            self.classifier = self.classifier.to(input_ids.device)
            self._classifier_moved = True
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return self.classifier(outputs.hidden_states[-1])

# ============================================================================
# TTA LOGIC
# ============================================================================
class TTADataset(Dataset):
    def __init__(self, questions, docs, tokenizer, max_len=512):
        self.questions, self.docs, self.tokenizer = questions, docs, tokenizer
        self.max_len = max_len
        
    def __len__(self): return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        qid = q.get('id')
        topic_info = self.docs.get(q.get('topic_id'), {})
        
        # 1. Retrieve chunks
        evidence_chunks, topic, graph = retriever.retrieve(topic_info, q.get('target_event', ''), [q.get(f'option_{opt}', '') for opt in 'ABCD'])
        
        causal_summary = f"\nCausal: {'; '.join([f'{e['cause'][:20]}->{e['effect'][:20]}' for e in graph['edges'][:2]])}" if graph.get('edges') else ""
        
        # 2. Generate TTA Variants
        variants = []
        
        # V0: Original Order
        variants.append(evidence_chunks)
        
        # V1-4: Random Shuffle
        for _ in range(config.TTA_N - 1):
            shuffled = evidence_chunks[:]
            random.shuffle(shuffled)
            variants.append(shuffled)
            
        # 3. Tokenize all variants
        batch = {'input_ids': [], 'attention_mask': [], 'id': qid}
        
        for v_chunks in variants:
            context = f"Topic: {topic}{causal_summary}\n\nEvidence:\n" + "\n".join(v_chunks)
            context = context[:config.MAX_CONTEXT]
            
            prompt = f"Context:\n{context}\n\nEvent: {q.get('target_event','')}\n\nOptions:\n" + "\n".join([f"{opt}: {q.get(f'option_{opt}', '')}" for opt in 'ABCD']) + "\n\nWhich option(s) best explain the cause of the event? Answer with the letter(s) only (e.g., A or A,B)."
            
            enc = self.tokenizer(prompt, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
            batch['input_ids'].append(enc['input_ids'].squeeze(0))
            batch['attention_mask'].append(enc['attention_mask'].squeeze(0))
            
        batch['input_ids'] = torch.stack(batch['input_ids'])
        batch['attention_mask'] = torch.stack(batch['attention_mask'])
        
        return batch

def main():
    print("\n[1/5] Loading Data & Model...")
    test_q = load_questions('test_data')
    test_docs = load_docs('test_data')
    retriever._load()
    
    # Load Base Model
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    
    # Load LoRA
    if (config.MODEL_DIR/'lora_weights').exists():
        print("  Loading LoRA adapters...")
        base_model = PeftModel.from_pretrained(base_model, config.MODEL_DIR/'lora_weights')
    else:
        print("⚠️ LoRA weights not found! TTA might fail if not zero-shot.")

    # Load Classifier
    model = CausalRAG7B(base_model, base_model.config.hidden_size)
    if (config.MODEL_DIR/'best_model.pt').exists():
        print("  Loading Classifier head...")
        ckpt = torch.load(config.MODEL_DIR/'best_model.pt', map_location='cpu', weights_only=False)
        model.classifier.load_state_dict(ckpt['classifier'])
        threshold = ckpt.get('threshold', 0.5)
    else:
        print("⚠️ Classifier weights not found!")
        threshold = 0.5

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\n[2/5] Running TTA Inference (N={config.TTA_N})...")
    ds = TTADataset(test_q, test_docs, tokenizer, config.MAX_LENGTH)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    model.eval()
    all_final_probs = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="TTA Inference"):
            # Batch shape: [1, TTA_N, SeqLen] -> [TTA_N, SeqLen]
            input_ids = batch['input_ids'].squeeze(0).to(config.DEVICE)
            attention_mask = batch['attention_mask'].squeeze(0).to(config.DEVICE)
            
            logits = model(input_ids, attention_mask) # [TTA_N, 4]
            probs = torch.sigmoid(logits).cpu().numpy() # [TTA_N, 4]
            
            # Average probabilities across TTA variants
            avg_probs = np.mean(probs, axis=0) # [4]
            
            all_final_probs.append(avg_probs)
            all_ids.append(batch['id'][0])
            
    print("\n[3/5] Saving Predictions...")
    results = []
    opts = list('ABCD')
    
    for qid, p in zip(all_ids, all_final_probs):
        # Apply threshold
        ans = [opts[i] for i in range(4) if p[i] >= threshold]
        # Ensure at least one answer
        if not ans:
            ans = [opts[np.argmax(p)]]
        
        results.append({'id': qid, 'answer': ','.join(sorted(ans))})
    
    with open(config.OUTPUT_DIR/'submission.jsonl', 'w') as f:
        for r in results: f.write(json.dumps(r)+'\n')
        
    shutil.make_archive(str(config.OUTPUT_DIR/'exp23_tta_submission'), 'zip', config.OUTPUT_DIR)
    
    # Save probs for ensemble
    np.save(config.OUTPUT_DIR/'test_probs.npy', np.array(all_final_probs))
    np.save(config.OUTPUT_DIR/'test_ids.npy', np.array(all_ids))
    
    print(f"✅ TTA Complete! Saved to {config.OUTPUT_DIR/'exp23_tta_submission.zip'}")

if __name__ == '__main__':
    main()
