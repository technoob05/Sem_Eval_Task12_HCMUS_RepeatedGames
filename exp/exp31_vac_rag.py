# -*- coding: utf-8 -*-
"""
================================================================================
Exp31: VAC-RAG - Verified Abductive CausalRAG
================================================================================
🎯 NOVEL CONTRIBUTION: First to combine abductive inference + causal graphs +
   self-verification for grounded generation in reasoning tasks

📊 METHOD:
   1. CausalRAG retrieves direct causal evidence
   2. If evidence gap detected → Abductive generation of missing causal links
   3. Self-verification: Check consistency with causal graph + documents
   4. Score = evidence_support * generation_consistency * plausibility

🏆 TARGET: 0.89+ (from 0.86) by filling evidence gaps with verified hypotheses

REQUIREMENTS: Trained exp22_7b model
GPU: H100 80GB
Time: ~4-5 hours
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        MODEL_DIR = Path('/kaggle/input/basline-semeval-task12/exp22_7b_output')  # Updated path
        OUTPUT_DIR = Path('/kaggle/working/exp31_vac_rag_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        MODEL_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp22_7b_output')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp31_vac_rag_output')
    
    MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
    MAX_LENGTH = 512
    CHUNK_SIZE = 350
    TOP_K_CHUNKS = 3
    MAX_CONTEXT = 1200
    
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
    
    # VAC-RAG settings
    GAP_THRESHOLD = 2  # If < 2 causal edges found, trigger abductive generation
    VERIFICATION_THRESHOLD = 0.6  # Only use generated links if verification >= this
    
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
            if len(sent) < 10: continue
            for pattern, edge_type in self.patterns:
                match = pattern.search(sent)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        cause, effect = groups[0].strip()[:100], groups[1].strip()[:100]
                        if len(cause) > 5 and len(effect) > 5:
                            edges.append({
                                'cause': cause,
                                'effect': effect,
                                'type': edge_type,
                                'sentence': sent[:200]
                            })
        return edges[:config.MAX_GRAPH_EDGES]
    
    def build_graph(self, topic_info):
        if not topic_info: return {'nodes': set(), 'edges': [], 'sentences': []}
        all_edges = []
        all_sentences = []
        for doc in topic_info.get('docs', []):
            content = doc.get('content', '') or doc.get('snippet', '')
            edges = self.extract_causal_edges(content)
            all_edges.extend(edges)
            all_sentences.append(content)
        
        nodes = set()
        for e in all_edges:
            nodes.add(e['cause'])
            nodes.add(e['effect'])
        
        return {'nodes': nodes, 'edges': all_edges, 'sentences': all_sentences}

causal_builder = CausalGraphBuilder()

# ============================================================================
# ⭐ NOVELTY: ABDUCTIVE CAUSAL LINK GENERATOR
# ============================================================================
class AbductiveCausalGenerator:
    """Generate missing causal links using LLM when evidence is incomplete"""
    
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
    
    def generate_missing_link(self, event, options, graph, device):
        """
        Generate hypothetical causal link to explain the event
        Returns: {'cause': str, 'effect': str, 'hypothesis': str}
        """
        # Build prompt for abductive generation
        existing_edges = "; ".join([f"{e['cause']}→{e['effect']}" for e in graph['edges'][:3]])
        
        prompt = f"""Given the event: "{event}"

Existing causal knowledge: {existing_edges if existing_edges else "None"}

Options:
{chr(10).join([f'{chr(65+i)}: {opt}' for i, opt in enumerate(options)])}

Task: Generate a plausible missing causal link that would explain why the event occurred.
Format your answer as: "X caused {event} because Y"

Missing causal link:"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=400, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse generated hypothesis
        hypothesis = generated.strip()
        
        # Try to extract cause from hypothesis
        cause_match = re.search(r'(.+?)\s+caused', hypothesis, re.IGNORECASE)
        cause = cause_match.group(1).strip() if cause_match else "unknown cause"
        
        return {
            'cause': cause,
            'effect': event,
            'hypothesis': hypothesis
        }

# ============================================================================
# ⭐ NOVELTY: SELF-VERIFICATION MODULE
# ============================================================================
class CausalLinkVerifier:
    """Verify generated causal links against graph structure and documents"""
    
    def __init__(self, embed_model):
        self.embed_model = embed_model
    
    def verify_link(self, generated_link, graph, documents):
        """
        Verify if generated link is:
        1. Consistent with existing graph (doesn't contradict)
        2. Semantically plausible with documents
        3. Not a duplicate
        
        Returns: verification_score [0,1]
        """
        cause = generated_link['cause']
        effect = generated_link['effect']
        hypothesis = generated_link['hypothesis']
        
        # Score 1: Graph consistency (check for contradictions)
        graph_score = self._check_graph_consistency(cause, effect, graph)
        
        # Score 2: Document plausibility (semantic similarity)
        doc_score = self._check_document_plausibility(hypothesis, documents)
        
        # Score 3: Novelty (not duplicate)
        novelty_score = self._check_novelty(cause, effect, graph)
        
        # Combined score
        verification_score = (graph_score * 0.3 + doc_score * 0.5 + novelty_score * 0.2)
        
        return verification_score
    
    def _check_graph_consistency(self, cause, effect, graph):
        """Check if new link doesn't contradict existing edges"""
        # Simple check: if exact opposite exists, penalize
        for edge in graph['edges']:
            if edge['effect'].lower() == cause.lower() and edge['cause'].lower() == effect.lower():
                return 0.0  # Contradiction found
        
        # Check if it forms a reasonable extension
        connected = False
        for edge in graph['edges']:
            if cause.lower() in edge['cause'].lower() or cause.lower() in edge['effect'].lower():
                connected = True
                break
        
        return 1.0 if connected else 0.7
    
    def _check_document_plausibility(self, hypothesis, documents):
        """Check semantic similarity with source documents"""
        if not documents:
            return 0.5
        
        # Encode hypothesis
        hyp_emb = self.embed_model.encode(hypothesis, convert_to_tensor=True)
        
        # Encode all document sentences
        all_sentences = []
        for doc in documents:
            sentences = re.split(r'[.!?]', doc)
            all_sentences.extend([s.strip() for s in sentences if len(s.strip()) > 20])
        
        if not all_sentences:
            return 0.5
        
        doc_embs = self.embed_model.encode(all_sentences[:50], convert_to_tensor=True)  # Limit to top 50
        
        # Max similarity
        similarities = torch.cosine_similarity(hyp_emb.unsqueeze(0), doc_embs).cpu().numpy()
        max_sim = float(np.max(similarities))
        
        return max_sim
    
    def _check_novelty(self, cause, effect, graph):
        """Check if this is a new link"""
        for edge in graph['edges']:
            if cause.lower() in edge['cause'].lower() and effect.lower() in edge['effect'].lower():
                return 0.0  # Duplicate
        return 1.0  # Novel

# ============================================================================
# VAC-RAG RETRIEVER
# ============================================================================
class VACRAGRetriever:
    """Retriever with Verified Abductive Causal reasoning"""
    
    def __init__(self, generator, verifier):
        self.embed_model = None
        self.generator = generator
        self.verifier = verifier
    
    def _load(self):
        if self.embed_model is None:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.verifier.embed_model = self.embed_model
    
    def retrieve(self, topic_info, event, options, device):
        """Retrieve with abductive augmentation if evidence is sparse"""
        if not topic_info:
            return f"Event: {event}", []
        
        self._load()
        
        topic = topic_info.get('topic', '')
        query = f"{event} " + " ".join([opt[:60] for opt in options])
        graph = causal_builder.build_graph(topic_info)
        
        # ⭐ Check if evidence gap exists
        relevant_edges = []
        for edge in graph['edges']:
            if event.lower() in edge['effect'].lower() or any(opt.lower() in edge['cause'].lower() for opt in options):
                relevant_edges.append(edge)
        
        # Collect chunks
        chunks = []
        for doc in topic_info.get('docs', []):
            title = doc.get('title', '')
            content = doc.get('content', '') or doc.get('snippet', '')
            for i in range(0, len(content), config.CHUNK_SIZE - 50):
                chunk = content[i:i+config.CHUNK_SIZE]
                if len(chunk) >= 50:
                    chunks.append(f"[{title}] {chunk}")
        
        if not chunks:
            chunks = [f"Topic: {topic}"]
        
        # Semantic retrieval
        query_emb = self.embed_model.encode(query, convert_to_tensor=True)
        chunk_embs = self.embed_model.encode(chunks, convert_to_tensor=True)
        sims = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        
        # Boost causal chunks
        for i, chunk in enumerate(chunks):
            for edge in graph['edges']:
                if edge['cause'].lower() in chunk.lower() or edge['effect'].lower() in chunk.lower():
                    sims[i] *= config.CAUSAL_EDGE_BOOST
        
        top_idx = np.argsort(sims)[-config.TOP_K_CHUNKS:][::-1]
        selected = [chunks[i] for i in top_idx]
        
        augmented_edges = []
        
        # ⭐ ABDUCTIVE GENERATION if gap exists
        if len(relevant_edges) < config.GAP_THRESHOLD:
            print(f"  [Gap detected: {len(relevant_edges)} edges] Generating abductive hypothesis...")
            
            generated_link = self.generator.generate_missing_link(event, options, graph, device)
            
            # Verify generated link
            verification_score = self.verifier.verify_link(
                generated_link,
                graph,
                graph['sentences']
            )
            
            print(f"  Generated: \"{generated_link['hypothesis'][:80]}...\"")
            print(f"  Verification score: {verification_score:.3f}")
            
            if verification_score >= config.VERIFICATION_THRESHOLD:
                augmented_edges.append({
                    **generated_link,
                    'verified': True,
                    'score': verification_score
                })
                print(f"  ✓ Verified and added to context!")
            else:
                print(f"  ✗ Rejected (low verification score)")
        
        # Build context
        context = f"Topic: {topic}\n\nEvidence:\n" + "\n".join(selected)
        
        # Add verified hypotheses
        if augmented_edges:
            hypotheses_text = "\n\nGenerated Hypotheses (verified):\n" + "\n".join([
                f"- {h['hypothesis']}" for h in augmented_edges
            ])
            context += hypotheses_text
        
        return context[:config.MAX_CONTEXT], augmented_edges

# ============================================================================
# DATASET
# ============================================================================
class VACRAGDataset(Dataset):
    def __init__(self, questions, docs, tokenizer, retriever, device, max_len=512, is_test=False):
        self.questions, self.docs, self.tokenizer = questions, docs, tokenizer
        self.retriever, self.device = retriever, device
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
            context, augmented = self.retriever.retrieve(topic_info, event, options, self.device)
            self._cache[qid] = (context, len(augmented))
        
        context, num_augmented = self._cache[qid]
        
        options_text = "\n".join([f"{opt}: {q.get(f'option_{opt}', '')}" for opt in 'ABCD'])
        prompt = f"{context}\n\nEvent: {q.get('target_event','')}\n\nOptions:\n{options_text}\n\nWhich option(s) best explain the cause? Answer with letter(s) only (e.g., A or A,B)."
        
        enc = self.tokenizer(prompt, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        
        result = {
            'id': qid,
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'num_augmented': num_augmented
        }
        
        if not self.is_test:
            golden = q.get('golden_answer', '')
            labels = torch.zeros(4)
            for ans in golden.split(','):
                if ans.strip().upper() in 'ABCD':
                    labels['ABCD'.index(ans.strip().upper())] = 1.0
            result['labels'] = labels
        
        return result

# ============================================================================
# MODEL COMPONENTS (from exp22_7b)
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
    
    def forward(self, input_ids, attention_mask):
        if not self._classifier_moved:
            self.classifier = self.classifier.to(input_ids.device)
            self._classifier_moved = True
        
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        logits = self.classifier(hidden_states)
        return logits

# ============================================================================
# MAIN
# ============================================================================
def main():
    print(f"\n{'='*70}")
    print("Exp31: VAC-RAG - Verified Abductive CausalRAG")
    print(f"{'='*70}\n")
    
    print("[1/5] Loading data...")
    test_q = load_questions('test_data')
    test_docs = load_docs('test_data')
    
    print("\n[2/5] Loading Qwen-7B model...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    
    if (config.MODEL_DIR/'lora_weights').exists():
        base_model = PeftModel.from_pretrained(base_model, config.MODEL_DIR/'lora_weights')
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    print("\n[3/5] Initializing VAC-RAG components...")
    generator = AbductiveCausalGenerator(base_model, tokenizer)
    verifier = CausalLinkVerifier(None)  # Embed model loaded in retriever
    retriever = VACRAGRetriever(generator, verifier)
    retriever._load()
    
    print("\n[4/5] Creating dataset with abductive augmentation...")
    test_ds = VACRAGDataset(test_q, test_docs, tokenizer, retriever, config.DEVICE, config.MAX_LENGTH, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Load classifier (classes defined above)
    model = CausalRAG7B(base_model, base_model.config.hidden_size)
    if (config.MODEL_DIR/'best_model.pt').exists():
        ckpt = torch.load(config.MODEL_DIR/'best_model.pt', map_location='cpu', weights_only=False)
        model.classifier.load_state_dict(ckpt['classifier'])
    
    print("\n[5/5] Inference with VAC-RAG...")
    model.eval()
    all_preds, all_ids, total_augmented = [], [], 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="VAC-RAG Inference"):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_preds.append(probs[0])
            all_ids.append(batch['id'][0])
            total_augmented += batch['num_augmented'].item()
    
    print(f"\n  Total abductive hypotheses generated: {total_augmented}")
    
    # Create submission
    opts = list('ABCD')
    results = []
    for qid, p in zip(all_ids, all_preds):
        ans = [opts[i] for i in range(4) if p[i] >= 0.5]
        if not ans: ans = [opts[np.argmax(p)]]
        results.append({'id': qid, 'answer': ','.join(sorted(ans))})
    
    with open(config.OUTPUT_DIR/'submission.jsonl', 'w') as f:
        for r in results: f.write(json.dumps(r)+'\n')
    
    shutil.make_archive(str(config.OUTPUT_DIR/'exp31_vac_rag_submission'), 'zip', config.OUTPUT_DIR)
    
    print(f"\n{'='*70}")
    print(f"🏆 VAC-RAG Complete!")
    print(f"  Augmented {total_augmented}/{len(all_ids)} questions with verified hypotheses")
    print(f"  Submission: {config.OUTPUT_DIR/'exp31_vac_rag_submission.zip'}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
