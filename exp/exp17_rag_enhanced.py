# -*- coding: utf-8 -*-
"""
================================================================================
SemEval-2026 Task 12: Abductive Event Reasoning
Exp17: RAG-Enhanced Causal Reasoning with Full Document Content
================================================================================
SELF-CONTAINED SCRIPT - Copy this entire file into a single Kaggle notebook cell

🔥 IMPROVEMENTS OVER EXP09:
1. Uses FULL document content (not just snippet) - RAG-style retrieval
2. Uses document titles for better context understanding  
3. Chunks long documents and retrieves most relevant passages
4. Combines evidence from multiple documents
5. All exp09 features: Dynamic few-shot + Multi-strategy + Weighted SC

GPU: H100 80GB recommended
Time: ~5-6 hours
Expected Score: 0.72+
================================================================================
"""

#%% IMPORTS
import json, random, shutil, warnings, re
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import Counter
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

#%% CONFIG
class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp17_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp17_output')
    
    # LLM Settings
    MODEL_NAME = 'Qwen/Qwen2.5-32B-Instruct'
    USE_MOCK = False
    TEMPERATURE = 0.2
    
    # ⭐ NEW: RAG Settings
    USE_FULL_CONTENT = True      # Use full document content
    USE_TITLES = True            # Include document titles
    CHUNK_SIZE = 500             # Characters per chunk
    CHUNK_OVERLAP = 100          # Overlap between chunks
    TOP_K_CHUNKS = 3             # Number of relevant chunks to retrieve
    MAX_CONTEXT_LENGTH = 1500    # Max context chars for prompt
    
    # Existing exp09 features
    USE_DYNAMIC_FEW_SHOT = True
    NUM_FEW_SHOT = 5
    USE_MULTI_STRATEGY = True
    USE_WEIGHTED_SC = True
    NUM_RUNS = 5
    
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*70}")
print(f"Exp17: RAG-Enhanced Causal Reasoning")
print(f"{'='*70}")
print(f"Model: {config.MODEL_NAME}")
print(f"Full Content: {config.USE_FULL_CONTENT}")
print(f"Use Titles: {config.USE_TITLES}")
print(f"Chunk Size: {config.CHUNK_SIZE}, Top-K: {config.TOP_K_CHUNKS}")
print(f"{'='*70}")

#%% UTILITIES
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
set_seed(config.SEED)

def compute_aer_score(preds, golds):
    scores = []
    for p, g in zip(preds, golds):
        ps = set(x.strip() for x in p.split(',')) if p else set()
        gs = set(x.strip() for x in g.split(',')) if g else set()
        scores.append(1.0 if ps==gs else 0.5 if ps and ps.issubset(gs) else 0.0)
    return sum(scores)/len(scores) if scores else 0.0

def load_questions(split):
    with open(config.DATA_DIR/split/'questions.jsonl', 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def load_docs(split):
    with open(config.DATA_DIR/split/'docs.json', 'r', encoding='utf-8') as f:
        return {d['topic_id']: d for d in json.load(f)}

def optimize_threshold(probs, golds):
    opts = ['A','B','C','D']
    best_th, best_sc = 0.5, 0
    for th in np.arange(0.20, 0.80, 0.05):
        ps = [','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]])) for p in probs]
        sc = compute_aer_score(ps, golds)
        if sc > best_sc: best_sc, best_th = sc, th
    return best_th, best_sc

def create_predictions(probs, ids, th):
    opts = ['A','B','C','D']
    return [{'id': qid, 'answer': ','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]]))} 
            for qid, p in zip(ids, probs)]

def create_submission(preds, out_dir, name):
    sub_dir = out_dir/'submission'; sub_dir.mkdir(exist_ok=True)
    with open(sub_dir/'submission.jsonl','w',encoding='utf-8') as f:
        for p in preds: f.write(json.dumps(p)+'\n')
    shutil.make_archive(str(out_dir/name), 'zip', sub_dir)
    return out_dir/f'{name}.zip'

#%% ⭐ NEW: RAG-STYLE CONTEXT RETRIEVAL
class RAGContextRetriever:
    """Retrieve relevant chunks from full document content."""
    
    def __init__(self):
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                import subprocess
                subprocess.run(['pip', 'install', 'sentence-transformers', '-q'])
                from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def chunk_document(self, text, chunk_size=500, overlap=100):
        """Split document into overlapping chunks."""
        if not text or len(text) < chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            chunks.append(chunk.strip())
            start = end - overlap
        return chunks
    
    def get_relevant_context(self, topic_info, query, max_length=1500, top_k=3):
        """Retrieve most relevant chunks for the query."""
        if not topic_info:
            return ""
        
        self._load_model()
        
        # Collect all document content with titles
        all_chunks = []
        chunk_metadata = []
        
        topic_name = topic_info.get('topic', '')
        
        for doc in topic_info.get('docs', []):
            title = doc.get('title', '')
            content = doc.get('content', '')
            
            if config.USE_FULL_CONTENT and content:
                # Chunk the full content
                chunks = self.chunk_document(content, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                for chunk in chunks:
                    if config.USE_TITLES and title:
                        all_chunks.append(f"[{title}] {chunk}")
                    else:
                        all_chunks.append(chunk)
                    chunk_metadata.append({'title': title, 'doc_id': doc.get('id', '')})
            elif snippet := doc.get('snippet', ''):
                # Fallback to snippet
                if config.USE_TITLES and title:
                    all_chunks.append(f"[{title}] {snippet}")
                else:
                    all_chunks.append(snippet)
                chunk_metadata.append({'title': title, 'doc_id': doc.get('id', '')})
        
        if not all_chunks:
            return topic_name
        
        # Encode and find most relevant
        query_emb = self.model.encode(query, convert_to_tensor=True)
        chunk_embs = self.model.encode(all_chunks, convert_to_tensor=True)
        
        similarities = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Combine relevant chunks
        relevant_chunks = [all_chunks[i] for i in top_indices]
        context = f"Topic: {topic_name}\n\nEvidence:\n" + "\n---\n".join(relevant_chunks)
        
        return context[:max_length]

#%% DYNAMIC FEW-SHOT SELECTOR (from exp09)
class DynamicFewShotSelector:
    """Select most similar examples using sentence embeddings."""
    
    def __init__(self, train_questions, train_docs):
        self.questions = train_questions
        self.docs = train_docs
        self.embeddings = None
        self.model = None
    
    def build_index(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            import subprocess
            subprocess.run(['pip', 'install', 'sentence-transformers', '-q'])
            from sentence_transformers import SentenceTransformer
        
        print("Building few-shot index...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        texts = []
        for q in self.questions:
            event = q.get('target_event', '')
            topic_info = self.docs.get(q.get('topic_id'), {})
            topic = topic_info.get('topic', '')
            texts.append(f"{topic} {event}")
        
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
        print(f"  Indexed {len(texts)} training examples")
    
    def get_similar_examples(self, event, context, num=5):
        if self.model is None:
            self.build_index()
        
        query = f"{context[:200]} {event}"
        query_emb = self.model.encode(query, convert_to_tensor=True)
        
        sims = torch.cosine_similarity(query_emb.unsqueeze(0), self.embeddings).cpu().numpy()
        top_idxs = np.argsort(sims)[-num:][::-1]
        
        examples = []
        for idx in top_idxs:
            q = self.questions[idx]
            topic_info = self.docs.get(q.get('topic_id'), {})
            examples.append({
                'event': q.get('target_event', ''),
                'context': topic_info.get('topic', '')[:150],
                'options': {opt: q.get(f'option_{opt}', '') for opt in ['A','B','C','D']},
                'answer': q.get('golden_answer', ''),
                'similarity': float(sims[idx])
            })
        return examples

def format_examples(examples, max_per_example=150):
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Event: {ex['event'][:max_per_example]}")
        for opt in ['A','B','C','D']:
            lines.append(f"  {opt}: {ex['options'][opt][:60]}")
        lines.append(f"Direct Cause(s): {ex['answer']}")
        lines.append("")
    return "\n".join(lines)

#%% MULTI-STRATEGY PROMPTS (Enhanced with RAG context)
STRATEGY_1_CAUSAL_CHAIN = """You are an expert in causal reasoning. Analyze cause-effect relationships.

{few_shot_examples}

Now analyze:
Event: {event}

{context}

Options:
A: {option_A}
B: {option_B}
C: {option_C}
D: {option_D}

Using the evidence above, trace the causal chain: [Option] → leads to → [Event]?
Rate 0-100 how likely each is a DIRECT cause (immediate, not distant).

JSON only: {{"A": score, "B": score, "C": score, "D": score}}"""

STRATEGY_2_ELIMINATION = """Identify the DIRECT cause(s) by elimination.

{few_shot_examples}

Event: {event}

{context}

Step 1: Check each option - is it an EFFECT of the event? (eliminate if yes)
Step 2: Is it just CORRELATED? (eliminate if yes)  
Step 3: Is it a DIRECT CAUSE? (keep and rate high)

A: {option_A}
B: {option_B}
C: {option_C}
D: {option_D}

Rate remaining options 0-100.
JSON: {{"A": score, "B": score, "C": score, "D": score}}"""

STRATEGY_3_COUNTERFACTUAL = """Use counterfactual reasoning to find causes.

{few_shot_examples}

Event: {event}

{context}

For each option, ask: "If [Option] had NOT happened, would [Event] still occur?"
- If NO → Option is likely a direct cause (rate high)
- If YES → Option is not the cause (rate low)

Options:
A: {option_A}
B: {option_B}
C: {option_C}
D: {option_D}

Scores (0-100): {{"A": score, "B": score, "C": score, "D": score}}"""

STRATEGIES = [STRATEGY_1_CAUSAL_CHAIN, STRATEGY_2_ELIMINATION, STRATEGY_3_COUNTERFACTUAL]

#%% LLM CLIENT
class MockLLM:
    def __init__(self): self.calls = 0
    def generate(self, prompt, **kwargs):
        self.calls += 1
        np.random.seed((hash(prompt[:100]) + self.calls) % (2**31))
        scores = {opt: int(np.random.randint(30, 80)) for opt in ['A','B','C','D']}
        scores[np.random.choice(['A','B','C','D'])] = int(np.random.randint(70, 95))
        return json.dumps(scores)

class RealLLM:
    def __init__(self, model_name):
        try:
            import bitsandbytes
        except ImportError:
            import subprocess
            subprocess.run(['pip', 'install', 'bitsandbytes', '-q'], check=True)
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"Loading {model_name}...")
        
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config,
                device_map="auto", trust_remote_code=True
            )
            print("  Loaded with 4-bit quantization")
        except Exception as e:
            print(f"  4-bit failed, trying fp16...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                device_map="auto", trust_remote_code=True
            )
        
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @torch.no_grad()
    def generate(self, prompt, temperature=0.2, max_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=3000, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=max(temperature, 0.01),
            do_sample=True, top_p=0.9, pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def parse_scores(response):
    match = re.search(r'\{[^{}]*\}', response)
    if match:
        try:
            scores = json.loads(match.group())
            return {k: max(0, min(100, float(scores.get(k, scores.get(k.lower(), 50)))))/100 
                    for k in ['A','B','C','D']}
        except: pass
    return {k: 0.5 for k in ['A','B','C','D']}

#%% CONFIDENCE-WEIGHTED SELF-CONSISTENCY
def compute_confidence(scores):
    probs = np.array([scores[k] for k in ['A','B','C','D']])
    probs = np.clip(probs, 0.01, 0.99)
    entropy = -np.sum(probs * np.log(probs))
    confidence = 1 / (1 + entropy)
    return confidence

def weighted_self_consistency(all_scores_list):
    if len(all_scores_list) == 1:
        return all_scores_list[0]
    
    weights = [compute_confidence(scores) for scores in all_scores_list]
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    final = {opt: 0.0 for opt in ['A','B','C','D']}
    for scores, w in zip(all_scores_list, weights):
        for opt in ['A','B','C','D']:
            final[opt] += scores[opt] * w
    
    return final

#%% MULTI-STRATEGY ENSEMBLE
def multi_strategy_predict(llm, event, context, options, few_shot_str, num_runs=3):
    all_results = []
    
    for strategy_template in STRATEGIES:
        prompt = strategy_template.format(
            few_shot_examples=few_shot_str,
            event=event, context=context,
            option_A=options['A'], option_B=options['B'],
            option_C=options['C'], option_D=options['D']
        )
        
        for _ in range(max(1, num_runs // len(STRATEGIES))):
            response = llm.generate(prompt, temperature=config.TEMPERATURE)
            scores = parse_scores(response)
            all_results.append(scores)
    
    return weighted_self_consistency(all_results)

#%% MAIN PREDICTION
def predict_question(llm, q, docs_dict, few_shot_selector, rag_retriever, use_dynamic, use_multi):
    """Predict for a single question with RAG-enhanced context."""
    event = q.get('target_event', '')
    topic_info = docs_dict.get(q.get('topic_id'), {})
    options = {opt: q.get(f'option_{opt}', '') for opt in ['A','B','C','D']}
    
    # ⭐ NEW: Build query from event + options for better retrieval
    query = f"{event} " + " ".join(options.values())
    
    # ⭐ NEW: Get RAG-enhanced context with full content
    context = rag_retriever.get_relevant_context(
        topic_info, query, 
        max_length=config.MAX_CONTEXT_LENGTH, 
        top_k=config.TOP_K_CHUNKS
    )
    
    # Get few-shot examples
    if use_dynamic and few_shot_selector:
        examples = few_shot_selector.get_similar_examples(event, context, config.NUM_FEW_SHOT)
    else:
        examples = []
    
    few_shot_str = format_examples(examples) if examples else ""
    
    # Predict
    if use_multi:
        scores = multi_strategy_predict(llm, event, context, options, few_shot_str, config.NUM_RUNS)
    else:
        all_scores = []
        prompt = STRATEGY_1_CAUSAL_CHAIN.format(
            few_shot_examples=few_shot_str, event=event, context=context,
            option_A=options['A'], option_B=options['B'],
            option_C=options['C'], option_D=options['D']
        )
        for _ in range(config.NUM_RUNS):
            response = llm.generate(prompt, temperature=config.TEMPERATURE)
            all_scores.append(parse_scores(response))
        scores = weighted_self_consistency(all_scores)
    
    return scores

#%% MAIN
def main():
    print("\n[1/7] Loading data...")
    train_q = load_questions('train_data')
    dev_q = load_questions('dev_data')
    test_q = load_questions('test_data')
    train_docs = load_docs('train_data')
    dev_docs = load_docs('dev_data')
    test_docs = load_docs('test_data')
    dev_gold = [q.get('golden_answer', '') for q in dev_q]
    print(f"  Train: {len(train_q)}, Dev: {len(dev_q)}, Test: {len(test_q)}")
    
    print("\n[2/7] Initializing RAG retriever...")
    rag_retriever = RAGContextRetriever()
    
    print("\n[3/7] Building dynamic few-shot index...")
    few_shot_selector = None
    if config.USE_DYNAMIC_FEW_SHOT:
        few_shot_selector = DynamicFewShotSelector(train_q, train_docs)
        few_shot_selector.build_index()
    
    print("\n[4/7] Initializing LLM...")
    llm = MockLLM() if config.USE_MOCK else RealLLM(config.MODEL_NAME)
    if config.USE_MOCK:
        print("  ⚠️ MOCK MODE - Set USE_MOCK=False for real predictions!")
    
    # ========== STEP 1: Tune threshold on DEV ==========
    print("\n[5/7] Tuning threshold on dev set...")
    dev_probs = []
    dev_ids = []
    
    for q in tqdm(dev_q, desc="Predicting DEV"):
        scores = predict_question(
            llm, q, dev_docs, few_shot_selector, rag_retriever,
            config.USE_DYNAMIC_FEW_SHOT, config.USE_MULTI_STRATEGY
        )
        dev_probs.append([scores['A'], scores['B'], scores['C'], scores['D']])
        dev_ids.append(q['id'])
    
    dev_probs = np.array(dev_probs)
    best_th, best_sc = optimize_threshold(dev_probs, dev_gold)
    print(f"  Dev Score: {best_sc:.4f} (threshold={best_th:.2f})")
    
    # ========== STEP 2: Predict on TEST ==========
    print("\n[6/7] Predicting on TEST set...")
    test_probs = []
    test_ids = []
    
    for q in tqdm(test_q, desc="Predicting TEST"):
        scores = predict_question(
            llm, q, test_docs, few_shot_selector, rag_retriever,
            config.USE_DYNAMIC_FEW_SHOT, config.USE_MULTI_STRATEGY
        )
        test_probs.append([scores['A'], scores['B'], scores['C'], scores['D']])
        test_ids.append(q['id'])
    
    test_probs = np.array(test_probs)
    
    # ========== STEP 3: Generate submission ==========
    print("\n[7/7] Generating submission...")
    preds = create_predictions(test_probs, test_ids, best_th)
    zip_path = create_submission(preds, config.OUTPUT_DIR, 'exp17_submission')
    print(f"  Submission: {zip_path}")
    
    # Save results
    np.save(config.OUTPUT_DIR / 'dev_predictions.npy', dev_probs)
    np.save(config.OUTPUT_DIR / 'test_predictions.npy', test_probs)
    with open(config.OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump({
            'dev_score': float(best_sc), 
            'threshold': float(best_th),
            'num_dev': len(dev_q),
            'num_test': len(test_q),
            'use_full_content': config.USE_FULL_CONTENT,
            'use_titles': config.USE_TITLES,
            'chunk_size': config.CHUNK_SIZE,
            'top_k_chunks': config.TOP_K_CHUNKS,
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🔥 DONE!")
    print(f"  Dev Score: {best_sc:.4f}")
    print(f"  Test predictions: {len(test_q)} questions")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
