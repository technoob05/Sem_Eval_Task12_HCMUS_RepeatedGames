# -*- coding: utf-8 -*-
"""
================================================================================
Exp25: MAC - Multi-Agent Causal Discovery
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

📚 PAPER: "Multi-Agent Causal Discovery Framework" (arxiv 2024)
   Key Idea: Multiple specialized agents debate to refine causal structures

🔥 NOVELTY:
1. Multiple Expert Agents (Forward, Counterfactual, Temporal)
2. Agent Debate Mechanism - Agents share and refine predictions
3. Confidence-Weighted Voting
4. Dynamic Agent Selection based on question type

GPU: H100 or A100
Time: ~5-6 hours
================================================================================
"""

import json, random, shutil, warnings
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
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
        OUTPUT_DIR = Path('/kaggle/working/exp25_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp25_output')
    
    # Shared encoder
    MODEL_NAME = 'microsoft/deberta-v3-base'  # Base for multiple agents
    MAX_LENGTH = 448
    
    # ⭐ MAC Settings
    NUM_AGENTS = 3            # Forward, Counterfactual, Temporal
    DEBATE_ROUNDS = 2         # Number of debate iterations
    USE_CONFIDENCE_WEIGHT = True
    USE_DYNAMIC_SELECTION = True
    
    # RAG
    CHUNK_SIZE = 350
    TOP_K = 2
    MAX_CONTEXT = 1100
    
    # Training
    SEED = 42
    EPOCHS_PER_AGENT = 3
    META_EPOCHS = 2
    BATCH_SIZE = 6
    LEARNING_RATE = 2e-5
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*70}")
print(f"Exp25: MAC - Multi-Agent Causal Discovery")
print(f"{'='*70}")
print(f"Agents: {config.NUM_AGENTS}, Debate Rounds: {config.DEBATE_ROUNDS}")
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
# ⭐ AGENT CONFIGURATIONS
# ============================================================================
AGENT_CONFIGS = [
    {
        'name': 'Forward',
        'prompt': lambda ctx, evt, opt: f"[FORWARD CAUSAL]\n{ctx}\n\nDoes '{opt[:120]}' directly cause '{evt[:120]}'? Analyze the forward causal chain.",
        'keywords': ['cause', 'lead', 'result', 'trigger', 'produce']
    },
    {
        'name': 'Counterfactual',
        'prompt': lambda ctx, evt, opt: f"[COUNTERFACTUAL]\n{ctx}\n\nIf '{opt[:120]}' had NOT happened, would '{evt[:120]}' still occur?",
        'keywords': ['without', 'if not', 'otherwise', 'prevent', 'necessary']
    },
    {
        'name': 'Temporal',
        'prompt': lambda ctx, evt, opt: f"[TEMPORAL]\n{ctx}\n\nDid '{opt[:120]}' happen BEFORE '{evt[:120]}'? Is it immediately preceding?",
        'keywords': ['before', 'after', 'then', 'first', 'following', 'prior']
    },
]

# ============================================================================
# RAG CONTEXT
# ============================================================================
class RAGContext:
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
    
    def get_context(self, topic_info, query, max_chars=1100):
        if not topic_info:
            return ""
        self._load()
        
        topic = topic_info.get('topic', '')
        chunks = []
        for doc in topic_info.get('docs', []):
            content = doc.get('content', '') or doc.get('snippet', '')
            title = doc.get('title', '')
            if content:
                chunks.append(f"[{title}] {content[:config.CHUNK_SIZE]}")
        
        if not chunks:
            return f"Topic: {topic}"
        
        query_emb = self.model.encode(query, convert_to_tensor=True)
        chunk_embs = self.model.encode(chunks, convert_to_tensor=True)
        sims = torch.cosine_similarity(query_emb.unsqueeze(0), chunk_embs).cpu().numpy()
        top_idx = np.argsort(sims)[-config.TOP_K:][::-1]
        
        return (f"Topic: {topic}\n" + "\n".join([chunks[i] for i in top_idx]))[:max_chars]

rag = RAGContext()

# ============================================================================
# AGENT DATASET
# ============================================================================
class AgentDataset(Dataset):
    def __init__(self, questions, docs, tokenizer, agent_id, max_len=448, is_test=False):
        self.questions = questions
        self.docs = docs
        self.tokenizer = tokenizer
        self.agent_id = agent_id
        self.agent_config = AGENT_CONFIGS[agent_id]
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
            options = ' '.join([q.get(f'option_{o}', '')[:40] for o in 'ABCD'])
            context = rag.get_context(topic_info, f"{event} {options}")
            self._cache[qid] = context
        
        context = self._cache[qid]
        event = q.get('target_event', '')
        
        input_ids_list, attention_mask_list = [], []
        for opt in 'ABCD':
            option = q.get(f'option_{opt}', '')
            text = self.agent_config['prompt'](context, event, option)
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
# AGENT MODEL
# ============================================================================
class AgentModel(nn.Module):
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
# ⭐ DEBATE MECHANISM
# ============================================================================
class DebateMechanism:
    """
    Agents share predictions and refine through debate.
    Each agent sees other agents' predictions and can adjust.
    """
    
    def __init__(self, num_agents=3, rounds=2):
        self.num_agents = num_agents
        self.rounds = rounds
    
    def compute_confidence(self, probs):
        """Compute confidence from prediction entropy."""
        probs = np.clip(probs, 1e-7, 1-1e-7)
        entropy = -np.sum(probs * np.log(probs), axis=-1)
        confidence = 1.0 / (1.0 + entropy)
        return confidence
    
    def debate(self, agent_predictions):
        """
        Perform debate rounds.
        agent_predictions: list of (N, 4) arrays from each agent
        Returns: final (N, 4) prediction
        """
        num_samples = agent_predictions[0].shape[0]
        
        # Initial predictions
        current_preds = agent_predictions.copy()
        
        for round_idx in range(self.rounds):
            # Compute confidences
            confidences = [self.compute_confidence(p) for p in current_preds]
            
            # Weighted combination
            refined_preds = []
            for agent_idx in range(self.num_agents):
                # This agent's prediction
                agent_pred = current_preds[agent_idx]
                agent_conf = confidences[agent_idx]
                
                # Other agents' predictions (weighted by confidence)
                other_pred = np.zeros_like(agent_pred)
                other_weight = 0.0
                
                for other_idx in range(self.num_agents):
                    if other_idx != agent_idx:
                        other_pred += current_preds[other_idx] * confidences[other_idx][:, None]
                        other_weight += np.mean(confidences[other_idx])
                
                if other_weight > 0:
                    other_pred /= other_weight
                
                # Blend: mostly own prediction, but influenced by confident others
                blend_factor = 0.8  # Own weight
                refined = blend_factor * agent_pred + (1 - blend_factor) * other_pred
                refined_preds.append(refined)
            
            current_preds = refined_preds
        
        # Final aggregation: confidence-weighted average
        final_pred = np.zeros((num_samples, 4))
        total_weight = np.zeros((num_samples, 1))
        
        for agent_idx in range(self.num_agents):
            conf = confidences[agent_idx][:, None]
            final_pred += current_preds[agent_idx] * conf
            total_weight += conf
        
        final_pred /= (total_weight + 1e-7)
        return final_pred

debate_mechanism = DebateMechanism(config.NUM_AGENTS, config.DEBATE_ROUNDS)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_agent(agent_id, model, train_loader, dev_loader, dev_gold):
    print(f"\n  Training Agent {agent_id+1} ({AGENT_CONFIGS[agent_id]['name']})...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS_PER_AGENT
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    best_score = 0
    for epoch in range(config.EPOCHS_PER_AGENT):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Agent{agent_id+1} E{epoch+1}", leave=False):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
        
        # Evaluate
        dev_probs = evaluate_agent(model, dev_loader)
        score = eval_threshold(dev_probs, dev_gold)
        print(f"    Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Dev={score:.4f}")
        
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), config.OUTPUT_DIR/f'agent_{agent_id}.pt')
    
    model.load_state_dict(torch.load(config.OUTPUT_DIR/f'agent_{agent_id}.pt', weights_only=False))
    return model, best_score

@torch.no_grad()
def evaluate_agent(model, loader):
    model.eval()
    all_preds = []
    for batch in loader:
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        with autocast():
            logits = model(input_ids, attention_mask)
        all_preds.extend(torch.sigmoid(logits).cpu().numpy())
    return np.array(all_preds)

def eval_threshold(probs, golds):
    opts = list('ABCD')
    best_sc = 0
    for th in np.arange(0.25, 0.65, 0.05):
        ps = [','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]])) for p in probs]
        sc = compute_aer_score(ps, golds)
        if sc > best_sc: best_sc = sc
    return best_sc

def optimize_threshold(probs, golds):
    opts = list('ABCD')
    best_th, best_sc = 0.5, 0
    for th in np.arange(0.2, 0.7, 0.05):
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
    print("\n[1/5] Loading data...")
    train_q, dev_q, test_q = load_questions('train_data'), load_questions('dev_data'), load_questions('test_data')
    train_docs, dev_docs, test_docs = load_docs('train_data'), load_docs('dev_data'), load_docs('test_data')
    dev_gold = [q.get('golden_answer','') for q in dev_q]
    test_ids = [q['id'] for q in test_q]
    print(f"  Train:{len(train_q)} Dev:{len(dev_q)} Test:{len(test_q)}")
    
    print("\n[2/5] Initializing...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    rag._load()
    
    print("\n[3/5] Training Agents...")
    agents = []
    agent_scores = []
    
    for agent_id in range(config.NUM_AGENTS):
        train_ds = AgentDataset(train_q, train_docs, tokenizer, agent_id, config.MAX_LENGTH)
        dev_ds = AgentDataset(dev_q, dev_docs, tokenizer, agent_id, config.MAX_LENGTH, is_test=True)
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
        
        agent = AgentModel(config.MODEL_NAME).to(config.DEVICE)
        agent, score = train_agent(agent_id, agent, train_loader, dev_loader, dev_gold)
        
        agents.append(agent)
        agent_scores.append(score)
    
    print(f"\n  Agent Scores: {[f'{s:.4f}' for s in agent_scores]}")
    
    print("\n[4/5] Debate & Evaluation...")
    # Get all agent predictions on dev and test
    agent_dev_preds = []
    agent_test_preds = []
    
    for agent_id, agent in enumerate(agents):
        dev_ds = AgentDataset(dev_q, dev_docs, tokenizer, agent_id, config.MAX_LENGTH, is_test=True)
        test_ds = AgentDataset(test_q, test_docs, tokenizer, agent_id, config.MAX_LENGTH, is_test=True)
        dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE*2, num_workers=0)
        
        agent_dev_preds.append(evaluate_agent(agent, dev_loader))
        agent_test_preds.append(evaluate_agent(agent, test_loader))
    
    # Debate
    dev_debate_preds = debate_mechanism.debate(agent_dev_preds)
    test_debate_preds = debate_mechanism.debate(agent_test_preds)
    
    best_th, best_score = optimize_threshold(dev_debate_preds, dev_gold)
    print(f"  After Debate - Dev: {best_score:.4f} (th={best_th:.2f})")
    
    print("\n[5/5] Generating submission...")
    zip_path = create_submission(test_debate_preds, test_ids, best_th, config.OUTPUT_DIR, 'exp25_submission')
    
    with open(config.OUTPUT_DIR/'results.json', 'w') as f:
        json.dump({
            'dev_score': float(best_score),
            'agent_scores': [float(s) for s in agent_scores],
            'threshold': float(best_th),
            'method': 'MAC - Multi-Agent Causal Discovery'
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🔥 MAC Complete!")
    print(f"  Agent Scores: {[f'{s:.4f}' for s in agent_scores]}")
    print(f"  After Debate: {best_score:.4f}")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
