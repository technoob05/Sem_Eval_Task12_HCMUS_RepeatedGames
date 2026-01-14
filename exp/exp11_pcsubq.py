# -*- coding: utf-8 -*-
"""
================================================================================
Exp11: PC-SubQ Causal Decomposition (arXiv 2024)
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

Key Innovation: Break causal question into structured sub-questions
- Q1: Temporal precedence (Did cause come before effect?)
- Q2: Mechanism (Is there a pathway from cause to effect?)
- Q3: Counterfactual (Would effect occur without cause?)

Paper: "PC-SubQ: Prompting Strategy for Causal Discovery" (arXiv 2024)

Expected: 0.62+ 
================================================================================
"""

import json, random, shutil, warnings, re
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')
import torch

class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp11_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp11_output')
    
    MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
    USE_MOCK = False
    TEMPERATURE = 0.2
    NUM_RUNS = 3
    NUM_FEW_SHOT = 5
    SEED = 42

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*60}\nExp11: PC-SubQ Decomposition (arXiv 2024)\n{'='*60}")

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
set_seed(config.SEED)

def compute_aer_score(preds, golds):
    return sum(1.0 if set(p.split(','))==set(g.split(',')) else 0.5 if set(p.split(',')).issubset(set(g.split(','))) else 0.0 for p,g in zip(preds,golds))/len(preds)

def load_questions(split):
    with open(config.DATA_DIR/split/'questions.jsonl','r',encoding='utf-8') as f: return [json.loads(l) for l in f]
def load_docs(split):
    with open(config.DATA_DIR/split/'docs.json','r',encoding='utf-8') as f: return {d['topic_id']:d for d in json.load(f)}
def get_context(ti, mx=400):
    if not ti: return ""
    return ' '.join([ti.get('topic','')]+[d.get('snippet','')[:150] for d in ti.get('docs',[])[:2]])[:mx]
def optimize_threshold(probs, golds):
    opts=['A','B','C','D']; best_th,best_sc=0.5,0
    for th in np.arange(0.25,0.75,0.05):
        ps=[','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]])) for p in probs]
        sc=compute_aer_score(ps,golds)
        if sc>best_sc: best_sc,best_th=sc,th
    return best_th,best_sc
def create_predictions(probs,ids,th):
    opts=['A','B','C','D']
    return [{'id':qid,'answer':','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]]))} for qid,p in zip(ids,probs)]
def create_submission(preds,out_dir,name):
    sub=out_dir/'submission';sub.mkdir(exist_ok=True)
    with open(sub/'submission.jsonl','w',encoding='utf-8') as f:
        for p in preds: f.write(json.dumps(p)+'\n')
    shutil.make_archive(str(out_dir/name),'zip',sub)
    return out_dir/f'{name}.zip'

# ⭐ PC-SUBQ PROMPT (arXiv 2024)
PCSUBQ_PROMPT = """PC-SubQ: Causal Discovery through Sub-Questions

For each option, answer three causal sub-questions:

SUB-QUESTIONS:
Q1: TEMPORAL - Did [Option] occur BEFORE [Event]? (Yes=+30pts, No=0pts)
Q2: MECHANISM - Is there a clear causal pathway [Option] → [Event]? (Yes=+40pts, No=0pts)
Q3: COUNTERFACTUAL - If [Option] had NOT happened, would [Event] still occur? (No=+30pts, Yes=0pts)

Total = Q1 + Q2 + Q3 (max 100)

{few_shot}

EVENT: {event}
CONTEXT: {context}

ANALYZE EACH OPTION:

Option A: {option_A}
- Q1 (Temporal): 
- Q2 (Mechanism): 
- Q3 (Counterfactual):
- Total A: ?

Option B: {option_B}
- Q1-Q3 analysis...
- Total B: ?

Option C: {option_C}
- Total C: ?

Option D: {option_D}
- Total D: ?

OUTPUT JSON: {{"A": total_A, "B": total_B, "C": total_C, "D": total_D}}"""

class DynamicFewShot:
    def __init__(self, train_q, train_docs):
        self.q=train_q; self.docs=train_docs; self.model=None; self.emb=None
    def build(self):
        try: from sentence_transformers import SentenceTransformer
        except: import subprocess; subprocess.run(['pip','install','sentence-transformers','-q'])
        from sentence_transformers import SentenceTransformer
        print("Building index..."); self.model=SentenceTransformer('all-MiniLM-L6-v2')
        texts=[f"{self.docs.get(q.get('topic_id'),{}).get('topic','')} {q.get('target_event','')}" for q in self.q]
        self.emb=self.model.encode(texts,show_progress_bar=True,convert_to_tensor=True)
    def get(self, query, n=5):
        if not self.model: self.build()
        qe=self.model.encode(query,convert_to_tensor=True)
        sims=torch.cosine_similarity(qe.unsqueeze(0),self.emb).cpu().numpy()
        idxs=np.argsort(sims)[-n:][::-1]
        return [{'event':self.q[i].get('target_event',''),'options':{o:self.q[i].get(f'option_{o}','') for o in 'ABCD'},'answer':self.q[i].get('golden_answer','')} for i in idxs]

def format_few_shot(exs):
    return '\n'.join([f"Ex{i+1}: Event: {e['event'][:80]} → Cause: {e['answer']}" for i,e in enumerate(exs)])

class RealLLM:
    def __init__(self, name):
        try: import bitsandbytes
        except: import subprocess; subprocess.run(['pip','install','bitsandbytes','-q'],check=True)
        from transformers import AutoTokenizer,AutoModelForCausalLM
        print(f"Loading {name}...")
        try:
            from transformers import BitsAndBytesConfig
            bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
            self.tok=AutoTokenizer.from_pretrained(name); self.mod=AutoModelForCausalLM.from_pretrained(name,quantization_config=bnb,device_map="auto",trust_remote_code=True)
        except:
            self.tok=AutoTokenizer.from_pretrained(name); self.mod=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float16,device_map="auto",trust_remote_code=True)
        self.mod.eval()
        if self.tok.pad_token is None: self.tok.pad_token=self.tok.eos_token
    @torch.no_grad()
    def generate(self,prompt,temp=0.2,mx=150):
        inp=self.tok(prompt,return_tensors="pt",max_length=2048,truncation=True)
        inp={k:v.to(self.mod.device) for k,v in inp.items()}
        out=self.mod.generate(**inp,max_new_tokens=mx,temperature=max(temp,0.01),do_sample=True,top_p=0.9,pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][inp['input_ids'].shape[1]:],skip_special_tokens=True)

def parse_scores(r):
    m=re.search(r'\{[^{}]*\}',r)
    if m:
        try: s=json.loads(m.group()); return {k:max(0,min(100,float(s.get(k,s.get(k.lower(),50)))))/100 for k in 'ABCD'}
        except: pass
    return {k:0.5 for k in 'ABCD'}

def main():
    print("\n[1/6] Loading...")
    train_q,dev_q,test_q=load_questions('train_data'),load_questions('dev_data'),load_questions('test_data')
    train_docs,dev_docs,test_docs=load_docs('train_data'),load_docs('dev_data'),load_docs('test_data')
    dev_gold=[q.get('golden_answer','') for q in dev_q]
    print(f"  Train:{len(train_q)} Dev:{len(dev_q)} Test:{len(test_q)}")
    
    print("\n[2/6] Building few-shot index..."); fs=DynamicFewShot(train_q,train_docs); fs.build()
    print("\n[3/6] Loading LLM..."); llm=RealLLM(config.MODEL_NAME)
    
    print("\n[4/6] Tuning on DEV...")
    dev_probs=[]
    for q in tqdm(dev_q,desc="DEV"):
        event,ctx=q.get('target_event',''),get_context(dev_docs.get(q.get('topic_id'),{}))
        opts={o:q.get(f'option_{o}','') for o in 'ABCD'}
        exs=fs.get(f"{ctx[:100]} {event}",config.NUM_FEW_SHOT)
        scores_list=[parse_scores(llm.generate(PCSUBQ_PROMPT.format(few_shot=format_few_shot(exs),event=event,context=ctx,option_A=opts['A'],option_B=opts['B'],option_C=opts['C'],option_D=opts['D']),config.TEMPERATURE)) for _ in range(config.NUM_RUNS)]
        dev_probs.append([np.mean([s[o] for s in scores_list]) for o in 'ABCD'])
    best_th,best_sc=optimize_threshold(np.array(dev_probs),[q.get('golden_answer','') for q in dev_q])
    print(f"  Dev Score: {best_sc:.4f} (th={best_th:.2f})")
    
    print("\n[5/6] Predicting on TEST...")
    test_probs,test_ids=[],[]
    for q in tqdm(test_q,desc="TEST"):
        event,ctx=q.get('target_event',''),get_context(test_docs.get(q.get('topic_id'),{}))
        opts={o:q.get(f'option_{o}','') for o in 'ABCD'}
        exs=fs.get(f"{ctx[:100]} {event}",config.NUM_FEW_SHOT)
        scores_list=[parse_scores(llm.generate(PCSUBQ_PROMPT.format(few_shot=format_few_shot(exs),event=event,context=ctx,option_A=opts['A'],option_B=opts['B'],option_C=opts['C'],option_D=opts['D']),config.TEMPERATURE)) for _ in range(config.NUM_RUNS)]
        test_probs.append([np.mean([s[o] for s in scores_list]) for o in 'ABCD'])
        test_ids.append(q['id'])
    
    print("\n[6/6] Generating submission...")
    preds=create_predictions(np.array(test_probs),test_ids,best_th)
    zip_path=create_submission(preds,config.OUTPUT_DIR,'exp11_submission')
    print(f"  Dev:{best_sc:.4f} | Test:{len(test_q)} | {zip_path}")

if __name__=='__main__': main()

