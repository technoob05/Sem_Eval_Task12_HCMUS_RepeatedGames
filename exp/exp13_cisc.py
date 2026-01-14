# -*- coding: utf-8 -*-
"""
Exp13: CISC - Confidence-Informed Self-Consistency (arXiv 2025)
Key: Weight predictions by confidence (low entropy = high confidence = higher weight)
"""

import json, random, shutil, warnings, re, numpy as np
from pathlib import Path
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')
import torch

class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset') if IS_KAGGLE else Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
    OUTPUT_DIR = Path('/kaggle/working/exp13_output') if IS_KAGGLE else Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp13_output')
    MODEL_NAME='Qwen/Qwen2.5-7B-Instruct'; USE_MOCK=False; TEMPERATURE=0.3; NUM_RUNS=7; NUM_FEW_SHOT=5; SEED=42
config = Config(); config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Exp13: CISC ({config.NUM_RUNS} runs)")

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
set_seed(config.SEED)

def aer_score(p,g): ps,gs=set(p.split(',')),set(g.split(',')); return 1.0 if ps==gs else 0.5 if ps.issubset(gs) else 0.0
def load_q(s): 
    with open(config.DATA_DIR/s/'questions.jsonl','r',encoding='utf-8') as f: return [json.loads(l) for l in f]
def load_d(s):
    with open(config.DATA_DIR/s/'docs.json','r',encoding='utf-8') as f: return {d['topic_id']:d for d in json.load(f)}
def ctx(ti,mx=400): return ' '.join([ti.get('topic','')]+[d.get('snippet','')[:150] for d in ti.get('docs',[])[:2]])[:mx] if ti else ""

def cisc_agg(scores_list):
    """CISC: Weight by confidence (lower entropy = higher weight)"""
    weights=[]
    for s in scores_list:
        probs=np.clip(np.array([s[k] for k in 'ABCD']),0.01,0.99); probs=probs/probs.sum()
        ent=-np.sum(probs*np.log(probs)); conf=1-(ent/np.log(4))
        weights.append(conf**2)
    weights=np.array(weights)/sum(weights)
    return {o:sum(s[o]*w for s,w in zip(scores_list,weights)) for o in 'ABCD'}

PROMPT="""Find DIRECT cause(s). {fs}
Event: {e} | Context: {c}
A:{a} B:{b} C:{c_} D:{d}
JSON: {{"A":score,"B":score,"C":score,"D":score}}"""

class LLM:
    def __init__(self,n):
        try: import bitsandbytes
        except: import subprocess; subprocess.run(['pip','install','bitsandbytes','-q'],check=True)
        from transformers import AutoTokenizer,AutoModelForCausalLM
        print(f"Loading {n}...")
        self.tok=AutoTokenizer.from_pretrained(n)
        try:
            from transformers import BitsAndBytesConfig
            bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
            self.mod=AutoModelForCausalLM.from_pretrained(n,quantization_config=bnb,device_map="auto",trust_remote_code=True)
        except:
            self.mod=AutoModelForCausalLM.from_pretrained(n,torch_dtype=torch.float16,device_map="auto",trust_remote_code=True)
        self.mod.eval()
        if self.tok.pad_token is None: self.tok.pad_token=self.tok.eos_token
    @torch.no_grad()
    def gen(self,p,t=0.3):
        i=self.tok(p,return_tensors="pt",max_length=2048,truncation=True)
        i={k:v.to(self.mod.device) for k,v in i.items()}
        o=self.mod.generate(**i,max_new_tokens=80,temperature=max(t,0.01),do_sample=True,pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(o[0][i['input_ids'].shape[1]:],skip_special_tokens=True)

def parse(r):
    m=re.search(r'\{[^{}]*\}',r)
    if m:
        try: s=json.loads(m.group()); return {k:max(0,min(100,float(s.get(k,50))))/100 for k in 'ABCD'}
        except: pass
    return {k:0.5 for k in 'ABCD'}

def main():
    print("\n[1/6] Loading...")
    train_q,dev_q,test_q=load_q('train_data'),load_q('dev_data'),load_q('test_data')
    train_d,dev_d,test_d=load_d('train_data'),load_d('dev_data'),load_d('test_data')
    print(f"  Train:{len(train_q)} Dev:{len(dev_q)} Test:{len(test_q)}")
    
    print("\n[2/6] Loading LLM...")
    llm=LLM(config.MODEL_NAME)
    
    print(f"\n[3/6] Tuning on DEV ({config.NUM_RUNS} runs per question)...")
    dev_probs=[]
    for q in tqdm(dev_q,"DEV"):
        e,c_=q.get('target_event',''),ctx(dev_d.get(q.get('topic_id'),{}))
        opts={o:q.get(f'option_{o}','') for o in 'ABCD'}
        fs='\n'.join([f"Ex:{train_q[i]['target_event'][:50]}→{train_q[i]['golden_answer']}" for i in random.sample(range(len(train_q)),config.NUM_FEW_SHOT)])
        scores=[parse(llm.gen(PROMPT.format(fs=fs,e=e,c=c_,a=opts['A'],b=opts['B'],c_=opts['C'],d=opts['D']),config.TEMPERATURE)) for _ in range(config.NUM_RUNS)]
        f=cisc_agg(scores)
        dev_probs.append([f['A'],f['B'],f['C'],f['D']])
    
    dev_probs=np.array(dev_probs); gold=[q.get('golden_answer','') for q in dev_q]
    best_th,best_sc=0.5,0
    for th in np.arange(0.25,0.75,0.05):
        ps=[','.join(sorted(['ABCD'[i] for i in range(4) if p[i]>=th] or ['ABCD'[np.argmax(p)]])) for p in dev_probs]
        sc=sum(aer_score(p,g) for p,g in zip(ps,gold))/len(gold)
        if sc>best_sc: best_sc,best_th=sc,th
    print(f"  Dev Score: {best_sc:.4f} (th={best_th:.2f})")
    
    print(f"\n[4/6] Predicting on TEST ({config.NUM_RUNS} runs per question)...")
    test_probs,test_ids=[],[]
    for q in tqdm(test_q,"TEST"):
        e,c_=q.get('target_event',''),ctx(test_d.get(q.get('topic_id'),{}))
        opts={o:q.get(f'option_{o}','') for o in 'ABCD'}
        fs='\n'.join([f"Ex:{train_q[i]['target_event'][:50]}→{train_q[i]['golden_answer']}" for i in random.sample(range(len(train_q)),config.NUM_FEW_SHOT)])
        scores=[parse(llm.gen(PROMPT.format(fs=fs,e=e,c=c_,a=opts['A'],b=opts['B'],c_=opts['C'],d=opts['D']),config.TEMPERATURE)) for _ in range(config.NUM_RUNS)]
        f=cisc_agg(scores)
        test_probs.append([f['A'],f['B'],f['C'],f['D']])
        test_ids.append(q['id'])
    
    print("\n[5/6] Generating submission...")
    test_probs=np.array(test_probs)
    preds=[{'id':i,'answer':','.join(sorted(['ABCD'[j] for j in range(4) if p[j]>=best_th] or ['ABCD'[np.argmax(p)]]))} for i,p in zip(test_ids,test_probs)]
    sub=config.OUTPUT_DIR/'submission'; sub.mkdir(exist_ok=True)
    with open(sub/'submission.jsonl','w') as f:
        for p in preds: f.write(json.dumps(p)+'\n')
    shutil.make_archive(str(config.OUTPUT_DIR/'exp13_submission'),'zip',sub)
    print(f"  Dev:{best_sc:.4f} | Test:{len(test_q)} | {config.OUTPUT_DIR/'exp13_submission.zip'}")

if __name__=='__main__': main()

