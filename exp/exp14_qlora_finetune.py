# -*- coding: utf-8 -*-
"""
================================================================================
Exp14: QLoRA Fine-tuning for Abductive Reasoning (SOTA Training-Based)
================================================================================
SELF-CONTAINED - Copy to Kaggle notebook

🔥 KEY INNOVATION: Fine-tune Qwen-7B with QLoRA on our specific task
- QLoRA (Quantized Low-Rank Adaptation) for efficient training
- Instruction-tuned prompts for causal reasoning
- Full training on train_data, evaluate on dev, predict on test

Paper: "QLoRA: Efficient Finetuning of Quantized LLMs" (NeurIPS 2023)

GPU: H100 80GB or 2x A100
Time: ~3-4 hours for full training
Expected Score: 0.65+ (trained model > prompting)
================================================================================
"""

import json, random, shutil, warnings, re
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn

class Config:
    IS_KAGGLE = Path('/kaggle/input').exists()
    if IS_KAGGLE:
        DATA_DIR = Path('/kaggle/input/semeval2026-task12-dataset/semeval2026-task12-dataset')
        OUTPUT_DIR = Path('/kaggle/working/exp14_output')
    else:
        DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
        OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\exp14_output')
    
    # Model
    MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
    
    # QLoRA Config
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Training
    EPOCHS = 3
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION = 8  # Effective batch = 16
    LEARNING_RATE = 2e-4
    MAX_LENGTH = 512
    WARMUP_RATIO = 0.1
    
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{'='*70}")
print(f"Exp14: QLoRA Fine-tuning for Abductive Reasoning")
print(f"{'='*70}")
print(f"Model: {config.MODEL_NAME}")
print(f"LoRA: r={config.LORA_R}, alpha={config.LORA_ALPHA}")
print(f"Training: {config.EPOCHS} epochs, batch={config.BATCH_SIZE}x{config.GRADIENT_ACCUMULATION}")
print(f"{'='*70}")

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
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

def get_context(topic_info, max_chars=400):
    if not topic_info: return ""
    parts = [topic_info.get('topic', '')]
    for doc in topic_info.get('docs', [])[:2]:
        if s := doc.get('snippet', ''): parts.append(s[:150])
    return ' '.join(parts)[:max_chars]

# ============================================================================
# TRAINING PROMPT TEMPLATE
# ============================================================================
TRAIN_PROMPT = """You are an expert causal reasoner. Identify the DIRECT cause(s) of the event.

Context: {context}
Event: {event}

Options:
A: {option_A}
B: {option_B}
C: {option_C}
D: {option_D}

Think step by step:
1. Which options could have directly caused the event?
2. Eliminate options that are effects or correlations.
3. Select only direct causes.

Answer (one or more letters like A, B, A,C, etc.): {answer}"""

INFERENCE_PROMPT = """You are an expert causal reasoner. Identify the DIRECT cause(s) of the event.

Context: {context}
Event: {event}

Options:
A: {option_A}
B: {option_B}
C: {option_C}
D: {option_D}

Think step by step:
1. Which options could have directly caused the event?
2. Eliminate options that are effects or correlations.
3. Select only direct causes.

Answer (one or more letters like A, B, A,C, etc.):"""

def prepare_training_data(questions, docs_dict):
    """Prepare training examples."""
    data = []
    for q in questions:
        topic_info = docs_dict.get(q.get('topic_id'), {})
        context = get_context(topic_info)
        event = q.get('target_event', '')
        options = {opt: q.get(f'option_{opt}', '') for opt in ['A','B','C','D']}
        answer = q.get('golden_answer', '')
        
        prompt = TRAIN_PROMPT.format(
            context=context, event=event,
            option_A=options['A'], option_B=options['B'],
            option_C=options['C'], option_D=options['D'],
            answer=answer
        )
        data.append({'text': prompt, 'id': q['id'], 'answer': answer})
    return data

# ============================================================================
# QLORA FINE-TUNING
# ============================================================================
def setup_qlora():
    """Setup QLoRA fine-tuning."""
    # Install dependencies
    import subprocess
    subprocess.run(['pip', 'install', 'peft', 'trl', 'bitsandbytes', '-q'], check=True)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    print("\n[1/4] Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model(model, tokenizer, train_data):
    """Train with SFTTrainer."""
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    
    print("\n[2/4] Preparing dataset...")
    dataset = Dataset.from_list([{'text': d['text']} for d in train_data])
    
    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR / 'checkpoints'),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
        optim="paged_adamw_8bit",
    )
    
    print("\n[3/4] Training with QLoRA...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=config.MAX_LENGTH,
    )
    
    trainer.train()
    
    # Save LoRA adapters
    model.save_pretrained(config.OUTPUT_DIR / 'lora_adapters')
    tokenizer.save_pretrained(config.OUTPUT_DIR / 'lora_adapters')
    
    return model, tokenizer

@torch.no_grad()
def predict(model, tokenizer, questions, docs_dict, desc="Predicting"):
    """Generate predictions."""
    model.eval()
    results = []
    
    for q in tqdm(questions, desc=desc):
        topic_info = docs_dict.get(q.get('topic_id'), {})
        context = get_context(topic_info)
        event = q.get('target_event', '')
        options = {opt: q.get(f'option_{opt}', '') for opt in ['A','B','C','D']}
        
        prompt = INFERENCE_PROMPT.format(
            context=context, event=event,
            option_A=options['A'], option_B=options['B'],
            option_C=options['C'], option_D=options['D']
        )
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=config.MAX_LENGTH, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse answer
        answer = parse_answer(response)
        results.append({'id': q['id'], 'answer': answer})
    
    return results

def parse_answer(response):
    """Extract answer letters from response."""
    response = response.strip().upper()
    # Find all valid letters
    valid = []
    for char in response:
        if char in 'ABCD' and char not in valid:
            valid.append(char)
        elif char not in 'ABCD, ':
            break  # Stop at first non-answer character
    return ','.join(sorted(valid)) if valid else 'A'

def optimize_threshold_for_probs(probs, golds):
    """Optimize threshold if using probability output."""
    opts = ['A','B','C','D']
    best_th, best_sc = 0.5, 0
    for th in np.arange(0.20, 0.80, 0.05):
        ps = [','.join(sorted([opts[i] for i in range(4) if p[i]>=th] or [opts[np.argmax(p)]])) for p in probs]
        sc = compute_aer_score(ps, golds)
        if sc > best_sc: best_sc, best_th = sc, th
    return best_th, best_sc

def create_submission(preds, out_dir, name):
    sub_dir = out_dir/'submission'; sub_dir.mkdir(exist_ok=True)
    with open(sub_dir/'submission.jsonl','w',encoding='utf-8') as f:
        for p in preds: f.write(json.dumps(p)+'\n')
    shutil.make_archive(str(out_dir/name), 'zip', sub_dir)
    return out_dir/f'{name}.zip'

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n[0/6] Loading data...")
    train_q = load_questions('train_data')
    dev_q = load_questions('dev_data')
    test_q = load_questions('test_data')
    train_docs = load_docs('train_data')
    dev_docs = load_docs('dev_data')
    test_docs = load_docs('test_data')
    print(f"  Train: {len(train_q)}, Dev: {len(dev_q)}, Test: {len(test_q)}")
    
    # Prepare training data
    train_data = prepare_training_data(train_q, train_docs)
    print(f"  Training examples: {len(train_data)}")
    
    # Setup and train
    model, tokenizer = setup_qlora()
    model, tokenizer = train_model(model, tokenizer, train_data)
    
    # Evaluate on dev
    print("\n[4/6] Evaluating on DEV...")
    dev_preds = predict(model, tokenizer, dev_q, dev_docs, "DEV")
    dev_gold = [q.get('golden_answer', '') for q in dev_q]
    dev_answers = [p['answer'] for p in dev_preds]
    dev_score = compute_aer_score(dev_answers, dev_gold)
    print(f"  Dev Score: {dev_score:.4f}")
    
    # Predict on test
    print("\n[5/6] Predicting on TEST...")
    test_preds = predict(model, tokenizer, test_q, test_docs, "TEST")
    
    # Generate submission
    print("\n[6/6] Generating submission...")
    zip_path = create_submission(test_preds, config.OUTPUT_DIR, 'exp14_submission')
    
    # Save results
    with open(config.OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump({
            'dev_score': float(dev_score),
            'num_train': len(train_q),
            'num_test': len(test_q),
            'method': 'QLoRA Fine-tuning',
            'lora_r': config.LORA_R,
            'epochs': config.EPOCHS
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🔥 DONE!")
    print(f"  Dev Score: {dev_score:.4f}")
    print(f"  Test predictions: {len(test_q)}")
    print(f"  Submission: {zip_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
