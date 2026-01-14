# -*- coding: utf-8 -*-
"""
SemEval-2026 Task 12: Abductive Event Reasoning
Simple Baseline: TF-IDF + Cosine Similarity
============================================================
"""

import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import shutil
import sys

# Fix encoding
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path(r"d:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning")
DATA_DIR = BASE_DIR / "semeval2026-task12-dataset"
OUTPUT_DIR = BASE_DIR / "baseline_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Results file
RESULTS_FILE = BASE_DIR / "baseline_results.txt"

def log(msg):
    print(msg)
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# Clear results file
with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
    f.write('')

# ============================================================================
# Data Loading
# ============================================================================
def load_questions(split):
    path = DATA_DIR / split / "questions.jsonl"
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_docs(split):
    path = DATA_DIR / split / "docs.json"
    with open(path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    return {d['topic_id']: d for d in docs}

# ============================================================================
# Metric Calculation
# ============================================================================
def compute_score(predictions, gold_labels):
    """Compute competition metric"""
    scores = []
    for pred, gold in zip(predictions, gold_labels):
        pred_set = set(pred.split(',')) if pred else set()
        gold_set = set(gold.split(',')) if gold else set()
        
        if pred_set == gold_set:
            scores.append(1.0)
        elif pred_set and pred_set.issubset(gold_set):
            scores.append(0.5)
        else:
            scores.append(0.0)
    
    return np.mean(scores)

# ============================================================================
# Simple TF-IDF Baseline
# ============================================================================
class TFIDFBaseline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.options = ['A', 'B', 'C', 'D']
        
    def fit(self, questions, docs_dict):
        """Fit TF-IDF on all text"""
        all_texts = []
        
        for q in questions:
            all_texts.append(q.get('target_event', ''))
            for opt in self.options:
                all_texts.append(q.get(f'option_{opt}', ''))
            
            topic_id = q.get('topic_id')
            if topic_id in docs_dict:
                for doc in docs_dict[topic_id].get('docs', []):
                    all_texts.append(doc.get('snippet', ''))
        
        self.vectorizer.fit(all_texts)
        log(f"Fitted TF-IDF with {len(self.vectorizer.vocabulary_)} features")
    
    def predict_one(self, question, docs_dict, threshold=0.3):
        """Predict for one question"""
        topic_id = question.get('topic_id')
        event = question.get('target_event', '')
        
        context_texts = []
        if topic_id in docs_dict:
            topic_info = docs_dict[topic_id]
            context_texts.append(topic_info.get('topic', ''))
            for doc in topic_info.get('docs', [])[:5]:
                context_texts.append(doc.get('snippet', ''))
        
        context = ' '.join(context_texts)
        query = f"{event} {context}"
        
        query_vec = self.vectorizer.transform([query])
        
        scores = []
        for opt in self.options:
            option_text = question.get(f'option_{opt}', '')
            opt_vec = self.vectorizer.transform([option_text])
            sim = cosine_similarity(query_vec, opt_vec)[0][0]
            scores.append(sim)
        
        scores = np.array(scores)
        
        selected = [self.options[i] for i in range(4) if scores[i] >= threshold]
        
        if not selected:
            selected = [self.options[np.argmax(scores)]]
        
        return ','.join(sorted(selected)), scores
    
    def predict(self, questions, docs_dict, threshold=0.3):
        """Predict for all questions"""
        predictions = []
        
        for q in questions:
            pred, _ = self.predict_one(q, docs_dict, threshold)
            predictions.append({
                'id': q.get('id'),
                'answer': pred
            })
        
        return predictions

# ============================================================================
# Majority Baseline
# ============================================================================
def majority_baseline(train_questions, test_questions):
    patterns = Counter()
    for q in train_questions:
        answer = q.get('golden_answer', '')
        patterns[answer] += 1
    
    most_common = patterns.most_common(1)[0][0]
    log(f"Most common answer: '{most_common}' ({patterns[most_common]} times)")
    
    predictions = []
    for q in test_questions:
        predictions.append({'id': q.get('id'), 'answer': most_common})
    
    return predictions

# ============================================================================
# Random Baseline
# ============================================================================
def random_baseline(test_questions, seed=42):
    np.random.seed(seed)
    options = ['A', 'B', 'C', 'D']
    
    predictions = []
    for q in test_questions:
        n = np.random.choice([1, 2], p=[0.6, 0.4])
        selected = np.random.choice(options, size=n, replace=False)
        predictions.append({'id': q.get('id'), 'answer': ','.join(sorted(selected))})
    
    return predictions

# ============================================================================
# Main
# ============================================================================
def main():
    log("=" * 60)
    log("SemEval-2026 Task 12: Simple Baselines")
    log("=" * 60)
    
    log("\nLoading data...")
    train_questions = load_questions('train_data')
    dev_questions = load_questions('dev_data')
    train_docs = load_docs('train_data')
    dev_docs = load_docs('dev_data')
    
    log(f"Train: {len(train_questions)} questions")
    log(f"Dev: {len(dev_questions)} questions")
    
    dev_gold = [q.get('golden_answer', '') for q in dev_questions]
    
    # Baseline 1: Random
    log("\n" + "-" * 40)
    log("Baseline 1: Random")
    log("-" * 40)
    random_preds = random_baseline(dev_questions)
    random_answers = [p['answer'] for p in random_preds]
    random_score = compute_score(random_answers, dev_gold)
    log(f"Dev Score: {random_score:.4f}")
    
    # Baseline 2: Majority
    log("\n" + "-" * 40)
    log("Baseline 2: Majority")
    log("-" * 40)
    majority_preds = majority_baseline(train_questions, dev_questions)
    majority_answers = [p['answer'] for p in majority_preds]
    majority_score = compute_score(majority_answers, dev_gold)
    log(f"Dev Score: {majority_score:.4f}")
    
    # Baseline 3: TF-IDF
    log("\n" + "-" * 40)
    log("Baseline 3: TF-IDF Similarity")
    log("-" * 40)
    
    tfidf_model = TFIDFBaseline()
    all_docs = {**train_docs, **dev_docs}
    all_questions = train_questions + dev_questions
    
    log("Fitting TF-IDF...")
    tfidf_model.fit(all_questions, all_docs)
    
    best_threshold = 0.3
    best_score = 0
    
    log("\nTuning threshold on dev set...")
    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        preds = tfidf_model.predict(dev_questions, dev_docs, threshold)
        answers = [p['answer'] for p in preds]
        score = compute_score(answers, dev_gold)
        log(f"  Threshold {threshold:.2f}: Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    log(f"\nBest threshold: {best_threshold}, Best score: {best_score:.4f}")
    
    # Generate Final Submission
    log("\n" + "=" * 60)
    log("Generating Dev Submission")
    log("=" * 60)
    
    final_preds = tfidf_model.predict(dev_questions, dev_docs, threshold=best_threshold)
    
    submission_dir = OUTPUT_DIR / "submission"
    submission_dir.mkdir(exist_ok=True)
    
    submission_path = submission_dir / "submission.jsonl"
    with open(submission_path, 'w', encoding='utf-8') as f:
        for pred in final_preds:
            f.write(json.dumps(pred) + '\n')
    
    log(f"Saved {len(final_preds)} predictions to {submission_path}")
    
    log("\nSample predictions:")
    for pred in final_preds[:5]:
        log(f"  {pred}")
    
    # Create zip
    zip_path = OUTPUT_DIR / "submission"
    shutil.make_archive(str(zip_path), 'zip', submission_dir)
    
    final_zip = OUTPUT_DIR / "submission.zip"
    log(f"\nSubmission zip: {final_zip}")
    log(f"File size: {final_zip.stat().st_size / 1024:.1f} KB")
    
    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"{'Baseline':<20} {'Dev Score':>10}")
    log("-" * 30)
    log(f"{'Random':<20} {random_score:>10.4f}")
    log(f"{'Majority':<20} {majority_score:>10.4f}")
    log(f"{'TF-IDF (best)':<20} {best_score:>10.4f}")
    log("-" * 30)
    log(f"\nUpload {final_zip} to Codabench (Development Phase)!")

if __name__ == "__main__":
    main()
