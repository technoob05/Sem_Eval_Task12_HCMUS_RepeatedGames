# -*- coding: utf-8 -*-
"""
EDA for SemEval-2026 Task 12: Abductive Event Reasoning (AER)
"""

import json
from pathlib import Path
from collections import Counter
import os

# Set paths
BASE_DIR = Path(r"d:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset")
OUTPUT_FILE = Path(r"d:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\eda_results.md")

def load_questions(split):
    path = BASE_DIR / split / "questions.jsonl"
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_docs(split):
    path = BASE_DIR / split / "docs.json"
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load all data
splits = ['sample_data', 'train_data', 'dev_data', 'test_data']
all_questions = {split: load_questions(split) for split in splits}
all_docs = {split: load_docs(split) for split in splits}

# Build report
lines = []
lines.append("# SemEval-2026 Task 12: Abductive Event Reasoning - EDA Report\n")
lines.append("---\n")

# Dataset Overview
lines.append("## 1. Dataset Overview\n")
lines.append("| Split | Questions | Topics | Total Documents |")
lines.append("|-------|-----------|--------|-----------------|")
for split in splits:
    q_count = len(all_questions[split])
    doc_count = len(all_docs[split])
    total_docs = sum(len(d.get('docs', [])) for d in all_docs[split])
    lines.append(f"| {split.replace('_data', '')} | {q_count} | {doc_count} | {total_docs} |")
lines.append("")

# Sample Question
lines.append("\n## 2. Sample Question (from train_data)\n")
if all_questions['train_data']:
    sample_q = all_questions['train_data'][0]
    lines.append(f"- **ID**: `{sample_q.get('id')}`")
    lines.append(f"- **Topic ID**: `{sample_q.get('topic_id')}`")
    lines.append(f"\n**Target Event:**")
    lines.append(f"> {sample_q.get('target_event')}\n")
    lines.append(f"**Options:**")
    lines.append(f"- A: {sample_q.get('option_A')}")
    lines.append(f"- B: {sample_q.get('option_B')}")
    lines.append(f"- C: {sample_q.get('option_C')}")
    lines.append(f"- D: {sample_q.get('option_D')}")
    lines.append(f"\n**Golden Answer:** `{sample_q.get('golden_answer', 'N/A')}`")

# Answer Distribution
lines.append("\n\n## 3. Answer Distribution Analysis\n")
train_dev_questions = all_questions['train_data'] + all_questions['dev_data'] + all_questions['sample_data']

single_answers = []
multi_answers = []
answer_counts = Counter()

for q in train_dev_questions:
    answer = q.get('golden_answer', '')
    if answer:
        answers = answer.split(',')
        if len(answers) == 1:
            single_answers.append(answers[0])
        else:
            multi_answers.append(tuple(sorted(answers)))
        for a in answers:
            answer_counts[a.strip()] += 1

total_with_answers = len(single_answers) + len(multi_answers)
lines.append(f"**Total questions with answers:** {total_with_answers}")
lines.append(f"- Single answer: {len(single_answers)} ({100*len(single_answers)/total_with_answers:.1f}%)")
lines.append(f"- Multiple answers: {len(multi_answers)} ({100*len(multi_answers)/total_with_answers:.1f}%)")

lines.append("\n**Individual Option Frequency:**")
lines.append("| Option | Count |")
lines.append("|--------|-------|")
for opt in ['A', 'B', 'C', 'D']:
    lines.append(f"| {opt} | {answer_counts.get(opt, 0)} |")

lines.append("\n**Multi-answer Patterns (top 10):**")
lines.append("| Pattern | Count |")
lines.append("|---------|-------|")
multi_counter = Counter(multi_answers)
for pattern, count in multi_counter.most_common(10):
    lines.append(f"| {','.join(pattern)} | {count} |")

# Text Length Analysis
lines.append("\n\n## 4. Text Length Analysis\n")
event_lens = [len(q.get('target_event', '')) for q in train_dev_questions]
option_lens = []
for q in train_dev_questions:
    for opt in ['option_A', 'option_B', 'option_C', 'option_D']:
        option_lens.append(len(q.get(opt, '')))

lines.append("**Target Event Length (characters):**")
lines.append(f"- Min: {min(event_lens)}, Max: {max(event_lens)}, Avg: {sum(event_lens)/len(event_lens):.1f}")
lines.append("\n**Option Length (characters):**")
lines.append(f"- Min: {min(option_lens)}, Max: {max(option_lens)}, Avg: {sum(option_lens)/len(option_lens):.1f}")

# Document Analysis
lines.append("\n\n## 5. Document Analysis\n")
train_docs = all_docs['train_data']
doc_counts = []
content_lens = []
sources = Counter()

for topic in train_docs:
    docs = topic.get('docs', [])
    doc_counts.append(len(docs))
    for doc in docs:
        content = doc.get('content', '')
        if content:
            content_lens.append(len(content))
        source = doc.get('source', 'Unknown')
        sources[source] += 1

if doc_counts:
    lines.append(f"**Documents per Topic:** Min={min(doc_counts)}, Max={max(doc_counts)}, Avg={sum(doc_counts)/len(doc_counts):.1f}")
if content_lens:
    lines.append(f"\n**Document Content Length (chars):** Min={min(content_lens)}, Max={max(content_lens)}, Avg={sum(content_lens)/len(content_lens):.0f}")

lines.append("\n**Top 10 News Sources:**")
lines.append("| Source | Count |")
lines.append("|--------|-------|")
for source, count in sources.most_common(10):
    lines.append(f"| {source} | {count} |")

# Topic Analysis
lines.append("\n\n## 6. Topic Analysis\n")
topic_ids_train = set(q.get('topic_id') for q in all_questions['train_data'])
topic_ids_dev = set(q.get('topic_id') for q in all_questions['dev_data'])
topic_ids_test = set(q.get('topic_id') for q in all_questions['test_data'])

lines.append(f"- Unique topics in train: {len(topic_ids_train)}")
lines.append(f"- Unique topics in dev: {len(topic_ids_dev)}")
lines.append(f"- Unique topics in test: {len(topic_ids_test)}")
lines.append(f"\n- Topic overlap (train & dev): {len(topic_ids_train & topic_ids_dev)}")
lines.append(f"- Topic overlap (train & test): {len(topic_ids_train & topic_ids_test)}")

# Sample Topics
lines.append("\n\n## 7. Sample Topics\n")
for i, topic in enumerate(train_docs[:5]):
    topic_text = topic.get('topic', 'N/A')
    if len(topic_text) > 150:
        topic_text = topic_text[:150] + "..."
    lines.append(f"{i+1}. **Topic ID {topic.get('topic_id')}:** {topic_text}")

# File Size
lines.append("\n\n## 8. File Size Summary\n")
lines.append("| Split | Questions | Docs |")
lines.append("|-------|-----------|------|")
total_size = 0
for split in splits:
    q_path = BASE_DIR / split / "questions.jsonl"
    d_path = BASE_DIR / split / "docs.json"
    q_size = os.path.getsize(q_path) / (1024*1024) if q_path.exists() else 0
    d_size = os.path.getsize(d_path) / (1024*1024) if d_path.exists() else 0
    total_size += q_size + d_size
    lines.append(f"| {split.replace('_data', '')} | {q_size:.2f} MB | {d_size:.2f} MB |")

lines.append(f"\n**Total dataset size: {total_size:.2f} MB**")

# Write to file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"EDA report saved to: {OUTPUT_FILE}")
