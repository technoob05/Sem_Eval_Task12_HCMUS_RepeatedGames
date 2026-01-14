# -*- coding: utf-8 -*-
"""
================================================================================
SemEval 2026 Task 12 - Improved EDA Figures for Paper
================================================================================
Generates publication-quality figures following AAdaM best paper format.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

# Configuration
DATA_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
OUTPUT_DIR = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\paper\figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style for publication
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (4, 3),
    'figure.dpi': 150,
})

def load_questions(split):
    with open(DATA_DIR/split/'questions.jsonl', 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def load_docs(split):
    with open(DATA_DIR/split/'docs.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# Load data
print("Loading data...")
train_q = load_questions('train_data')
dev_q = load_questions('dev_data')
test_q = load_questions('test_data')

print(f"Train: {len(train_q)}, Dev: {len(dev_q)}, Test: {len(test_q)}")

# ============================================================================
# Figure 1: Data Distribution (Pie Chart like AAdaM paper)
# ============================================================================
print("\n[Figure 1] Data distribution...")

splits = ['train\n64%', 'dev\n14%', 'test\n22%']
sizes = [len(train_q), len(dev_q), len(test_q)]
colors = ['#4472C4', '#ED7D31', '#70AD47']
explode = (0.02, 0.02, 0.02)

fig, ax = plt.subplots(figsize=(4.5, 4.5))
wedges, texts, autotexts = ax.pie(
    sizes, explode=explode, labels=None, colors=colors,
    autopct='', pctdistance=0.6, startangle=90
)

# Add custom labels with counts
labels_with_counts = [f'train\n{len(train_q)}\n({len(train_q)/sum(sizes)*100:.0f}%)',
                      f'dev\n{len(dev_q)}\n({len(dev_q)/sum(sizes)*100:.0f}%)', 
                      f'test\n{len(test_q)}\n({len(test_q)/sum(sizes)*100:.0f}%)']

ax.legend(wedges, labels_with_counts, title="Split", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
ax.set_title('AER Dataset Distribution', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'data_distribution.pdf', bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_DIR / 'data_distribution.png', bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: data_distribution.pdf")

# ============================================================================
# Figure 2: Answer Distribution (Single vs Multi-label)
# ============================================================================
print("\n[Figure 2] Answer distribution...")

def count_labels(questions):
    single, multi = 0, 0
    label_counts = Counter()
    for q in questions:
        answer = q.get('golden_answer', '')
        labels = [x.strip() for x in answer.split(',') if x.strip()]
        if len(labels) == 1:
            single += 1
        else:
            multi += 1
        label_counts[len(labels)] += 1
    return single, multi, label_counts

train_single, train_multi, train_label_counts = count_labels(train_q)
dev_single, dev_multi, dev_label_counts = count_labels(dev_q)

# Bar chart for answer type distribution
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

# Left: Single vs Multi-label
categories = ['Single-label', 'Multi-label']
train_vals = [train_single, train_multi]
dev_vals = [dev_single, dev_multi]

x = np.arange(len(categories))
width = 0.35

bars1 = axes[0].bar(x - width/2, train_vals, width, label='Train', color='#4472C4')
bars2 = axes[0].bar(x + width/2, dev_vals, width, label='Dev', color='#ED7D31')

axes[0].set_ylabel('Count')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories)
axes[0].set_title('(a) Single vs Multi-label', fontsize=10)
axes[0].legend()
axes[0].bar_label(bars1, padding=2, fontsize=8)
axes[0].bar_label(bars2, padding=2, fontsize=8)

# Right: Number of correct answers
label_range = range(1, 5)
train_by_count = [train_label_counts.get(i, 0) for i in label_range]
dev_by_count = [dev_label_counts.get(i, 0) for i in label_range]

x = np.arange(len(label_range))
bars1 = axes[1].bar(x - width/2, train_by_count, width, label='Train', color='#4472C4')
bars2 = axes[1].bar(x + width/2, dev_by_count, width, label='Dev', color='#ED7D31')

axes[1].set_xlabel('Number of Correct Answers')
axes[1].set_ylabel('Count')
axes[1].set_xticks(x)
axes[1].set_xticklabels(label_range)
axes[1].set_title('(b) Answer Count Distribution', fontsize=10)
axes[1].legend()
axes[1].bar_label(bars1, padding=2, fontsize=7)
axes[1].bar_label(bars2, padding=2, fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'answer_distribution.pdf', bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_DIR / 'answer_distribution.png', bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: answer_distribution.pdf")
print(f"  Train: {train_single} single, {train_multi} multi ({train_multi/len(train_q)*100:.1f}% multi)")

# ============================================================================
# Figure 3: Model Scaling Impact (Clean Bar Chart)
# ============================================================================
print("\n[Figure 3] Model scaling...")

models = ['DeBERTa-v3\n(435M)', 'Qwen-7B', 'Qwen-32B']
scores = [0.63, 0.86, 0.90]
colors = ['#A5A5A5', '#5B9BD5', '#70AD47']

fig, ax = plt.subplots(figsize=(5, 3.5))
bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels
for bar, score in zip(bars, scores):
    ax.annotate(f'{score:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('AER Score')
ax.set_ylim(0.5, 1.0)
ax.set_title('Impact of Model Scaling', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'model_scaling.pdf', bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_DIR / 'model_scaling.png', bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: model_scaling.pdf")

# ============================================================================
# Figure 4: Ablation Study (Clean Table-like Horizontal Bar)
# ============================================================================
print("\n[Figure 4] Ablation study (improved)...")

techniques = ['Baseline', '+Dense RAG', '+Causal Boost', '+Hybrid (BM25)',
              '+Qwen-7B', '+Qwen-32B', '+Extended LoRA', '+Dev Tuning']
scores = [0.63, 0.74, 0.78, 0.80, 0.86, 0.88, 0.89, 0.90]

fig, ax = plt.subplots(figsize=(6.5, 4))

# Create gradient colors
cmap = plt.cm.Blues
colors = [cmap(0.3 + 0.5 * i / len(techniques)) for i in range(len(techniques))]

y_pos = np.arange(len(techniques))
bars = ax.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=0.3, height=0.7)

# Add score labels
for i, (bar, score) in enumerate(zip(bars, scores)):
    ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{score:.2f}', va='center', fontsize=9, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(techniques, fontsize=9)
ax.set_xlabel('AER Score', fontsize=10)
ax.set_xlim(0.55, 1.0)
ax.set_title('Cumulative Ablation Study', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add vertical line at baseline
ax.axvline(x=0.63, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.text(0.635, -0.5, 'Baseline', fontsize=8, color='red', alpha=0.7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ablation_study.pdf', bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_DIR / 'ablation_study.png', bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved: ablation_study.pdf")

# ============================================================================
# Statistics Summary
# ============================================================================
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)
print(f"Train: {len(train_q)} questions, {len(set(q.get('topic_id') for q in train_q))} topics")
print(f"Dev:   {len(dev_q)} questions, {len(set(q.get('topic_id') for q in dev_q))} topics")
print(f"Test:  {len(test_q)} questions, {len(set(q.get('topic_id') for q in test_q))} topics")
print(f"\nAnswer Distribution (Train):")
print(f"  Single-label: {train_single} ({train_single/len(train_q)*100:.1f}%)")
print(f"  Multi-label:  {train_multi} ({train_multi/len(train_q)*100:.1f}%)")
print(f"\nAll figures saved to: {OUTPUT_DIR}")
