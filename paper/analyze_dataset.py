# -*- coding: utf-8 -*-
"""
Deep Dataset Analysis for Paper
"""
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\semeval2026-task12-dataset')
OUTPUT = Path(r'd:\Work\SemEval 2026 Task 12 - Abductive Event Reasoning\paper\figures')
OUTPUT.mkdir(parents=True, exist_ok=True)

# Set style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
})

def load_data(split):
    with open(DATA/split/'questions.jsonl', 'r', encoding='utf-8') as f:
        questions = [json.loads(l) for l in f if l.strip()]
    with open(DATA/split/'docs.json', 'r', encoding='utf-8') as f:
        docs = json.load(f)
    return questions, docs

print("Loading data...")
train_q, train_docs = load_data('train_data')
dev_q, dev_docs = load_data('dev_data')
test_q, test_docs = load_data('test_data')

print(f"Train: {len(train_q)}, Dev: {len(dev_q)}, Test: {len(test_q)}")

# ============================================================================
# ANALYSIS 1: Answer Option Distribution
# ============================================================================
print("\n" + "="*60)
print("ANSWER OPTION DISTRIBUTION")
print("="*60)

answer_counter = Counter()
combo_counter = Counter()
for q in train_q:
    ans = q.get('golden_answer', '')
    labels = sorted([x.strip() for x in ans.split(',') if x.strip()])
    combo_counter[','.join(labels)] += 1
    for l in labels:
        answer_counter[l] += 1

print(f"A: {answer_counter.get('A', 0)}")
print(f"B: {answer_counter.get('B', 0)}")
print(f"C: {answer_counter.get('C', 0)}")
print(f"D (None): {answer_counter.get('D', 0)}")

print("\nTop answer combinations:")
for combo, count in combo_counter.most_common(10):
    print(f"  {combo}: {count}")

# ============================================================================
# ANALYSIS 2: Document Statistics
# ============================================================================
print("\n" + "="*60)
print("DOCUMENT STATISTICS")
print("="*60)

doc_per_topic = []
doc_lengths = []
for doc_info in train_docs:
    docs = doc_info.get('docs', [])
    doc_per_topic.append(len(docs))
    for doc in docs:
        content = doc.get('content', '') or doc.get('snippet', '')
        doc_lengths.append(len(content))

print(f"Topics: {len(train_docs)}")
print(f"Total docs: {len(doc_lengths)}")
print(f"Docs per topic: avg={np.mean(doc_per_topic):.1f}, min={min(doc_per_topic)}, max={max(doc_per_topic)}")
print(f"Doc length: avg={np.mean(doc_lengths):.0f}, min={min(doc_lengths)}, max={max(doc_lengths)} chars")

# ============================================================================
# ANALYSIS 3: Event Length
# ============================================================================
print("\n" + "="*60)
print("EVENT LENGTH")
print("="*60)

event_lengths = [len(q.get('target_event', '')) for q in train_q]
print(f"Avg: {np.mean(event_lengths):.0f} chars")
print(f"Min/Max: {min(event_lengths)}/{max(event_lengths)} chars")

# ============================================================================
# FIGURE: Improved Answer Distribution
# ============================================================================
print("\n" + "="*60)
print("GENERATING FIGURES")
print("="*60)

# Figure: Answer option frequency
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

# Left: Option frequency
options = ['A', 'B', 'C', 'D\n(None)']
counts = [answer_counter.get('A', 0), answer_counter.get('B', 0), 
          answer_counter.get('C', 0), answer_counter.get('D', 0)]
colors = ['#4472C4', '#ED7D31', '#70AD47', '#A5A5A5']
bars = axes[0].bar(options, counts, color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_ylabel('Frequency')
axes[0].set_title('(a) Answer Option Distribution', fontweight='bold')
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                 str(count), ha='center', fontsize=9)
axes[0].set_ylim(0, max(counts) * 1.15)

# Right: Multi-label combinations
top_combos = combo_counter.most_common(7)
combo_labels = [c[0] for c in top_combos]
combo_counts = [c[1] for c in top_combos]
y_pos = np.arange(len(combo_labels))
bars = axes[1].barh(y_pos, combo_counts, color='#5B9BD5', edgecolor='black', linewidth=0.3)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(combo_labels)
axes[1].set_xlabel('Count')
axes[1].set_title('(b) Top Answer Combinations', fontweight='bold')
for bar, count in zip(bars, combo_counts):
    axes[1].text(count + 10, bar.get_y() + bar.get_height()/2, 
                 str(count), va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT / 'answer_options.pdf', bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT / 'answer_options.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: answer_options.pdf")

# ============================================================================
# FIGURE: Model Scaling with Details (IMPROVED)
# ============================================================================
fig, ax = plt.subplots(figsize=(7, 4))

# Data
models = ['DeBERTa-v3\n(435M)', 'Qwen-7B\n(7B)', 'Qwen-32B\n(32B)']
scores = [0.63, 0.86, 0.90]
params = [0.435, 7, 32]  # in billions
colors = ['#A5A5A5', '#5B9BD5', '#70AD47']

# Create bars
x = np.arange(len(models))
width = 0.6
bars = ax.bar(x, scores, width, color=colors, edgecolor='black', linewidth=1)

# Add score labels
for i, (bar, score) in enumerate(zip(bars, scores)):
    ax.annotate(f'{score:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords='offset points',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add improvement arrows
ax.annotate('', xy=(1, 0.85), xytext=(0.15, 0.65),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(0.5, 0.72, '+0.23', fontsize=11, color='red', fontweight='bold', ha='center')

ax.annotate('', xy=(2, 0.89), xytext=(1.15, 0.87),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(1.5, 0.88, '+0.04', fontsize=10, color='red', fontweight='bold', ha='center')

# Add parameter count text
for i, (bar, param) in enumerate(zip(bars, params)):
    if param < 1:
        param_str = f'{param*1000:.0f}M'
    else:
        param_str = f'{param:.0f}B'
    ax.text(bar.get_x() + bar.get_width()/2, 0.52, 
            param_str, ha='center', fontsize=9, style='italic', color='gray')

# Add novelty annotations
ax.annotate('Fine-tuned\n(QLoRA)', xy=(1, 0.86), xytext=(1.5, 0.75),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1, ls='--'))

ax.annotate('Extended LoRA\n(7 modules)', xy=(2, 0.90), xytext=(2.5, 0.82),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='green', lw=1, ls='--'))

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('AER Score', fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.set_title('Model Scaling Impact: From 435M to 32B Parameters', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add total improvement text
ax.text(2.8, 0.55, 'Total: +0.27', fontsize=10, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

plt.tight_layout()
plt.savefig(OUTPUT / 'model_scaling.pdf', bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT / 'model_scaling.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: model_scaling.pdf (IMPROVED)")

# ============================================================================
# FIGURE: Ablation Waterfall Chart (IMPROVED)
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

components = ['Baseline', 'Dense\nRAG', 'Causal\nBoost', 'Hybrid\n(BM25)', 
              'Qwen-7B', 'Qwen-32B', 'Extended\nLoRA', 'Dev\nTuning']
scores = [0.67, 0.76, 0.80, 0.82, 0.88, 0.89, 0.92, 0.936]
deltas = [0] + [scores[i] - scores[i-1] for i in range(1, len(scores))]

# Create waterfall
x = np.arange(len(components))
colors = ['#A5A5A5'] + ['#4472C4'] * 3 + ['#ED7D31'] * 2 + ['#70AD47'] * 2

# Cumulative bars
bottom = 0.5
for i, (comp, score, delta, color) in enumerate(zip(components, scores, deltas, colors)):
    if i == 0:
        ax.bar(i, score - 0.5, bottom=0.5, color=color, edgecolor='black', linewidth=0.5, width=0.7)
    else:
        ax.bar(i, delta, bottom=scores[i-1], color=color, edgecolor='black', linewidth=0.5, width=0.7)

# Add connecting lines
for i in range(len(scores) - 1):
    ax.plot([i + 0.35, i + 0.65], [scores[i], scores[i]], 'k--', alpha=0.3, linewidth=1)

# Add score labels
for i, score in enumerate(scores):
    ax.text(i, score + 0.01, f'{score:.2f}', ha='center', fontsize=9, fontweight='bold')

# Add delta labels
for i, delta in enumerate(deltas):
    if delta > 0:
        ax.text(i, scores[i] - delta/2 - 0.005, f'+{delta:.2f}', 
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(components, fontsize=9)
ax.set_ylabel('AER Score', fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.set_title('Waterfall: Cumulative Contribution of Each Component', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#A5A5A5', label='Baseline'),
    Patch(facecolor='#4472C4', label='RAG Improvements'),
    Patch(facecolor='#ED7D31', label='Model Scaling'),
    Patch(facecolor='#70AD47', label='Fine-tuning'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT / 'ablation_waterfall.pdf', bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT / 'ablation_waterfall.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: ablation_waterfall.pdf (NEW)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("PAPER-READY STATISTICS")
print("="*60)
print(f"Train: {len(train_q)} questions, {len(train_docs)} topics")
print(f"Dev: {len(dev_q)} questions")
print(f"Test: {len(test_q)} questions")
print(f"Multi-label rate: {sum(1 for q in train_q if ',' in q.get('golden_answer',''))/len(train_q)*100:.1f}%")
print(f"Option D (None) rate: {answer_counter.get('D',0)/sum(answer_counter.values())*100:.1f}%")
print(f"Avg docs per topic: {np.mean(doc_per_topic):.1f}")
print(f"Avg doc length: {np.mean(doc_lengths):.0f} chars")
print("\nAll figures saved to:", OUTPUT)
