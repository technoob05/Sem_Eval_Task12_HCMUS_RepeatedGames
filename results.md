# SemEval 2026 Task 12: Abductive Event Reasoning
## Experiment Results & Analysis

> **Last Updated:** 2026-01-14 22:52  
> **Current Best:** 0.90 (exp32_32b_extended_lora Qwen-32B) 🏆  
> **Target:** 0.90+ (Achieved!)

---

## 📊 Official Leaderboard Submissions

### 🏆 Best Performers (Score ≥ 0.85)

| ID | Experiment | Method | Date | Score |
|----|------------|--------|------|-------|
| 485547 | **exp32_32b_extended_lora** | **Qwen-32B + Extended LoRA + Hybrid RAG** | 2026-01-13 14:33 | **0.90** 🏆 |
| 485240 | exp43_32b_devtuned | Qwen-32B + Dev Tuning | 2026-01-13 09:00 | 0.88 |
| 484482 | exp32_32b | Qwen-32B + QLoRA | 2026-01-12 17:53 | 0.88 |
| 484738 | exp40_full_sota | Full SOTA (7B) | 2026-01-12 20:54 | 0.87 |
| 484742 | exp39_self_rag | Self-RAG Gating | 2026-01-12 20:56 | 0.86 |
| 483875 | exp22_7b | Qwen2.5-7B + QLoRA | 2026-01-12 09:32 | 0.86 |

### 📈 Mid-Range (Score 0.70 - 0.84)

| ID | Experiment | Method | Date | Score |
|----|------------|--------|------|-------|
| 484890 | 32b_sub_1 | Qwen-32B Initial | 2026-01-12 23:34 | 0.82 |
| 485245 | exp32_32b_submission_1 | Qwen-32B v1 | 2026-01-13 09:05 | 0.81 |
| 483161 | exp22_causalrag | CausalRAG (DeBERTa) | 2026-01-11 18:56 | 0.78 |
| 483880 | exp22_multihop | CausalRAG + Multi-hop | 2026-01-12 09:38 | 0.78 |
| 484888 | exp41_32b_full_sota | 32B Full SOTA (buggy) | 2026-01-12 23:31 | 0.77 |
| 483231 | exp23_cfrag | CF-RAG | 2026-01-11 20:10 | 0.76 |
| 483216 | exp20_care | CARE | 2026-01-11 20:07 | 0.76 |
| 482886 | exp19_contrastive_rag | Contrastive + RAG | 2026-01-11 11:39 | 0.74 |
| 482860 | exp09_sota_causal | Qwen-32B Zero-shot | 2026-01-11 10:22 | 0.70 |

### ❌ Low Performers (Score < 0.70)

| ID | Experiment | Method | Date | Score |
|----|------------|--------|------|-------|
| 482526 | exp01_baseline | DeBERTa Baseline | 2026-01-11 00:59 | 0.63 |
| 482906 | exp15_contrastive | Contrastive Multi-task | 2026-01-11 13:06 | 0.62 |
| 483392 | exp18_multiagent_rag | Multi-Agent + RAG | 2026-01-11 22:08 | 0.59 |
| 482846 | exp13_cisc | CISC | 2026-01-11 09:59 | 0.30 |
| 484012 | exp30_c3 | C3 Calibration | 2026-01-12 13:00 | 0.29 |
| 482849 | exp11_pcsubq | PC-SubQ | 2026-01-11 10:01 | 0.28 |

---

## 📈 Score Progression Timeline

```
Score
0.90 ┤                                                    ╭──○ 0.90 🏆 BEST!
0.88 ┤                                               ╭────┴───○ 0.88 (exp32, exp43)
0.87 ┤                                          ╭────┘         
0.86 ┤                                     ╭────┴───○ 0.86 (exp22_7b, exp39)
0.82 ┤                                ╭────┘
0.78 ┤                           ╭────┴───○ 0.78 (exp22)
0.76 ┤                      ╭────┴───○ 0.76 (exp20, exp23)
0.74 ┤                 ╭────┘        
0.70 ┤            ╭────┘ 0.74 (exp19)
0.63 ┤       ╭────┴─ 0.70 (exp09)
0.30 ┤──────┴────○ 0.63 (exp01) ... 0.28-0.30 (exp11/13)
     └────┬────┬────┬────┬────┬────┬────┬────┬────┬────→ Time
        Jan11 ...                                   Jan13
```

---

## 🔬 Method Comparison

### By Model Size

| Model | Best Exp | Score | Improvement over Baseline |
|-------|----------|-------|---------------------------|
| DeBERTa-v3-large (435M) | exp22_causalrag | 0.78 | +0.15 |
| Qwen2.5-7B (7B) | exp22_7b | 0.86 | +0.23 |
| Qwen2.5-32B (32B) | **exp32_extended_lora** | **0.90** | **+0.27** |

### By Technique

| Technique | Best Score | Key Experiments |
|-----------|------------|-----------------|
| Extended LoRA + Hybrid RAG | **0.90** 🏆 | exp32_32b_extended_lora |
| Dev Set Fine-tuning | 0.88 | exp43_32b_devtuned |
| Label Powerset + RAG-Fusion | 0.87 | exp40_full_sota |
| Self-RAG Gating | 0.86 | exp39_self_rag |
| CausalRAG (7B) | 0.86 | exp22_7b |
| CausalRAG (DeBERTa) | 0.78 | exp22_causalrag |
| Basic RAG | 0.74 | exp19_contrastive_rag |
| Zero-shot LLM | 0.70 | exp09_sota_causal |
| Baseline | 0.63 | exp01_deberta_baseline |

---

## 🎯 Key Findings for Paper

### 1. 🏆 Model Scaling is Critical (+0.27 total gain)
```
DeBERTa-v3-large (435M):  0.63 (baseline)
Qwen2.5-7B (7B):          0.86  → +0.23
Qwen2.5-32B (32B):        0.90  → +0.27
```

### 2. Hybrid RAG > Dense Only (+0.02-0.04)
Combining BM25 with Dense retrieval captures causal keywords better.

### 3. Extended LoRA Works (+0.02)
Training `gate_proj`, `up_proj`, `down_proj` in addition to attention layers improves reasoning.

### 4. Dev Set Fine-tuning Helps
Training 1 extra epoch on validation data squeezes out extra performance.

### 5. Training >> Zero-shot
Fine-tuned 7B (0.86) beats zero-shot 32B (0.70) by 16 points!

---

## 📁 Experiment Files

| File | Method | Score |
|------|--------|-------|
| `Exp043_32B_extended_Lora.py` | **32B + Extended LoRA + Hybrid RAG** | **0.90** 🏆 |
| `exp43_32b_devtuned.py` | 32B + Dev Tuning | 0.88 |
| `exp32_32b.py` | 32B + QLoRA + Hybrid RAG | 0.88 |
| `exp40_full_sota.py` | 7B Full SOTA | 0.87 |
| `exp39_self_rag.py` | Self-RAG Gating | 0.86 |
| `exp22_7b.py` | 7B + QLoRA + CausalRAG | 0.86 |
| `exp22_causalrag.py` | DeBERTa + CausalRAG | 0.78 |
| `exp41_32b_full_sota.py` | 32B SOTA (failed) | 0.77 |
| `exp23_cfrag.py` | CF-RAG | 0.76 |
| `exp20_care_acl.py` | CARE | 0.76 |
| `exp19_contrastive_rag.py` | Contrastive + RAG | 0.74 |
| `exp01_deberta_baseline.py` | DeBERTa Baseline | 0.63 |

---

## ✅ Paper-Ready Contributions

| Contribution | Score | Status |
|--------------|-------|--------|
| Hybrid RAG (BM25 + Dense) for Causal Retrieval | 0.90 | 🏆 SOTA |
| Extended LoRA (7 modules) for 32B | 0.90 | 🏆 SOTA |
| Dev Set Fine-tuning Strategy | 0.88 | ✅ Validated |
| CausalRAG: Causal Graph + Retrieval | 0.78 | ✅ Published |
| Model Scaling Analysis (435M → 7B → 32B) | - | ✅ Ablation Ready |

---

*For EACL/ACL 2025-2026 Paper Submission*
