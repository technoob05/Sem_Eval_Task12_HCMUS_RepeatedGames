# SemEval 2026 Task 12: Abductive Event Reasoning (AER)

**Team:** HCMUS_RepeatedGames  
**Repository:** [Sem_Eval_Task12_HCMUS_TheFangs](https://github.com/technoob05/Sem_Eval_Task12_HCMUS_TheFangs)

This repository contains the source code, experiments, and system description paper for our participation in the **SemEval 2026 Task 12: Abductive Event Reasoning**.

---

## 📖 Introduction

Understanding the causes behind real-world events is a fundamental aspect of human reasoning. While large language models (LLMs) have made substantial progress in tasks such as information extraction and summarization, they still face challenges in **abductive reasoning** — inferring the most plausible cause of an observed outcome from incomplete and potentially noisy evidence.

The AER task is formulated as a **multiple-choice question answering** problem where the system must output the label(s) of the correct option(s) based on reasoning over the provided context. 

For more details on the competition, please refer to the [Task Website](https://sites.google.com/view/semeval2026-task12/).

---

## 📁 Repository Structure

Our repository is organized as follows to maintain a clean and standardized workflow:

*   **`exp/`**: Contains all experimental scripts investigating different architectures, including:
    *   State-of-the-Art (SOTA) Causal RAG variants.
    *   Multi-agent debate models and ensemble fusion techniques.
    *   Finetuning scripts (LoRA/QLoRA) and zero-shot baseline evaluations on various LLMs (e.g., 7B, 32B, 72B models).
*   **`paper/`**: Contains the LaTeX source code, figures, and styling files necessary to compile our system description paper.
*   **`baseline_train.py` / `baseline_simple.py`**: Initial baseline models provided for the competition.
*   **`create_manual_answers.py`**: Script for generating or formatting answers for manual submissions.
*   **`eda_semeval_task12.py`**: Useful scripts for Exploratory Data Analysis (EDA) on the dataset.
*   **`results.md`**: Tracks the evaluation metrics and benchmark scores achieved across different internal experiments.
*   **`Comp_overview.md`**: An overview of the competition format and evaluation metrics.
*   **`Submission_Guide.md`**: Guide for assembling and creating submission files.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed. The required packages for running our experiments heavily depend on transformers and standard data-science libraries.

*(Optional: Set up a virtual environment)*
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

Install the typical dependencies (Note: exact requirements might vary per experiment in the `exp/` folder):
```bash
pip install torch transformers datasets pandas tqdm scikit-learn
```

### Dataset

Download the official dataset from the [AER Dataset Repository](https://github.com/sooo66/semeval2026-task12-dataset/) and place it in the designated data directory relative to the scripts you are running. 

By default, the scripts expect the dataset to be structured according to the SemEval competition format. (Ensure to check `.gitignore` to avoid committing large datasets directly).

### Running the Winning Model (0.90 Score)

For reproducibility purposes, we have isolated our best-performing official submission (Extended LoRA on Qwen-32B + Hybrid Retrieval + Causal Graphing) into the **`best_submission/`** directory. 

To easily reproduce our 5th-place matching score without navigating all experiments:
```bash
cd best_submission
python main_causalrag.py
```
*(Please refer to `best_submission/README.md` for specific hardware and parameter details).*

### Running Other Experiments

To run any other exploratory script, navigate to the `exp/` directory or run from the root:
```bash
python exp/exp40_full_sota.py
```
*(Replace `exp40_full_sota.py` with your targeted script).*

---

## 📝 Paper Compilation

If you want to compile our system description paper locally:

1. Navigate to the `paper/` directory:
   ```bash
   cd paper
   ```
2. Use standard `pdflatex` and `bibtex` commands to compile:
   ```bash
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```
3. The generated output will be `main.pdf`.

---

## 🤝 Contributors

*   **Team:** HCMUS_RepeatedGames

*For questions or issues involving this repository, please create an issue or contact the team members directly.*
