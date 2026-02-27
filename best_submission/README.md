# CausalRAG - Best Performing Model

This folder contains the standalone script for reproducing our best submission for **SemEval-2026 Task 12: Abductive Event Reasoning**. 

This specific configuration (`main_causalrag.py`) achieved a score of **0.90** on the official test set (Ranking tied for 5th).

## Architecture Details

*   **Base Model:** `Qwen/Qwen2.5-32B-Instruct`
*   **Fine-Tuning:** Extended LoRA (QLoRA 4-bit) targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
*   **Retrieval:** Hybrid Retrieval (BM25 + Dense Sentence-BERT) with Causal Graph Boosting ($\alpha=0.3$ for BM25).
*   **Optimization:** Binary Cross-Entropy (BCE) over 4 option logits for multi-label classification.

## Reproducing the Results

To run this experiment, ensure you have the required dependencies (check the main repository `README.md`) and the official SemEval dataset placed in the expected directory structure.

```bash
# Run the model (Requires ~80GB VRAM, e.g., NVIDIA H100)
python main_causalrag.py
```

*Note: The script was originally named `Exp043_32B_extended_Lora.py` during our iterative development phase.*
