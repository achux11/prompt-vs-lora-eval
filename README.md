# Prompting vs. LoRA Fine-Tuning — Evaluation Series

A structured evaluation of LLM tuning strategies: three versioned prompts from a prompt library vs. a LoRA fine-tune of a small language model, on the same task and test set.

**Task:** classify clinical notes into 1 of 6 specialties with **Qwen2.5-0.5B-Instruct**.
**Data:** MTSamples (Kaggle: tboyle10/medicaltranscriptions) — 150 test cases, 900 LoRA training notes.

## Method
- **Prompt library with versioning** (`prompt_library.json`): v1.0 zero-shot, v1.1 "strict" zero-shot, v2.0 few-shot
- **LoRA (PEFT):** r=8, alpha=16 on q/v projections — trains **0.11%** of parameters (540,672 of 494,573,440); loss on answer tokens only
- **Output-validation guardrail:** only exact labels from the allowed list are accepted; anything else counts as a hallucinated/invalid answer
- **Metrics:** accuracy, coverage (valid answers), hallucination rate, latency
- **A/B test:** paired bootstrap 95% CI for LoRA vs. the best prompt

## Results (150 test cases, T4 GPU)
| Strategy | Accuracy | Coverage | Hallucination rate | Avg latency |
|---|---|---|---|---|
| **LoRA fine-tune** | **0.753** | 0.993 | **0.007** | 0.257s |
| v1.0 zero-shot | 0.367 | 0.633 | 0.367 | 0.352s |
| v2.0 few-shot | 0.347 | 1.000 | 0.000 | 0.190s |
| v1.1 zero-shot "strict" | 0.167 | 0.347 | 0.653 | 0.159s |

**A/B:** LoRA vs. best prompt = **+0.387 accuracy (95% CI +0.300 to +0.480)**.

## Findings
- LoRA doubled accuracy and nearly eliminated invalid answers while training 0.11% of weights.
- **Prompt regression caught:** the "stricter" v1.1 prompt *raised* hallucinations to 65.3% — every prompt change needs a fixed evaluation set before release.
- Few-shot fixed the output format (100% coverage) but not the judgment (34.7% accuracy).

## Run
- GPU: `prompt_vs_lora_eval.py` (Colab T4, ~25 min)
- No GPU: `prompt_vs_lora_eval_cpu.py` (smaller sample, ~30 min)
```bash
pip install kagglehub peft accelerate transformers torch scikit-learn pandas
```
