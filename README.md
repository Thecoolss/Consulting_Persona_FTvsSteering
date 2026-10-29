# The Consultant-ifier

**Three ways to make an LLM say "let's circle back": Prompting vs. Fine-Tuning vs. Activation Steering.**

A head-to-head comparison of three methods for giving `Llama-3.1-8B-Instruct` a
management-consultant persona, evaluated with an LLM-as-Judge and presented in an
interactive Streamlit app.

## Methods compared

| Method | How the persona is applied |
|---|---|
| **Baseline** (control) | None — the raw model, for reference |
| **Prompting** | System-prompt engineering (minimal / medium / detailed variants) |
| **Fine-tuning** | QLoRA adapter trained on 636 synthetic persona pairs |
| **Steering** | Contrastive activation vector (mean difference over response tokens) added to the residual stream at one layer |

All four are evaluated on the **same 50 fixed prompts** across 5 categories, with
identical greedy decoding for a fair comparison.

## Evaluation

An LLM judge (Claude Sonnet 4.5 on Bedrock) scores every response 1–5 on five
dimensions: **persona adherence, content relevance, fluency, consistency,
naturalness**. Analysis reports per-method means with 95% CIs and **paired
Wilcoxon signed-rank tests** (Holm-corrected) so "winner" claims are defensible.

The data **generator** (Claude Haiku 4.5) is deliberately a different model from
the **judge** (Claude Sonnet 4.5) to avoid the fine-tuned model simply learning
to mimic the judge's style.

## Project layout

```
config.py                 All hyperparameters, paths, model IDs
persona_spec.md           Single source of truth for the persona
s3_sync.py                Back up / restore artifacts to S3
common/                   Shared Bedrock, model-loading, IO helpers
data/                     Data generators + fixed test prompts
methods/prompting/        System prompts + runner (+ baseline control)
methods/finetuning/       QLoRA train / merge / inference
methods/steering/         Vector extraction, layer/scale sweep, steering hooks, runner
evaluation/               Judge rubrics, Bedrock runner, analysis + charts
app/                      Streamlit demo
outputs/                  Per-method response JSON
evaluation/results/       Scores, stats, charts
```

## Setup

### Local (data generation + evaluation, CPU only)
```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install boto3 python-dotenv numpy pandas tqdm jsonlines scipy matplotlib
# requires: aws configure (us-east-1) and HF_TOKEN in .env
```

### GPU box (fine-tuning + steering + inference)
Use a g5.xlarge with the Deep Learning OSS Nvidia Driver AMI (PyTorch, Ubuntu 22.04), then:
```bash
source /opt/pytorch/bin/activate
pip install -r requirements.txt
# copy .env (HF_TOKEN) to the instance with scp; never commit it
```
Tested with `transformers` 5.21, `trl` 1.16, `peft` 0.21 and `torch` 2.7. The
training script uses the trl 1.x API (`SFTConfig`, prompt/completion datasets)
and will not run on trl 0.x.

See `aws_setup.md` for the full instance/IAM/S3/Bedrock walkthrough.

## Pipeline

```bash
# Phase 1 — data (local, Bedrock, ~$2-4)
python data/make_test_prompts.py
python data/generate_finetuning_data.py
python data/generate_steering_pairs.py
python s3_sync.py upload --dir data

# Phase 2 — prompting + baseline (GPU)
python methods/prompting/run_prompting.py
python s3_sync.py upload --dir outputs

# Phase 3 — fine-tuning (GPU)
python methods/finetuning/train_qlora.py
python methods/finetuning/run_finetuning.py
python s3_sync.py upload

# Phase 4 — steering (GPU)
python methods/steering/extract_vectors.py     # one unit vector per layer 12-24
python methods/steering/sweep_steering.py      # picks layer + scale on held-out prompts
python methods/steering/run_steering.py --layer <L> --scale <S>   # from steering_vectors/sweep_results.json
python s3_sync.py upload

# Phase 5 — evaluate (local, Bedrock)
python evaluation/run_evaluation.py
python evaluation/analyze_results.py
python s3_sync.py upload

# Phase 6 — demo
streamlit run app/streamlit_app.py --server.port 8501
```

## Status

Complete. 350 responses generated, all scored by the LLM judge, analyzed.

- **Data:** 636 fine-tuning pairs and 247 aligned steering pairs (v1). Steering
  v2 adds 219 *matched* contrastive pairs (each persona answer rewritten in plain
  language at the same length).
- **Fine-tuning:** QLoRA, 3 epochs, 120 optimizer steps, ~21 min on one A10G.
  Mean training loss 1.35.
- **Steering v1:** single layer, large absolute scale. Collapsed into repetition
  past ~100 words (44/50 outputs degenerate).
- **Steering v2:** matched pairs, norm-preserving hook, held-out two-stage sweep
  picked layer 12 at alpha 0.5. Fixed the collapse (1/50) but the persona stayed
  weak — one steering direction carries a register, not a specific vocabulary.

## Results

LLM-as-judge (Claude Sonnet 4.5), 1-5 on 50 fixed prompts. Full tables, radar and
per-dimension charts in `evaluation/results/`.

| Method | Persona | Content | Fluency | Consistency | Naturalness | Mean |
|---|---|---|---|---|---|---|
| **Fine-tuned (QLoRA)** | 4.98 | 4.76 | 3.32 | 4.96 | 4.34 | **4.47** |
| Prompt — detailed | 5.00 | 4.52 | 3.40 | 4.96 | 3.86 | 4.35 |
| Prompt — medium | 5.00 | 4.40 | 3.42 | 4.80 | 3.92 | 4.31 |
| Prompt — minimal | 4.98 | 3.48 | 3.94 | 4.04 | 2.34 | 3.76 |
| Steering v2 | 3.70 | 4.27 | 3.76 | 2.82 | 1.63 | 3.24 |
| Baseline (control) | 1.22 | 4.92 | 4.72 | 1.02 | 1.00 | 2.58 |
| Steering v1 | 3.90 | 1.31 | 1.06 | 1.49 | 1.00 | 1.75 |

Findings (paired Wilcoxon, Holm-corrected):

- **Fine-tuning and the detailed/medium prompt are statistically tied** on persona
  adherence; all three max it out. Fine-tuning edges ahead on naturalness and
  content but not significantly.
- **Prompt detail trades content for persona:** the minimal prompt still nails the
  persona (4.98) but drops content relevance to 3.48, significantly below the
  others.
- **Every persona method costs fluency** (3.3-3.9 vs 4.72 baseline) — the judge
  reads heavy jargon as awkward.
- **Steering v2 fixed v1's collapse** (content 1.31 -> 4.27, fluency 1.06 -> 3.76,
  both significant) but could not match prompting/fine-tuning on persona,
  consistency or naturalness. It produces structured, mildly formal answers rather
  than a sustained consultant voice.

Caveats recorded in the data: 8/1750 judge calls were refused by both the primary
and fallback judge and are unscored; 18 were scored by the fallback judge
(Sonnet 4.6) and tagged `judge: "fallback"` in `evaluation_results.json`.

## Cost & AWS services

- **EC2** g5.xlarge (~$1/hr) — GPU compute for fine-tuning, steering, inference
- **S3** — artifact storage (survives instance termination)
- **Bedrock** — Claude Haiku 4.5 (data gen) + Claude Sonnet 4.5 (judge, with
  Sonnet 4.6 as a refusal fallback)
- **CloudWatch** — training-loss metrics

> Remember to **stop/terminate the EC2 instance** when done. S3 holds all
> artifacts, so you can safely re-download later with `python s3_sync.py download`.

## Notes

- No secrets or large model files are committed (see `.gitignore`).
- The persona satirizes a *communication style*, not people.
