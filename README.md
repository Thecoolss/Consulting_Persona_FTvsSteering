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
| **Fine-tuning** | QLoRA adapter trained on ~750 synthetic persona pairs |
| **Steering** | Contrastive activation vector added to the residual stream at inference |

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
methods/steering/         Vector extraction, steering hooks, runner
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
Use a g5.xlarge with the Deep Learning AMI, then:
```bash
pip install -r requirements.txt
huggingface-cli login     # or put HF_TOKEN in .env
```
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
python methods/steering/extract_vectors.py     # pick best layer from validation
python methods/steering/run_steering.py --layer <best>
python s3_sync.py upload

# Phase 5 — evaluate (local, Bedrock)
python evaluation/run_evaluation.py
python evaluation/analyze_results.py
python s3_sync.py upload

# Phase 6 — demo
streamlit run app/streamlit_app.py --server.port 8501
```

## Results

_Populated after the evaluation run — see `evaluation/results/analysis_summary.md`,
`radar.png`, and the per-dimension bar charts._

## Cost & AWS services

- **EC2** g5.xlarge (~$1/hr) — GPU compute for fine-tuning, steering, inference
- **S3** — artifact storage (survives instance termination)
- **Bedrock** — Claude Haiku 4.5 (data gen) + Claude Sonnet 4.5 (judge)
- **CloudWatch** — training-loss metrics

> Remember to **stop/terminate the EC2 instance** when done. S3 holds all
> artifacts, so you can safely re-download later with `python s3_sync.py download`.

## Notes

- No secrets or large model files are committed (see `.gitignore`).
- The persona satirizes a *communication style*, not people.
