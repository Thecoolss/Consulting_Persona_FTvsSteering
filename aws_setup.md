# AWS Setup Guide

Everything needed to run the GPU phases. Items marked ✅ are already done for
this account (verified from the CLI).

## Already in place ✅
- AWS CLI configured (account `382293180751`, region `us-east-1`)
- S3 bucket: `consulting-persona-comparison-382293180751-us-east-1-an`
- IAM role `consulting-persona-ec2-role` with S3 + Bedrock + CloudWatch, instance profile created
- Bedrock Anthropic use-case agreement accepted; Haiku 4.5 + Sonnet 4.5 invokable
- HF token valid; Llama 3.1 8B access granted
- Billing alarms configured
- GPU quota for g5.xlarge available (4 instances)

## Budget protection
- AWS Budgets: a $100 cost budget with alerts at the chosen thresholds.
- CloudWatch billing alarm (Billing → Total Estimated Charge, **us-east-1 only**).

## Launch a g5.xlarge
- **Region:** us-east-1
- **AMI:** Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04) — CUDA/PyTorch/conda preinstalled
- **Instance type:** g5.xlarge (1× A10G 24 GB, 4 vCPU, 16 GB RAM)
- **Storage:** 150 GB gp3
- **IAM instance profile:** `consulting-persona-ec2-role`
- **Security group:** SSH (22) from your IP; TCP 8501 from your IP (Streamlit)
- **Key pair:** `Consulting_Benchmark.pem`
- **Tags:** `Name=consulting-persona-gpu`, `Project=persona-comparison`

## SSH in and set up
```bash
chmod 400 Consulting_Benchmark.pem           # on your local machine, once
ssh -i Consulting_Benchmark.pem ubuntu@<instance-ip>
conda activate pytorch
git clone <repo-url> && cd consulting-persona-comparison
pip install -r requirements.txt

# verify keyless AWS access via the instance role:
aws s3 ls s3://consulting-persona-comparison-382293180751-us-east-1-an/
```

## Hugging Face
```bash
huggingface-cli login     # paste HF token, or copy .env with HF_TOKEN
```

## Pull artifacts and run
```bash
python s3_sync.py download            # get data generated locally
# ... run phases (see README) ...
python s3_sync.py upload              # after every phase
```

## ⚠️ When done
At ~$1/hr, a forgotten instance is a slow leak. Final `python s3_sync.py upload`,
then **stop or terminate** the instance in the console. To resume: launch a new
instance, attach the same role, `python s3_sync.py download`.

### Running two instances in parallel (optional, time-saving)
Fine-tuning (base model) and steering (base model) are independent, so they can
run on separate instances at the same time. Same total instance-hours (same
cost), roughly half the wall-clock. Downside: two instances to remember to stop.
