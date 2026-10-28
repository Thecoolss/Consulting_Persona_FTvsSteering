# AWS Setup Guide

How the GPU phases were run, including the fixes that were needed along the way.

## Account resources (persist between runs)
- Region `us-east-1`, account `382293180751`
- S3 bucket: `consulting-persona-comparison-382293180751-us-east-1-an`
- IAM role + instance profile `consulting-persona-ec2-role` (S3, Bedrock, CloudWatch)
- Security group `consulting-persona-sg`: SSH (22) and Streamlit (8501) from one IP.
  Update the source IP if yours changes.
- Key pair `Consulting_Benchmark`
- Bedrock: Anthropic use-case form submitted; Haiku 4.5 and Sonnet 4.5 invokable.
  AWS has announced deprecation of the 4.5 family. `us.anthropic.claude-sonnet-4-6`
  is a verified judge fallback (set `BEDROCK_JUDGE_MODEL`).
- GPU quota: 4 vCPU per g5.xlarge; the account allows 4 instances.

No EC2 instances or EBS volumes are kept between runs. All artifacts live in S3.

## Budget protection
- AWS Budgets: a $100 cost budget with tiered email alerts.
- CloudWatch billing alarm (Billing → Total Estimated Charge, **us-east-1 only**).
- Cost Explorer lags up to 24 hours; check it the day after a run.

## Launch a g5.xlarge
- **AMI:** latest "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)".
  The PyTorch env is at `/opt/pytorch` (not a conda env).
- **Instance type:** g5.xlarge (1x A10G 24 GB, 4 vCPU, 16 GB RAM), ~$1.006/hr
- **Storage:** 150 GB gp3, delete on termination
- **IAM instance profile:** `consulting-persona-ec2-role`
- **Tags:** `Project=persona-comparison`

```bash
AMI=$(aws ec2 describe-images --region us-east-1 --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
  --query "reverse(sort_by(Images,&CreationDate))[0].ImageId" --output text)

aws ec2 run-instances --region us-east-1 --image-id "$AMI" --instance-type g5.xlarge \
  --key-name Consulting_Benchmark --security-groups consulting-persona-sg \
  --iam-instance-profile Name=consulting-persona-ec2-role \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=consulting-persona-gpu},{Key=Project,Value=persona-comparison}]'
```

## Set up the instance
```bash
# from your machine
scp -i Consulting_Benchmark.pem .env ubuntu@<ip>:~/      # HF_TOKEN; never via git
ssh -i Consulting_Benchmark.pem ubuntu@<ip>

# on the instance
git clone https://github.com/Thecoolss/Consulting_Persona_FTvsSteering.git consulting_benchmark
cd consulting_benchmark && mv ~/.env .
source /opt/pytorch/bin/activate
pip install -r requirements.txt
python s3_sync.py download --dir data
```

The first model load downloads ~15 GB of Llama weights into
`~/.cache/huggingface`, which takes several minutes.

## Run long jobs detached
A job started in a foreground SSH session dies when the connection drops.
Always detach and log to a file:

```bash
nohup bash -c '
  python -u methods/prompting/run_prompting.py && python s3_sync.py upload --dir outputs &&
  python -u methods/finetuning/train_qlora.py && python s3_sync.py upload --dir finetuning_adapter &&
  python -u methods/finetuning/run_finetuning.py && python s3_sync.py upload --dir outputs
' > ~/run.log 2>&1 &

tail -f ~/run.log        # safe to Ctrl-C; the job keeps running
```

Run `--dry-run` on a script first after any code or library change. It takes a
few minutes and catches API breaks before a long run.

## Two instances in parallel
Fine-tuning and steering both start from the base model, so they can run on
separate instances at once. Total compute is the same; wall-clock drops to the
length of the longer track.

| Instance | Steps | Approx. compute in v1 |
|---|---|---|
| A | prompting + baseline (200 generations, ~70 min), QLoRA training (~25 min), fine-tuned inference (~25 min) | ~2 h |
| B | `extract_vectors.py`, `sweep_steering.py`, `run_steering.py` | ~45 min |

A was the long track because all 200 prompting generations ran there. Splitting
the prompting variants across both instances would shorten the next run.

Each instance pulls `data/` from S3 and uploads only its own outputs. Do not let
a test run on one instance upload a partial file over the other's results.

## When done
1. Confirm every expected file is in S3: `aws s3 ls s3://<bucket>/v1/outputs/`
2. Terminate the instances. Their disks are deleted with them.
3. In the console, check EC2 → Instances, EC2 → Volumes and EC2 Global View.

Run v1 (Oct 27-28): 2 instances x ~4 h, about $8.20 including disks.
