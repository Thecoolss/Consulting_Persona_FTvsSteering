"""Central configuration — all hyperparameters, paths, and model IDs in one place.

Every script imports from here. No hardcoded paths or model names elsewhere.
"""

import os

# ---------------------------------------------------------------------------
# Base model (all three methods share this)
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "main"

# ---------------------------------------------------------------------------
# Data generation sizes
# ---------------------------------------------------------------------------
NUM_TRAINING_SAMPLES = 750          # Target; v1 dataset landed at 636 after filters
NUM_STEERING_PAIRS = 300            # Target; v1 has 247 aligned pairs
NUM_TEST_PROMPTS = 50               # Fixed eval set

# ---------------------------------------------------------------------------
# Fine-tuning hyperparameters (QLoRA)
# ---------------------------------------------------------------------------
LORA_R = 32
LORA_ALPHA = 64                     # 2x rank
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 1024
WARMUP_RATIO = 0.03
BF16 = True                         # A10G supports bf16

# Quantization (NF4)
QUANT_BITS = 4
QUANT_TYPE = "nf4"
DOUBLE_QUANT = True

# ---------------------------------------------------------------------------
# Generation config — SHARED across ALL methods for a fair comparison.
# Greedy decoding (temperature=0) for full reproducibility.
# ---------------------------------------------------------------------------
GEN_DO_SAMPLE = False
GEN_TEMPERATURE = 0.0
GEN_MAX_NEW_TOKENS = 512
SEED = 42

# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------
STEERING_LAYER_RANGE = (12, 24)     # Middle-to-late layers (Llama 3.1 8B has 32)
# Vectors are L2-normalized (unit length); the residual stream at these layers
# has per-token norms ~20. Useful scales are therefore ~4-10, not <3. The layer
# and scale used for evaluation are picked by methods/steering/sweep_steering.py
# on held-out prompts (see steering_vectors/sweep_results.json).
STEERING_SCALE_DEFAULT = 6.0
STEERING_SCALES_TO_TRY = [4.0, 6.0, 8.0, 10.0]
# Activation capture strategy: "mean" (over response tokens) or "last" (last token).
# "mean" gives a more stable persona direction — see steering_utils.py.
STEERING_CAPTURE = "mean"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_DIMENSIONS = [
    "persona_adherence",   # Does it sound like a consultant?
    "content_relevance",   # Does it answer the question?
    "fluency",             # Grammatically sound, logically structured?
    "consistency",         # Does persona hold or break?
    "naturalness",         # Real persona vs forced caricature?
]

# Methods that get evaluated (includes a plain base-model control).
METHODS = ["baseline", "prompting", "finetuning", "steering"]

# ---------------------------------------------------------------------------
# Paths (local)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
ADAPTER_DIR = "finetuning_adapter"
MERGED_MODEL_DIR = "merged_model"
STEERING_VECTORS_DIR = "steering_vectors"
STEERING_V2_DIR = "steering_vectors_v2"     # vectors from matched pairs + sweep v2
RESULTS_DIR = "evaluation/results"

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.jsonl")
STEERING_POS_PATH = os.path.join(DATA_DIR, "steering_positive.jsonl")
STEERING_NEG_PATH = os.path.join(DATA_DIR, "steering_negative.jsonl")
TEST_PROMPTS_PATH = os.path.join(DATA_DIR, "test_prompts.json")

# ---------------------------------------------------------------------------
# AWS — S3 persistent storage
# ---------------------------------------------------------------------------
S3_BUCKET = "consulting-persona-comparison-382293180751-us-east-1-an"
S3_PREFIX = "v1"
# S3 layout:
#   s3://{bucket}/{prefix}/data/          — training data, steering pairs, test prompts
#   s3://{bucket}/{prefix}/checkpoints/   — LoRA adapter checkpoints
#   s3://{bucket}/{prefix}/steering/      — steering vectors (.pt)
#   s3://{bucket}/{prefix}/outputs/       — response JSON files
#   s3://{bucket}/{prefix}/evaluation/    — eval results, charts

# Directories that s3_sync.py manages
S3_SYNC_DIRS = [DATA_DIR, OUTPUT_DIR, RESULTS_DIR, STEERING_VECTORS_DIR, STEERING_V2_DIR,
                ADAPTER_DIR]

# ---------------------------------------------------------------------------
# AWS — Bedrock (model IDs VERIFIED against the account via US inference profiles).
# NOTE: these must be the "us." inference-profile IDs — bare model IDs are
# rejected for the Claude 4.5 family in this account.
# ---------------------------------------------------------------------------
BEDROCK_REGION = "us-east-1"
# Judge: the stronger model, for consistent scoring.
# NOTE: AWS has signaled deprecation of the Claude 4.5 family. As of the last
# check, 4.5 still invokes. If it stops, set BEDROCK_JUDGE_MODEL in the
# environment (or edit here) to a confirmed-working successor, e.g.
# "us.anthropic.claude-sonnet-4-6". Verified-available judges in this account:
#   us.anthropic.claude-sonnet-4-5-20250929-v1:0   (current)
#   us.anthropic.claude-sonnet-4-6                 (confirmed fallback)
BEDROCK_JUDGE_MODEL = os.environ.get(
    "BEDROCK_JUDGE_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
# Used only when the main judge refuses to score a response (empty reply with
# stop_reason="refusal"). Those scores are tagged judge="fallback" in the cache
# and counted in the analysis summary.
BEDROCK_JUDGE_FALLBACK_MODEL = os.environ.get(
    "BEDROCK_JUDGE_FALLBACK_MODEL", "us.anthropic.claude-sonnet-4-6")
# Generator: a cheaper, DIFFERENT model than the judge (avoids the fine-tuned
# model simply learning to mimic the judge's own style). Data gen is already
# done, so this only matters if you regenerate data.
BEDROCK_GENERATOR_MODEL = os.environ.get(
    "BEDROCK_GENERATOR_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

# Throttling / retry / concurrency
BEDROCK_CALL_DELAY_SECONDS = 0.1   # small per-call delay (concurrency spaces load)
BEDROCK_MAX_RETRIES = 4      # transient errors (timeouts, 503s)
BEDROCK_THROTTLE_RETRIES = 10  # rate-limit errors; jittered backoff up to 30s each
# Sonnet 4.5 in this account accepts roughly 50 requests/minute. More than ~4
# concurrent judge calls just produces throttling.
BEDROCK_READ_TIMEOUT = 120   # seconds per call; a stalled socket raises & retries
BEDROCK_MAX_CONCURRENCY = 8  # parallel Bedrock calls (well under the 5M TPM limit)

# ---------------------------------------------------------------------------
# AWS — CloudWatch (training metrics)
# ---------------------------------------------------------------------------
CLOUDWATCH_NAMESPACE = "ConsultingPersona"
CLOUDWATCH_LOG_GROUP = "/consulting-persona/training"

# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------
AWS_BUDGET_LIMIT = 100.0   # USD — warn loudly as usage approaches this
EC2_HOURLY_RATE = 1.006    # g5.xlarge on-demand, us-east-1 (verify at launch)

# Approx Bedrock pricing (USD per 1M tokens) for running cost estimates.
# These are estimates for console display only — verify against current pricing.
BEDROCK_PRICING = {
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {"input": 3.00, "output": 15.00},
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.00, "output": 5.00},
    "us.anthropic.claude-sonnet-4-6": {"input": 3.00, "output": 15.00},  # assumed = 4.5
}
