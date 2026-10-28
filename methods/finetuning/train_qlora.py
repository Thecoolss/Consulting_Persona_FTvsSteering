"""Method 2 — QLoRA fine-tuning of Llama 3.1 8B on the consulting dataset.

Trains a LoRA adapter (4-bit NF4 base), logs loss to CloudWatch, prints a cost
estimate, and saves the adapter to finetuning_adapter/.

Usage:
    python methods/finetuning/train_qlora.py
    python methods/finetuning/train_qlora.py --dry-run   # tiny run to test the pipeline
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from common import model_utils
from methods.prompting.system_prompts import MINIMAL


def load_dataset_records() -> list[dict]:
    recs = []
    with open(config.TRAIN_DATA_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not recs:
        sys.exit(f"No training data in {config.TRAIN_DATA_PATH}. Generate it first.")
    return recs


def build_cloudwatch_callback():
    """A TrainerCallback that pushes step loss to CloudWatch (no-op if unavailable)."""
    from transformers import TrainerCallback
    try:
        import boto3
        cw = boto3.client("cloudwatch", region_name=config.BEDROCK_REGION)
    except Exception:  # noqa: BLE001
        cw = None

    def put(name: str, value: float, unit: str = "None") -> None:
        if cw is None:
            return
        try:
            cw.put_metric_data(
                Namespace=config.CLOUDWATCH_NAMESPACE,
                MetricData=[{"MetricName": name, "Value": float(value), "Unit": unit}],
            )
        except Exception:  # noqa: BLE001
            pass  # never let metric logging break training

    class CWCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                put("StepLoss", logs["loss"])

    return CWCallback(), put


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch  # noqa: F401
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    model_utils.preflight_hf()
    model_utils.set_seeds()

    records = load_dataset_records()
    if args.dry_run:
        records = records[:8]

    tokenizer = model_utils.load_tokenizer()
    # Llama 3.1 ships a reserved pad token. Using it (instead of eos/<|eot_id|>)
    # keeps padding from colliding with the end-of-turn token the model must learn.
    if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # Conversational prompt/completion format (trl >= 1.x): trl applies the chat
    # template itself and computes loss only on the assistant completion.
    def to_example(rec: dict) -> dict:
        return {
            "prompt": [
                {"role": "system", "content": MINIMAL},
                {"role": "user", "content": rec["instruction"]},
            ],
            "completion": [
                {"role": "assistant", "content": rec["response"]},
            ],
        }

    dataset = Dataset.from_list([to_example(r) for r in records])
    print(f"Training examples: {len(dataset)}")

    print("Loading base model (4-bit, training mode)...")
    model = model_utils.load_base_model(for_training=True)

    lora = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # transformers 5.x dropped warmup_ratio; derive warmup_steps from it instead.
    epochs = 1 if args.dry_run else config.NUM_EPOCHS
    steps_per_epoch = math.ceil(
        len(dataset) / (config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, round(total_steps * config.WARMUP_RATIO))
    print(f"Optimizer steps: {total_steps} ({steps_per_epoch}/epoch x {epochs}) | "
          f"warmup_steps={warmup_steps}")

    train_args = SFTConfig(
        output_dir=config.ADAPTER_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        bf16=config.BF16,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=1 if args.dry_run else 10,
        save_strategy="epoch",
        save_total_limit=1,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=config.SEED,
        max_length=config.MAX_SEQ_LENGTH,
        packing=False,
        gradient_checkpointing=True,
    )

    cw_callback, put_metric = build_cloudwatch_callback()

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,
        callbacks=[cw_callback],
    )
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    t0 = time.time()
    result = trainer.train()
    train_seconds = time.time() - t0

    trainer.save_model(config.ADAPTER_DIR)
    tokenizer.save_pretrained(config.ADAPTER_DIR)

    hours = train_seconds / 3600
    est_cost = hours * config.EC2_HOURLY_RATE
    final_loss = getattr(result, "training_loss", float("nan"))

    put_metric("TrainingLoss", final_loss)
    put_metric("TrainingTimeHours", hours, "Count")
    put_metric("EstimatedCostUSD", est_cost, "Count")
    put_metric("TrainingEpochs", epochs, "Count")

    print("\n--- TRAINING SUMMARY ---")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Training time: {hours:.2f}h | Estimated EC2 cost: ${est_cost:.2f}")
    print(f"Adapter saved to: {config.ADAPTER_DIR}")
    if est_cost > config.AWS_BUDGET_LIMIT * 0.8:
        print("WARNING: approaching the $100 budget limit — watch instance uptime.")
    print("\nNext: python s3_sync.py upload --dir finetuning_adapter")


if __name__ == "__main__":
    main()
