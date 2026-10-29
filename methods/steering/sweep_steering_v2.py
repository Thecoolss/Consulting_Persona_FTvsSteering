"""Steering v2 — choose layers, strength and norm handling on held-out prompts.

Changes from v1 (sweep_steering.py):
  * Vectors come from matched pairs (data/generate_matched_pairs.py), so length
    and layout no longer leak into the direction.
  * Strength is relative: alpha x the layer's typical activation norm, so the
    same alpha means the same relative push at every layer.
  * Multi-layer bands: a smaller push at several layers instead of one big push.
  * preserve_norm: each steered token is rescaled to its original norm.
  * v1 was picked on 120-token outputs, but its text collapsed into repetition
    after ~100 words (40/50 test answers). Stage 1 now generates 300 tokens and
    flags collapse anywhere in the text.
  * Stage 2 scores the best Stage-1 configs with the Bedrock judge on 15 more
    held-out prompts at the full 512-token length.

Selection: Stage 1 keeps configs with zero collapsed outputs and ranks them by
jargon count; the top 4 go to Stage 2. The pick maximizes the mean of judged
persona_adherence, content_relevance and fluency.

Usage:
    python methods/steering/sweep_steering_v2.py
    python methods/steering/sweep_steering_v2.py --quick     # pipeline smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from common import io_utils, model_utils
from common.bedrock import BedrockClient
from data.generate_finetuning_data import jargon_count
from evaluation.run_evaluation import score_one
from methods.steering import steering_utils

JUDGE_DIMS = ["persona_adherence", "content_relevance", "fluency"]

# (label, layers, alpha, preserve_norm)
CONFIGS = (
    [("v1-equivalent L12", [12], 0.74, False)]
    + [(f"L{l} a{a}", [l], a, True) for l in (12, 16) for a in (0.5, 0.75, 1.0)]
    + [(f"L{lo}-{lo + 4} a{a}", list(range(lo, lo + 5)), a, True)
       for lo in (12, 14, 16) for a in (0.2, 0.3, 0.45)]   # vectors exist for 12-24 only
)


def collapsed(text: str, window: int = 80, min_ratio: float = 0.45) -> bool:
    """True if any 80-word window repeats itself heavily (distinct-word ratio low)."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    if len(words) < 25:
        return True
    for i in range(0, max(1, len(words) - window + 1), window // 2):
        w = words[i:i + window]
        if len(w) >= window // 2 and len(set(w)) / len(w) < min_ratio:
            return True
    return False


def heldout_prompts() -> list[str]:
    """Fine-tuning instructions not used for steering pairs and not in the test set."""
    used = set()
    for path in (config.STEERING_POS_PATH,
                 os.path.join(config.DATA_DIR, "steering_positive_v2.jsonl")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                used |= {json.loads(l)["instruction"] for l in fh if l.strip()}
    test = {p["prompt"] for p in io_utils.load_test_prompts()}
    with open(config.TRAIN_DATA_PATH, encoding="utf-8") as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    out, seen = [], set()
    for r in recs:
        if r["instruction"] in used or r["instruction"] in test or r["category"] in seen:
            continue
        seen.add(r["category"])
        out.append(r["instruction"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default=config.STEERING_V2_DIR)
    ap.add_argument("--quick", action="store_true", help="2 configs, 1+2 prompts, short")
    args = ap.parse_args()

    model_utils.set_seeds()
    pool = heldout_prompts()
    configs = CONFIGS
    s1_prompts, s2_prompts = pool[:6], pool[6:21]
    s1_tokens, top_k = 300, 4
    if args.quick:
        configs, s1_prompts, s2_prompts, s1_tokens, top_k = CONFIGS[:2], pool[:1], pool[1:3], 80, 1
    print(f"Held-out prompts: stage1={len(s1_prompts)} stage2={len(s2_prompts)} "
          f"(pool {len(pool)}) | configs={len(configs)}", flush=True)

    tokenizer = model_utils.load_tokenizer()
    model = model_utils.load_base_model()
    norms = steering_utils.load_norms(args.vectors)
    all_layers = sorted({l for _, ls, _, _ in configs for l in ls})
    vectors = steering_utils.load_vector_set(all_layers, args.vectors)

    def gen(prompt, cfg, max_new_tokens):
        if cfg is None:
            text, _ = model_utils.generate(model, tokenizer, prompt, None,
                                           max_new_tokens=max_new_tokens)
            return text
        _, layers, alpha, pn = cfg
        with steering_utils.multi_steering(
                model, {l: vectors[l] for l in layers},
                {l: alpha * norms[l] for l in layers}, preserve_norm=pn):
            text, _ = model_utils.generate(model, tokenizer, prompt, None,
                                           max_new_tokens=max_new_tokens)
        return text

    t0 = time.time()
    # ---------------- Stage 1: local screen ----------------
    print(f"\n=== Stage 1: {s1_tokens} tokens, local metrics ===", flush=True)
    print(f"{'config':22s} {'jargon':>7s} {'collapsed':>9s}", flush=True)
    base_texts = [gen(p, None, s1_tokens) for p in s1_prompts]
    print(f"{'(unsteered)':22s} {mean(map(jargon_count, base_texts)):7.2f} "
          f"{sum(map(collapsed, base_texts)):5d}/{len(s1_prompts)}", flush=True)
    stage1 = []
    for cfg in configs:
        texts = [gen(p, cfg, s1_tokens) for p in s1_prompts]
        row = {"label": cfg[0], "layers": cfg[1], "alpha": cfg[2], "preserve_norm": cfg[3],
               "jargon": mean(map(jargon_count, texts)),
               "collapsed": sum(map(collapsed, texts)), "sample": texts[0][:300]}
        stage1.append(row)
        print(f"{cfg[0]:22s} {row['jargon']:7.2f} {row['collapsed']:5d}/{len(s1_prompts)}",
              flush=True)

    safe = sorted([r for r in stage1 if r["collapsed"] == 0],
                  key=lambda r: -r["jargon"])[:top_k]
    print(f"\nStage 1 done in {time.time() - t0:.0f}s. "
          f"Safe configs: {sum(r['collapsed'] == 0 for r in stage1)}/{len(stage1)}", flush=True)
    if not safe:
        print("NO SAFE CONFIG in stage 1.")
        sys.exit(2)

    # ---------------- Stage 2: full length + judge ----------------
    print(f"\n=== Stage 2: {len(safe)} configs x {len(s2_prompts)} prompts, "
          f"{config.GEN_MAX_NEW_TOKENS} tokens, judged ===", flush=True)
    client = BedrockClient()
    stage2 = []
    for r in safe:
        cfg = (r["label"], r["layers"], r["alpha"], r["preserve_norm"])
        texts = [gen(p, cfg, None) for p in s2_prompts]
        jobs = [(p, t, d) for p, t in zip(s2_prompts, texts) for d in JUDGE_DIMS]
        with ThreadPoolExecutor(max_workers=4) as ex:
            scored = list(ex.map(lambda j: score_one(client, j[0], j[1], j[2]), jobs))
        per_dim = {d: [] for d in JUDGE_DIMS}
        for (_, _, d), s in zip(jobs, scored):
            if s.get("score") is not None:
                per_dim[d].append(s["score"])
        row = dict(r)
        row.update({f"judge_{d}": (mean(v) if v else None) for d, v in per_dim.items()})
        valid = [row[f"judge_{d}"] for d in JUDGE_DIMS if row[f"judge_{d}"] is not None]
        row["judge_composite"] = mean(valid) if valid else 0.0
        row["collapsed_full"] = sum(map(collapsed, texts))
        row["sample_full"] = texts[0][:500]
        stage2.append(row)
        print(f"{r['label']:22s} persona={row['judge_persona_adherence']:.2f} "
              f"content={row['judge_content_relevance']:.2f} fluency={row['judge_fluency']:.2f} "
              f"composite={row['judge_composite']:.2f} collapsed={row['collapsed_full']}"
              f"/{len(s2_prompts)} | {client.cost.summary()}", flush=True)

    pick = max(stage2, key=lambda r: (r["judge_composite"], -r["alpha"]))
    out = {"stage1_prompts": s1_prompts, "stage2_prompts": s2_prompts,
           "stage1": stage1, "stage2": stage2, "pick": pick}
    os.makedirs(args.vectors, exist_ok=True)
    path = os.path.join(args.vectors, "sweep_results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nSweep time {time.time() - t0:.0f}s -> {path}")
    print(f"PICK: {pick['label']} layers={pick['layers']} alpha={pick['alpha']} "
          f"preserve_norm={pick['preserve_norm']} composite={pick['judge_composite']:.2f}")
    print(f"Sample: {pick['sample_full'][:300]}")


if __name__ == "__main__":
    main()
