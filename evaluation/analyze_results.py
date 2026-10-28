"""Phase 5 analysis — aggregate judge scores, run statistics, make charts.

For each method x dimension: mean, std, 95% CI. Pairwise method comparisons use
the paired Wilcoxon signed-rank test (all methods answer the SAME 50 prompts),
with rank-biserial effect size and Holm-Bonferroni correction across the family
of comparisons. Produces a radar chart, per-dimension bar charts, a markdown
summary, and a cost/effort comparison table.

Usage:
    python evaluation/analyze_results.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def load_results() -> list[dict]:
    path = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run run_evaluation.py first.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def method_key(rec: dict) -> str:
    """Collapse to a comparable method label.

    Prompting keeps its best/named variant; steering may carry layer/scale in
    the variant. For the headline comparison we group by `method`, but keep a
    detailed per-variant breakdown too.
    """
    return rec["method"]


def build_frame(results: list[dict]):
    import pandas as pd
    rows = []
    for r in results:
        for dim, sc in r["scores"].items():
            if sc.get("score") is None:
                continue
            rows.append({
                "method": r["method"],
                "variant": r["variant"],
                "prompt_id": r["prompt_id"],
                "category": r.get("category", ""),
                "dimension": dim,
                "score": sc["score"],
            })
    return pd.DataFrame(rows)


def summary_table(df):
    import numpy as np
    import pandas as pd

    def ci95(x):
        x = np.asarray(x, dtype=float)
        if len(x) < 2:
            return 0.0
        return 1.96 * x.std(ddof=1) / np.sqrt(len(x))

    agg = df.groupby(["method", "dimension"])["score"].agg(
        mean="mean", std="std", n="count")
    agg["ci95"] = df.groupby(["method", "dimension"])["score"].apply(
        lambda s: ci95(s.values))
    return agg.reset_index()


def paired_tests(df):
    """Wilcoxon signed-rank per dimension for each method pair, Holm-corrected."""
    from scipy.stats import wilcoxon
    import numpy as np

    methods = sorted(df["method"].unique())
    dims = [d for d in config.EVAL_DIMENSIONS if d in df["dimension"].unique()]
    raw = []

    for dim in dims:
        sub = df[df["dimension"] == dim]
        # pivot to prompt_id x method mean score (handles variant duplicates)
        pivot = sub.pivot_table(index="prompt_id", columns="method",
                                values="score", aggfunc="mean")
        for a, b in itertools.combinations(methods, 2):
            if a not in pivot or b not in pivot:
                continue
            pair = pivot[[a, b]].dropna()
            if len(pair) < 5:
                continue
            x, y = pair[a].values, pair[b].values
            diff = x - y
            if np.all(diff == 0):
                p, stat = 1.0, 0.0
            else:
                try:
                    stat, p = wilcoxon(x, y, zero_method="wilcox")
                except ValueError:
                    stat, p = 0.0, 1.0
            # rank-biserial effect size
            n_pos = int(np.sum(diff > 0))
            n_neg = int(np.sum(diff < 0))
            denom = n_pos + n_neg
            rbc = (n_pos - n_neg) / denom if denom else 0.0
            raw.append({"dimension": dim, "a": a, "b": b,
                        "mean_a": float(np.mean(x)), "mean_b": float(np.mean(y)),
                        "p_raw": float(p), "effect_rbc": float(rbc)})

    # Holm-Bonferroni across the whole family
    raw.sort(key=lambda r: r["p_raw"])
    m = len(raw)
    for i, r in enumerate(raw):
        r["p_holm"] = min(1.0, r["p_raw"] * (m - i))
        r["significant"] = r["p_holm"] < 0.05
    return raw


def radar_chart(summary, out_path):
    import matplotlib.pyplot as plt
    import numpy as np

    dims = [d for d in config.EVAL_DIMENSIONS
            if d in summary["dimension"].unique()]
    methods = sorted(summary["method"].unique())
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for method in methods:
        vals = []
        for dim in dims:
            row = summary[(summary["method"] == method) &
                          (summary["dimension"] == dim)]
            vals.append(float(row["mean"].iloc[0]) if len(row) else 0.0)
        vals += vals[:1]
        ax.plot(angles, vals, label=method, linewidth=2)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=9)
    ax.set_ylim(0, 5)
    ax.set_title("Persona methods — mean judge scores", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def bar_charts(summary, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    dims = [d for d in config.EVAL_DIMENSIONS
            if d in summary["dimension"].unique()]
    methods = sorted(summary["method"].unique())
    for dim in dims:
        means, errs = [], []
        for method in methods:
            row = summary[(summary["method"] == method) &
                          (summary["dimension"] == dim)]
            means.append(float(row["mean"].iloc[0]) if len(row) else 0.0)
            errs.append(float(row["ci95"].iloc[0]) if len(row) else 0.0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(methods, means, yerr=errs, capsize=5, color="#1f3a5f")
        ax.set_ylim(0, 5)
        ax.set_title(f"{dim.replace('_', ' ').title()} (mean ± 95% CI)")
        ax.set_ylabel("score (1-5)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"bar_{dim}.png"), dpi=150)
        plt.close(fig)


def effort_table() -> str:
    return (
        "| Method       | Data Prep        | Compute Cost     | Inference Overhead | Setup Complexity |\n"
        "|--------------|------------------|------------------|--------------------|------------------|\n"
        "| Baseline     | None             | ~$0              | None               | Trivial          |\n"
        "| Prompting    | None             | ~$0              | Longer context     | Trivial          |\n"
        "| Fine-tuning  | 636 gen. pairs   | ~0.35 GPU-h train| Same as base       | High             |\n"
        "| Steering     | 247 gen. pairs   | ~0.3 GPU-h ext+sweep | Minimal (one add) | Medium         |\n"
    )


def main() -> None:
    import pandas as pd

    results = load_results()
    df = build_frame(results)
    if df.empty:
        sys.exit("No valid scores to analyze.")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary = summary_table(df)
    tests = paired_tests(df)

    # overall ranking
    overall = df.groupby("method")["score"].mean().sort_values(ascending=False)
    per_dim_best = (summary.sort_values("mean", ascending=False)
                    .groupby("dimension").first().reset_index())

    # ---- console + markdown ----
    lines = ["# Evaluation Results\n", "## Mean scores by method x dimension\n"]
    pivot = summary.pivot(index="method", columns="dimension", values="mean").round(2)
    lines.append(pivot.to_markdown() + "\n")

    lines.append("## Overall ranking (mean across all dimensions)\n")
    for i, (method, score) in enumerate(overall.items(), 1):
        lines.append(f"{i}. **{method}** — {score:.2f}")
    lines.append("")

    lines.append("## Best method per dimension\n")
    for _, row in per_dim_best.iterrows():
        lines.append(f"- {row['dimension']}: **{row['method']}** ({row['mean']:.2f})")
    lines.append("")

    lines.append("## Pairwise significance (Wilcoxon, Holm-corrected)\n")
    lines.append("| Dimension | A | B | mean A | mean B | p (Holm) | effect (rbc) | sig |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in tests:
        lines.append(
            f"| {t['dimension']} | {t['a']} | {t['b']} | {t['mean_a']:.2f} | "
            f"{t['mean_b']:.2f} | {t['p_holm']:.4f} | {t['effect_rbc']:+.2f} | "
            f"{'yes' if t['significant'] else 'no'} |")
    lines.append("")

    lines.append("## Cost / effort comparison\n")
    lines.append(effort_table())

    report = "\n".join(lines)
    report_path = os.path.join(config.RESULTS_DIR, "analysis_summary.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    summary.to_csv(os.path.join(config.RESULTS_DIR, "summary_scores.csv"), index=False)
    with open(os.path.join(config.RESULTS_DIR, "pairwise_tests.json"), "w") as fh:
        json.dump(tests, fh, indent=2)

    radar_chart(summary, os.path.join(config.RESULTS_DIR, "radar.png"))
    bar_charts(summary, config.RESULTS_DIR)

    print(report)
    print(f"\nSaved: {report_path}, radar.png, bar_*.png, summary_scores.csv, "
          f"pairwise_tests.json in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
