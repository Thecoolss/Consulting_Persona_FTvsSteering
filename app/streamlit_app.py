"""The Consultant-ifier — interactive Streamlit demo.

Four tabs: side-by-side chat, evaluation results, an example gallery, and
methodology. Defaults to PRE-GENERATED responses (loaded from outputs/ and
evaluation/results/) so the demo never requires a live GPU. Live inference is
opt-in and only attempted if torch + a GPU are present.

Run:
    streamlit run app/streamlit_app.py --server.port 8501
"""

from __future__ import annotations

import glob
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

st.set_page_config(page_title="The Consultant-ifier", layout="wide")

NAVY = "#1f3a5f"
GOLD = "#c8a24b"

st.markdown(
    f"""<style>
    .stApp {{ background-color: #0e1117; }}
    h1, h2, h3 {{ color: {GOLD}; }}
    .method-box {{ border: 1px solid {NAVY}; border-radius: 8px; padding: 12px; }}
    </style>""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data loading (pre-generated fallback)
# --------------------------------------------------------------------------
@st.cache_data
def load_outputs() -> dict:
    data = {}
    for path in glob.glob(os.path.join(config.OUTPUT_DIR, "*_responses.json")):
        with open(path, encoding="utf-8") as fh:
            for rec in json.load(fh):
                data.setdefault(rec["method"], {}).setdefault(rec["prompt_id"], []).append(rec)
    return data


@st.cache_data
def load_eval() -> list[dict]:
    path = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return []


@st.cache_data
def load_test_prompts() -> list[dict]:
    if os.path.exists(config.TEST_PROMPTS_PATH):
        with open(config.TEST_PROMPTS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return []


def gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------
st.title("The Consultant-ifier")
st.caption("Three Ways to Make an LLM Say 'Let's Circle Back'")

outputs = load_outputs()
eval_results = load_eval()
prompts = load_test_prompts()

tab_chat, tab_eval, tab_gallery, tab_method = st.tabs(
    ["💬 Interactive Chat", "📊 Evaluation Results", "🖼️ Example Gallery", "📐 Methodology"]
)

# --------------------------------------------------------------------------
# Tab 1 — Chat
# --------------------------------------------------------------------------
with tab_chat:
    live = gpu_available()
    mode = "Live GPU inference" if live else "Pre-generated responses (no GPU detected)"
    st.info(f"Mode: {mode}")

    if prompts:
        labels = [f"[{p['category']}] {p['prompt']}" for p in prompts]
        choice = st.selectbox("Pick one of the evaluation prompts:", range(len(labels)),
                              format_func=lambda i: labels[i])
        pid = prompts[choice]["id"]
    else:
        st.warning("No test prompts found. Run data/make_test_prompts.py.")
        pid = None

    show_methods = ["prompting", "finetuning", "steering"]
    cols = st.columns(len(show_methods))
    for col, method in zip(cols, show_methods):
        with col:
            st.subheader(method.title())
            recs = outputs.get(method, {}).get(pid, []) if pid is not None else []
            if not recs:
                st.write("_No pre-generated response for this prompt._")
                continue
            rec = recs[0]
            st.markdown(f"<div class='method-box'>{rec['response']}</div>",
                        unsafe_allow_html=True)
            st.caption(f"variant: {rec['variant']} · "
                       f"{rec.get('generation_time_seconds', 0):.1f}s")

# --------------------------------------------------------------------------
# Tab 2 — Evaluation
# --------------------------------------------------------------------------
with tab_eval:
    if not eval_results:
        st.warning("No evaluation results yet. Run evaluation/run_evaluation.py "
                   "and analyze_results.py.")
    else:
        import pandas as pd
        rows = []
        for r in eval_results:
            for dim, sc in r["scores"].items():
                if sc.get("score") is not None:
                    rows.append({"method": r["method"], "dimension": dim,
                                 "category": r.get("category", ""),
                                 "score": sc["score"], "prompt": r["prompt"]})
        df = pd.DataFrame(rows)

        radar_path = os.path.join(config.RESULTS_DIR, "radar.png")
        if os.path.exists(radar_path):
            st.image(radar_path, caption="Mean judge scores by method")

        st.subheader("Mean score by method × dimension")
        st.dataframe(df.pivot_table(index="method", columns="dimension",
                                    values="score", aggfunc="mean").round(2))

        cats = ["(all)"] + sorted(df["category"].unique())
        pick = st.selectbox("Filter by category:", cats)
        view = df if pick == "(all)" else df[df["category"] == pick]
        st.dataframe(view.sort_values(["method", "dimension"]).reset_index(drop=True))

# --------------------------------------------------------------------------
# Tab 3 — Gallery
# --------------------------------------------------------------------------
with tab_gallery:
    if not eval_results:
        st.warning("Run the evaluation to populate the gallery.")
    else:
        by_prompt: dict = {}
        for r in eval_results:
            by_prompt.setdefault(r["prompt_id"], {})[r["method"]] = r
        st.write("Side-by-side responses with the judge's persona-adherence note.")
        for pid in list(by_prompt.keys())[:10]:
            group = by_prompt[pid]
            any_rec = next(iter(group.values()))
            with st.expander(any_rec["prompt"]):
                for method, rec in group.items():
                    pa = rec["scores"].get("persona_adherence", {})
                    st.markdown(f"**{method}** — persona {pa.get('score', '?')}/5")
                    st.write(rec["response"])
                    if pa.get("justification"):
                        st.caption("Judge: " + pa["justification"])

# --------------------------------------------------------------------------
# Tab 4 — Methodology
# --------------------------------------------------------------------------
with tab_method:
    st.subheader("How each method works")
    st.markdown(
        "- **Baseline** — raw Llama 3.1 8B, no persona (control).\n"
        "- **Prompting** — persona via system prompt (minimal / medium / detailed).\n"
        "- **Fine-tuning** — QLoRA adapter trained on 636 synthetic persona pairs.\n"
        "- **Steering** — a contrastive activation vector added to the residual "
        "stream at inference."
    )
    st.subheader("Hyperparameters")
    st.json({
        "base_model": config.BASE_MODEL,
        "lora_r": config.LORA_R, "lora_alpha": config.LORA_ALPHA,
        "learning_rate": config.LEARNING_RATE, "epochs": config.NUM_EPOCHS,
        "quantization": f"{config.QUANT_BITS}-bit {config.QUANT_TYPE}",
        "steering_layers": config.STEERING_LAYER_RANGE,
        "steering_capture": config.STEERING_CAPTURE,
        "decoding": "greedy (temperature=0)",
    })
    st.subheader("AWS services")
    st.markdown(
        "- **EC2** g5.xlarge — GPU compute\n"
        "- **S3** — artifact storage\n"
        "- **Bedrock** — Haiku 4.5 (data gen), Sonnet 4.5 (judge)\n"
        "- **CloudWatch** — training metrics"
    )
    summary_md = os.path.join(config.RESULTS_DIR, "analysis_summary.md")
    if os.path.exists(summary_md):
        with open(summary_md, encoding="utf-8") as fh:
            st.markdown(fh.read())
