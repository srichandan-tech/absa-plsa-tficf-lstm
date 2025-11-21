# src/app_streamlit.py
# -----------------------------------------------------------
# Streamlit UI for ABSA: PLSA + TF-ICF + LSTM
# - Single-text & batch CSV inference
# - Per-aspect confidence plotting
# - Optional training flow
# - Evaluate tab (explore + metrics)
# - Multi-aspect mode in Single Text tab (AUTO or Top-K)
# - HARD CHECKS + robust model discovery (mono or per-aspect)
# - Always shows full table of ALL six aspects
# - NEW: Overall verdict + natural-language summary
# - NEW: ENGINE_VERSION used to bust Streamlit cache after code changes
# -----------------------------------------------------------

# ---- Ensure project root is importable BEFORE any src imports ----
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # <project_root>
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -----------------------------------------------------------------

# ---- Suppress TF & absl noise (must be BEFORE TF imports) ----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity("error")
except Exception:
    pass

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# (optional) hide most TF python warnings
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass

# ----- Import config early so we can validate resources before loading engine -----
try:
    from src import config
except ModuleNotFoundError as e:
    st.error(
        f"Failed to import project modules.\n\n{e}\n\n"
        "How to run:\n"
        f"  conda activate absa310\n"
        f"  export PYTHONPATH={PROJECT_ROOT}\n"
        f"  python -m streamlit run src/app_streamlit.py"
    )
    st.stop()

# Absolute imports now that sys.path is guarded
try:
    from src.pipeline_infer import InferenceEngine
    from src.pipeline_train import main as train_main
except Exception as e:
    st.error(f"Import error while loading inference/train pipeline: {e}")
    st.stop()

# -------------------- Page setup --------------------
st.set_page_config(page_title="ABSA — PLSA + TF-ICF + LSTM", layout="wide")
st.title("Aspect-Based Sentiment Analysis (PLSA + TF-ICF + LSTM)")

# -------------------- Engine cache version (bust cache on change) --------------------
ENGINE_VERSION = "v2"  # bump this string whenever you update inference code

# -------------------- Environment / resource summary --------------------
def _has_mono_model() -> bool:
    return os.path.exists(os.path.join(config.MODELS_DIR, "lstm", "sentiment_lstm.h5"))

def _per_aspect_missing():
    lstm_dir = os.path.join(config.MODELS_DIR, "lstm")
    missing = []
    if not os.path.isdir(lstm_dir):
        return config.ASPECTS[:]  # dir missing => all missing
    for a in config.ASPECTS:
        c1 = os.path.join(lstm_dir, f"{a}.h5")
        c2 = os.path.join(lstm_dir, f"{a.replace(' ', '_')}.h5")
        if not (os.path.exists(c1) or os.path.exists(c2)):
            missing.append(a)
    return missing

with st.expander("Environment & resources (debug)"):
    st.write("**Python executable**:", sys.executable)
    st.write("**PYTHONPATH contains project root**:", PROJECT_ROOT in sys.path)
    st.write("**Models dir**:", getattr(config, "MODELS_DIR", "(unset)"))
    st.write("**Tokenizer exists**:", os.path.exists(os.path.join(config.MODELS_DIR, "tokenizer.pkl")))
    st.write("**GloVe path**:", getattr(config, "GLOVE_PATH", "(unset)"))
    st.write("**GloVe exists**:", os.path.exists(getattr(config, "GLOVE_PATH", "")))

    mono = _has_mono_model()
    missing = _per_aspect_missing()
    st.write("**LSTM model (mono)**:", bool(mono))
    st.write("**Per-aspect models present**:", len(missing) == 0)
    if missing:
        st.write("Missing aspects:", missing)

# -------------------- Sidebar --------------------
st.sidebar.header("Settings")

# Sentiment thresholds (probability of positive)
st.sidebar.subheader("Sentiment thresholds")
_default_pos = float(getattr(config, "POS_PROB_THRESHOLD", 0.70))
_default_neg = float(getattr(config, "NEG_PROB_THRESHOLD", 0.30))
pos_thr = st.sidebar.slider("Positive if prob ≥", 0.50, 0.90, _default_pos, 0.01)
neg_thr = st.sidebar.slider("Negative if prob ≤", 0.10, 0.50, _default_neg, 0.01)
if neg_thr >= pos_thr:
    st.sidebar.warning("Tip: set Negative threshold < Positive threshold to create a neutral band.")

st.sidebar.divider()
st.sidebar.header("Data & Training (optional)")
up_train = st.sidebar.file_uploader(
    "Upload CSV for training (columns: review_text, [aspect], [sentiment])",
    type=["csv"],
    help="If 'aspect'/'sentiment' are missing, the pipeline can still run (it can auto-label sentiment heuristically).",
)
if st.sidebar.button("Run Training"):
    if not up_train:
        st.warning("Please upload a CSV first.")
    else:
        raw_dir = os.path.join(config.DATA_DIR, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        in_path = os.path.join(raw_dir, "_uploaded.csv")
        with open(in_path, "wb") as f:
            f.write(up_train.getbuffer())
        with st.spinner("Training… (PLSA → TF-ICF → Aspect mapping → LSTM per aspect)"):
            train_main(in_path)
        st.success("Training complete. Models saved under ./models")

st.sidebar.divider()
show_clean = st.sidebar.checkbox("Show cleaned text", value=False)

# -------------------- Helpers --------------------
def prob_to_label(p: float, pos_cut: float, neg_cut: float) -> str:
    if p >= pos_cut:
        return "Positive"
    if p <= neg_cut:
        return "Negative"
    return "Neutral"

def tri_scores(p: float, pos_cut: float, neg_cut: float) -> tuple[float, float, float, str]:
    """
    Convert a positive probability to (pos, neu, neg, leans_to).
    - pos = p
    - neg = 1 - p
    - neu peaks at the center of the neutral band, 0 at the edges/outside.
    """
    p = float(p)
    pos = p
    neg = 1.0 - p
    if p <= neg_cut or p >= pos_cut:
        neu = 0.0
    else:
        band = max(1e-9, pos_cut - neg_cut)
        center = (pos_cut + neg_cut) / 2.0
        neu = max(0.0, 1.0 - abs(p - center) / (band / 2.0))
    scores = {"Positive": pos, "Neutral": neu, "Negative": neg}
    leans_to = max(scores.items(), key=lambda kv: kv[1])[0]
    return round(pos, 3), round(neu, 3), round(neg, 3), leans_to

def ensure_review_text(df: pd.DataFrame) -> pd.DataFrame:
    if "review_text" in df.columns:
        return df
    for cand in df.columns:
        if str(cand).lower() in ("review", "text", "ulasan", "content", "comment", "review text"):
            df = df.copy()
            df.insert(0, "review_text", df[cand].astype(str))
            return df
    raise ValueError("No 'review_text' column found and no suitable fallback (e.g., 'review').")

def plot_aspect_confidence(aspect_scores: dict):
    if not aspect_scores:
        st.info("No per-aspect scores available to plot.")
        return
    df_plot = (
        pd.DataFrame({"aspect": list(aspect_scores.keys()), "score": list(aspect_scores.values())})
        .sort_values("score", ascending=False)
    )
    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X("score:Q", title="Confidence (0–1)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("aspect:N", sort="-x", title="Aspect"),
            tooltip=[alt.Tooltip("aspect:N"), alt.Tooltip("score:Q", format=".3f")],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)

def confusion_heatmap(cm: np.ndarray, labels: list, title: str):
    df_cm = (
        pd.DataFrame(cm, index=pd.Index(labels, name="True"), columns=pd.Index(labels, name="Pred"))
        .reset_index()
        .melt(id_vars="True", var_name="Pred", value_name="count")
    )
    base = alt.Chart(df_cm).mark_rect().encode(
        x=alt.X("Pred:N", sort=labels),
        y=alt.Y("True:N", sort=labels),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["True:N", "Pred:N", "count:Q"],
    ).properties(title=title, height=220)
    text = alt.Chart(df_cm).mark_text(baseline="middle").encode(
        x="Pred:N", y="True:N", text="count:Q"
    )
    st.altair_chart(base + text, use_container_width=True)

def normalize_sentiment_col(s):
    """Map various ground truth formats to {Negative, Neutral, Positive}."""
    if pd.isna(s):
        return None
    val = str(s).strip().lower()
    if val in {"1", "pos", "positive"}:
        return "Positive"
    if val in {"0", "neg", "negative"}:
        return "Negative"
    if val in {"neutral", "neu", "2"}:
        return "Neutral"
    return s

# ---------- NEW: overall verdict & summary helpers ----------
def _weighted_overall(rows: list[dict]) -> dict:
    """
    Given predict_all rows, compute an overall direction.
    Weight by similarity if available; otherwise uniform.
    Returns dict with: pos, neg, neu, verdict, drivers_pos, drivers_neg
    """
    if not rows:
        return {
            "pos": 0.5, "neg": 0.5, "neu": 0.0,
            "verdict": "Neutral",
            "drivers_pos": [], "drivers_neg": [],
        }

    sims = np.array([float(r.get("sim", 0.0)) for r in rows], dtype=float)
    sims = sims / sims.sum() if sims.sum() > 1e-9 else np.ones_like(sims) / max(1, len(rows))

    pos = float(np.sum([sims[i] * float(rows[i]["prob_positive"]) for i in range(len(rows))]))
    neg = float(np.sum([sims[i] * float(rows[i]["prob_negative"]) for i in range(len(rows))]))
    neu = max(0.0, 1.0 - abs(pos - 0.5) * 2.0)

    # drivers: sort by weighted contribution
    contrib_pos = sorted(
        [(r["aspect"], sims[i] * float(r["prob_positive"])) for i, r in enumerate(rows)],
        key=lambda x: x[1], reverse=True
    )
    contrib_neg = sorted(
        [(r["aspect"], sims[i] * float(r["prob_negative"])) for i, r in enumerate(rows)],
        key=lambda x: x[1], reverse=True
    )
    verdict = "Positive" if pos >= 0.5 else ("Negative" if neg > pos else "Neutral")
    return {
        "pos": pos, "neg": neg, "neu": neu, "verdict": verdict,
        "drivers_pos": [a for a, _ in contrib_pos[:3]],
        "drivers_neg": [a for a, _ in contrib_neg[:3]],
    }

def _summary_sentence(rows: list[dict], overall: dict) -> str:
    """
    Build a friendly one-liner summarizing the review direction and main drivers.
    """
    v = overall["verdict"]
    pos_as = [r["aspect"] for r in rows if r.get("sentiment_label") == "Positive"]
    neg_as = [r["aspect"] for r in rows if r.get("sentiment_label") == "Negative"]

    def fmt_list(xs):
        if not xs: return ""
        if len(xs) == 1: return xs[0]
        return ", ".join(xs[:-1]) + f", and {xs[-1]}"

    if v == "Positive":
        lead = "Overall this review leans **Positive**."
    elif v == "Negative":
        lead = "Overall this review leans **Negative**."
    else:
        lead = "Overall this review is **Neutral/mixed**."

    bits = []
    if pos_as:
        bits.append(f"strengths in {fmt_list(pos_as)}")
    if neg_as:
        bits.append(f"concerns around {fmt_list(neg_as)}")

    tail = ""
    if bits:
        tail = " Key takeaways: " + "; ".join(bits) + "."

    return lead + tail

# -------------------- Engine (with robust model discovery) --------------------
@st.cache_resource
def load_engine(_engine_version: str = ENGINE_VERSION):
    """
    The _engine_version parameter is included purely to change the cache key
    whenever ENGINE_VERSION changes. Bump ENGINE_VERSION to force a reload.
    """
    tok_path  = os.path.join(config.MODELS_DIR, "tokenizer.pkl")
    lstm_dir  = os.path.join(config.MODELS_DIR, "lstm")
    asp_vocab = os.path.join(config.MODELS_DIR, "aspect_vocab.json")

    assert os.path.exists(tok_path),  f"Tokenizer not found: {tok_path}"
    assert os.path.exists(config.GLOVE_PATH), f"GloVe not found: {config.GLOVE_PATH}"
    assert os.path.isdir(lstm_dir), f"LSTM dir not found: {lstm_dir}"
    assert os.path.exists(asp_vocab), f"Aspect vocab not found: {asp_vocab}"

    # Accept either mono model or per-aspect models
    mono_path = os.path.join(lstm_dir, "sentiment_lstm.h5")
    has_mono  = os.path.exists(mono_path)

    def _aspect_paths(a: str):
        return [
            os.path.join(lstm_dir, f"{a}.h5"),
            os.path.join(lstm_dir, f"{a.replace(' ', '_')}.h5"),
        ]

    missing = []
    if not has_mono:
        for a in config.ASPECTS:
            if not any(os.path.exists(p) for p in _aspect_paths(a)):
                missing.append(a)

    if not has_mono and missing:
        raise AssertionError(
            "LSTM model(s) not found.\n"
            f"- Looked for monolithic model: {mono_path}\n"
            f"- Or per-aspect models (*.h5) in: {lstm_dir}\n"
            f"- Missing aspects: {missing}"
        )

    try:
        eng = InferenceEngine(
            model_dir=lstm_dir,               # engine decides mono vs per-aspect
            tokenizer_path=tok_path,
            aspect_vocab_path=asp_vocab,
            threshold=0.5,
        )
        return eng
    except Exception as e:
        st.error(f"Failed to initialize inference engine:\n{e}")
        raise

ie = load_engine()

# -------------------- Tabs --------------------
tab_single, tab_batch, tab_eval = st.tabs(["🔎 Single Text", "🧾 Batch CSV", "📊 Evaluate"])

# ==================== SINGLE (manual input) ====================
with tab_single:
    st.subheader("Analyze one review")
    example = "The location is perfect but the room was dirty and the staff were not helpful."
    txt = st.text_area("Enter a review:", example, height=140)

    # Multi-aspect controls
    multi = st.checkbox("Multi-aspect mode (Top-K by similarity)", value=True)
    auto_k = True
    k = 3
    min_conf = 0.15
    if multi:
        auto_k = st.checkbox("Auto-detect aspects (by min confidence)", value=True)
        if auto_k:
            min_conf = float(st.slider("Min confidence (normalized)", 0.0, 1.0, 0.15, 0.01))
            k = 0  # 0 → AUTO (no cap; engine will select any aspects that pass min_conf)
        else:
            # Let user pick up to 6 (your aspect set size)
            k = int(st.slider("Pick Top-K aspects", min_value=1, max_value=6, value=3, step=1))
            min_conf = float(st.slider("Min confidence (normalized)", 0.0, 1.0, 0.15, 0.01))

    show_full_table = st.checkbox("Show full table of ALL six aspects", value=True)

    if "single_res" not in st.session_state:
        st.session_state["single_res"] = None
    if "all_res" not in st.session_state:
        st.session_state["all_res"] = None
    if "overall" not in st.session_state:
        st.session_state["overall"] = None

    # Trigger analysis
    colL, colR = st.columns([1, 2], gap="large")

    with colL:
        if st.button("Analyze", type="primary"):
            if not txt.strip():
                st.warning("Please enter some text.")
            else:
                # --- Single or Multi predictions ---
                if multi:
                    res = ie.predict_multi(txt, aspect_top_k=k, aspect_min_conf=min_conf)
                    rows = []
                    for r in res["aspects"]:
                        ppos = float(r["prob"])
                        p_pos, p_neu, p_neg, leans = tri_scores(ppos, pos_thr, neg_thr)
                        rows.append({
                            "aspect": r["aspect"],
                            "sentiment_label": prob_to_label(ppos, pos_thr, neg_thr),
                            "prob_positive": p_pos,
                            "prob_neutral": p_neu,
                            "prob_negative": p_neg,
                            "leans_to": leans,
                        })
                    st.session_state["single_res"] = {
                        "mode": "multi",
                        "k": k,
                        "auto": auto_k,
                        "min_conf": min_conf,
                        "data": rows,
                        "aspect_scores": res.get("aspect_scores", {}),
                        "cleaned_text": res.get("cleaned_text", "")
                    }
                else:
                    res = ie.predict(txt)
                    ppos = float(res.get("prob", 0.5))
                    p_pos, p_neu, p_neg, leans = tri_scores(ppos, pos_thr, neg_thr)
                    st.metric("Aspect", res.get("aspect", "-"))
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Positive", f"{p_pos:.3f}")
                    c2.metric("Neutral", f"{p_neu:.3f}")
                    c3.metric("Negative", f"{p_neg:.3f}")
                    c4.metric("Leans to", leans)
                    if show_clean and "cleaned_text" in res:
                        st.caption("Cleaned text")
                        st.write(res["cleaned_text"])
                    st.session_state["single_res"] = {
                        "mode": "single",
                        "data": {
                            "aspect": res.get("aspect"),
                            "sentiment_label": prob_to_label(ppos, pos_thr, neg_thr),
                            "prob_positive": p_pos,
                            "prob_neutral": p_neu,
                            "prob_negative": p_neg,
                            "leans_to": leans,
                        },
                        "aspect_scores": res.get("aspect_scores", {}),
                        "cleaned_text": res.get("cleaned_text", "")
                    }

                # --- Always call the all-aspects API so the 6-aspect table is ready ---
                res_all = ie.predict_all(txt, pos_thr=pos_thr, neg_thr=neg_thr)
                st.session_state["all_res"] = res_all

                # --- NEW: overall verdict + summary ---
                overall = _weighted_overall(res_all.get("aspects", []))
                st.session_state["overall"] = overall

    with colR:
        st.caption("Per-aspect confidence")
        # Prefer the normalized similarity scores from the full-aspects call (if available)
        sess_all = st.session_state.get("all_res")
        if sess_all and "aspect_scores" in sess_all:
            plot_aspect_confidence(sess_all.get("aspect_scores", {}))
        else:
            sess = st.session_state.get("single_res")
            if sess:
                plot_aspect_confidence(sess.get("aspect_scores", {}))
            else:
                st.info("Run analysis first to see per-aspect confidence.")

    # ---------- Overall verdict panel ----------
    overall = st.session_state.get("overall")
    if overall:
        verdict = overall["verdict"]
        rows_all = (st.session_state.get("all_res") or {}).get("aspects", [])
        summary = _summary_sentence(rows_all, overall)

        if verdict == "Positive":
            st.success(f"Overall verdict: **{verdict}** — weighted pos={overall['pos']:.3f}, neg={overall['neg']:.3f}.")
        elif verdict == "Negative":
            st.error(f"Overall verdict: **{verdict}** — weighted pos={overall['pos']:.3f}, neg={overall['neg']:.3f}.")
        else:
            st.info(f"Overall verdict: **{verdict}** — weighted pos={overall['pos']:.3f}, neg={overall['neg']:.3f}.")

        st.markdown(summary)
        if overall["drivers_pos"]:
            st.caption("Top positive drivers: " + ", ".join(overall["drivers_pos"]))
        if overall["drivers_neg"]:
            st.caption("Top negative drivers: " + ", ".join(overall["drivers_neg"]))

    # --- FULL-WIDTH TABLES ---
    # (A) Multi AUTO or Top-K table for quick glance
    sess = st.session_state.get("single_res")
    if sess and sess.get("mode") == "multi":
        k_used = sess.get("k", 0)
        title = "Per-aspect sentiment (Auto selection)" if k_used in (0, None) else f"Per-aspect sentiment (Top-{k_used} selection)"
        st.markdown(f"### {title}")
        df_show = pd.DataFrame(sess["data"])
        st.dataframe(
            df_show[["aspect","sentiment_label","prob_positive","prob_neutral","prob_negative","leans_to"]],
            use_container_width=True,
            height=min(100 + 38 * len(df_show), 360),
            column_config={
                "aspect": st.column_config.TextColumn("aspect", width="large"),
                "sentiment_label": st.column_config.TextColumn("sentiment_label", width="medium"),
                "prob_positive": st.column_config.NumberColumn("prob_positive", format="%.3f", width="small"),
                "prob_neutral": st.column_config.NumberColumn("prob_neutral", format="%.3f", width="small"),
                "prob_negative": st.column_config.NumberColumn("prob_negative", format="%.3f", width="small"),
                "leans_to": st.column_config.TextColumn("leans_to", width="small"),
            },
            hide_index=True,
        )
        if show_clean and sess.get("cleaned_text"):
            st.caption("Cleaned text")
            st.write(sess["cleaned_text"])

    # (B) Full table of ALL six aspects with conclusions
    if show_full_table:
        all_res = st.session_state.get("all_res")
        if all_res and isinstance(all_res.get("aspects"), list) and all_res["aspects"]:
            st.markdown("### All six aspects — tri-sentiment scores & conclusion")
            df_all = pd.DataFrame(all_res["aspects"])
            preferred = [
                "aspect", "prob_positive", "prob_neutral", "prob_negative",
                "leans_to", "sentiment", "conclusion", "sim"
            ]
            cols = [c for c in preferred if c in df_all.columns] + [c for c in df_all.columns if c not in preferred]
            df_all = df_all[cols]

            st.dataframe(
                df_all,
                use_container_width=True,
                height=min(100 + 38 * len(df_all), 520),
                column_config={
                    "aspect": st.column_config.TextColumn("aspect", width="large"),
                    "prob_positive": st.column_config.NumberColumn("positive", format="%.3f", width="small"),
                    "prob_neutral": st.column_config.NumberColumn("neutral", format="%.3f", width="small"),
                    "prob_negative": st.column_config.NumberColumn("negative", format="%.3f", width="small"),
                    "leans_to": st.column_config.TextColumn("leans_to", width="small"),
                    "sentiment": st.column_config.NumberColumn("binary_sentiment (1=pos)", width="small"),
                    "conclusion": st.column_config.TextColumn("conclusion", width="large"),
                    "sim": st.column_config.NumberColumn("sim(conf)", format="%.3f", width="small"),
                },
                hide_index=True,
            )
            if show_clean and all_res.get("cleaned_text"):
                st.caption("Cleaned text")
                st.write(all_res["cleaned_text"])
        else:
            st.info("Run analysis to see the full aspects table.")

# ==================== BATCH (show all reviews) ====================
with tab_batch:
    st.subheader("Batch CSV inference")
    up = st.file_uploader(
        "Upload a CSV with a 'review_text' column (fallbacks: review/text/ulasan/content/comment).",
        type=["csv"],
        key="batch",
    )
    limit = st.number_input("Limit rows (optional, 0 = all)", min_value=0, value=0, step=50)

    run = st.button("Run Batch Prediction", type="primary")
    if run:
        if not up:
            st.warning("Please upload a CSV.")
        else:
            try:
                df_in = pd.read_csv(up)
            except Exception as e:
                st.error(f"Failed reading CSV: {e}")
                df_in = None

            if df_in is not None:
                try:
                    df_in = ensure_review_text(df_in)
                except Exception as e:
                    st.error(str(e))
                    df_in = None

            if df_in is not None:
                if limit and limit > 0:
                    df_in = df_in.head(limit).copy()

                if hasattr(ie, "batch_predict"):
                    out_df = ie.batch_predict(df_in["review_text"].astype(str).tolist())
                    if not isinstance(out_df, pd.DataFrame):
                        out_df = pd.DataFrame(out_df)
                else:
                    rows = []
                    for t in df_in["review_text"].astype(str).tolist():
                        r = ie.predict(t)
                        rows.append(r)
                    out_df = pd.DataFrame(rows)

                if "prob" in out_df.columns:
                    out_df["sentiment_label"] = out_df["prob"].apply(
                        lambda p: prob_to_label(float(p), pos_thr, neg_thr)
                    )
                elif "sentiment_pred" in out_df.columns and "score" in out_df.columns:
                    out_df["sentiment_label"] = out_df["score"].apply(
                        lambda p: prob_to_label(float(p), pos_thr, neg_thr)
                    )
                else:
                    out_df["sentiment_label"] = "Neutral"

                if "review_text" not in out_df.columns:
                    out_df.insert(0, "review_text", df_in["review_text"].values)

                cols_pref = [c for c in [
                    "review_text", "aspect", "aspect_pred", "prob", "score",
                    "sentiment", "sentiment_pred", "sentiment_label", "cleaned_text"
                ] if c in out_df.columns]
                cols_rest = [c for c in out_df.columns if c not in cols_pref and not c.startswith("aspect_scores")]
                out_df = out_df[cols_pref + cols_rest]

                st.success(f"Predicted {len(out_df)} rows.")
                st.dataframe(out_df, use_container_width=True, height=480)

                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions as CSV",
                    data=csv_bytes,
                    file_name="reviews_predicted.csv",
                    mime="text/csv",
                )

                st.markdown("### Aggregates")
                c1, c2 = st.columns(2)
                with c1:
                    asp_col = "aspect" if "aspect" in out_df.columns else ("aspect_pred" if "aspect_pred" in out_df.columns else None)
                    if asp_col:
                        st.write("By Aspect")
                        st.bar_chart(out_df[asp_col].value_counts())
                with c2:
                    st.write("By Sentiment (3-class)")
                    st.bar_chart(out_df["sentiment_label"].value_counts())

                st.session_state["last_batch_df"] = out_df.copy()

# ==================== EVALUATE (Explore + metrics with ground-truth) ====================
with tab_eval:
    st.subheader("Evaluate predictions (Accuracy, F1, Confusion Matrix) + Explore")
    st.caption("Explore predictions without labels, or upload ground-truth to compute metrics.")

    src_col, _ = st.columns([1, 1])
    with src_col:
        use_last = st.checkbox("Use last batch predictions from this session", value=True)
    pred_file = None
    if not use_last:
        pred_file = st.file_uploader(
            "Upload predictions CSV (needs at least review_text + prob/sentiment_pred/sentiment_label)",
            type=["csv"], key="pred_eval"
        )
    gt_file = st.file_uploader(
        "Upload ground-truth CSV (optional — review_text, aspect, sentiment)",
        type=["csv"], key="gt_eval"
    )

    df_pred = None
    if use_last and "last_batch_df" in st.session_state:
        df_pred = st.session_state["last_batch_df"].copy()
    elif not use_last and pred_file is not None:
        try:
            df_pred = pd.read_csv(pred_file)
        except Exception as e:
            st.error(f"Failed reading predictions CSV: {e}")
            df_pred = None
    else:
        st.info("Load predictions via Batch CSV tab, or upload a predictions CSV above.")

    st.markdown("### 🔎 Explore predictions (no ground-truth required)")
    if df_pred is not None:
        try:
            df_pred = ensure_review_text(df_pred)
        except Exception as e:
            st.error(f"Predictions CSV: {e}")
            df_pred = None

    if df_pred is not None:
        if "sentiment_label" not in df_pred.columns:
            if "prob" in df_pred.columns:
                df_pred["sentiment_label"] = df_pred["prob"].apply(
                    lambda p: prob_to_label(float(p), pos_thr, neg_thr)
                )
            elif "sentiment_pred" in df_pred.columns:
                df_pred["sentiment_label"] = df_pred["sentiment_pred"].map({1: "Positive", 0: "Negative"}).fillna("Neutral")

        asp_col = "aspect_pred" if "aspect_pred" in df_pred.columns else ("aspect" if "aspect" in df_pred.columns else None)

        f1, f2, f3, f4 = st.columns([1.4, 1, 1, 1.2])
        with f1:
            kw = st.text_input("Search text (contains)", value="")
        with f2:
            opts_s = [x for x in ["Negative", "Neutral", "Positive"]
                      if "sentiment_label" in df_pred.columns and x in df_pred["sentiment_label"].unique()]
            sel_s = st.multiselect("Sentiment", options=opts_s, default=opts_s)
        with f3:
            if asp_col:
                opts_a = sorted(df_pred[asp_col].dropna().unique().tolist())
                sel_a = st.multiselect("Aspect", options=opts_a, default=opts_a)
            else:
                sel_a, opts_a = [], []
        with f4:
            prob_col = "prob" if "prob" in df_pred.columns else ("score" if "score" in df_pred.columns else None)
            min_prob = st.slider("Min prob(score)", 0.0, 1.0, 0.0, 0.01) if prob_col else 0.0

        df_view = df_pred.copy()
        if kw:
            df_view = df_view[df_view["review_text"].str.contains(kw, case=False, na=False)]
        if "sentiment_label" in df_view.columns and sel_s:
            df_view = df_view[df_view["sentiment_label"].isin(sel_s)]
        if asp_col and sel_a:
            df_view = df_view[df_view[asp_col].isin(sel_a)]
        if prob_col:
            df_view = df_view[df_view[prob_col].astype(float) >= float(min_prob)]

        st.write(f"Showing **{len(df_view)}** rows")
        show_cols = [c for c in ["review_text", asp_col, "sentiment_label", prob_col, "cleaned_text"] if c and c in df_view.columns]
        st.dataframe(df_view[show_cols], use_container_width=True, height=500)

        csv_bytes = df_view.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered table (CSV)", data=csv_bytes, file_name="predictions_filtered.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("### 📊 Evaluate (requires ground-truth with aspect & sentiment)")

    eval_btn = st.button("Run Evaluation", type="primary", key="run_eval_btn")
    if eval_btn:
        if df_pred is None:
            st.error("No predictions available. Load predictions first.")
            st.stop()
        if gt_file is None:
            st.error("Please upload a ground-truth CSV (review_text, aspect, sentiment).")
            st.stop()

        try:
            df_gt = pd.read_csv(gt_file)
        except Exception as e:
            st.error(f"Failed reading ground-truth CSV: {e}")
            st.stop()

        try:
            df_gt = ensure_review_text(df_gt)
        except Exception as e:
            st.error(f"Ground-truth CSV: {e}")
            st.stop()

        missing_cols = [c for c in ["aspect", "sentiment"] if c not in df_gt.columns]
        if missing_cols:
            st.error(f"Ground-truth is missing columns: {missing_cols}. Required: aspect, sentiment.")
            st.stop()

        df_gt["sentiment_norm"] = df_gt["sentiment"].apply(normalize_sentiment_col)

        if "sentiment_label" not in df_pred.columns:
            if "prob" in df_pred.columns:
                df_pred["sentiment_label"] = df_pred["prob"].apply(lambda p: prob_to_label(float(p), pos_thr, neg_thr))
            elif "sentiment_pred" in df_pred.columns:
                df_pred["sentiment_label"] = df_pred["sentiment_pred"].map({1: "Positive", 0: "Negative"})
            else:
                st.error("Predictions must contain 'sentiment_label', or 'prob', or 'sentiment_pred'.")
                st.stop()

        merged = df_gt.merge(df_pred, on="review_text", how="inner", suffixes=("_gt", "_pred"))
        if merged.empty:
            st.error("No rows matched by 'review_text'. Ensure both files use the exact same text.")
            st.stop()

        st.success(f"Merged {len(merged)} rows for evaluation.")
        st.dataframe(merged.head(50), use_container_width=True)

        asp_pred_col = "aspect_pred" if "aspect_pred" in merged.columns else None
        if asp_pred_col:
            aspect_labels = sorted(list(set(merged["aspect"].dropna()) | set(merged[asp_pred_col].dropna())))
            acc_aspect = accuracy_score(merged["aspect"], merged[asp_pred_col])
            st.metric("Aspect Accuracy", f"{acc_aspect:.3f}")
            cm_aspect = confusion_matrix(merged["aspect"], merged[asp_pred_col], labels=aspect_labels)
            confusion_heatmap(cm_aspect, aspect_labels, "Aspect Confusion Matrix")
        else:
            st.info("No 'aspect_pred' column in predictions — skipping aspect metrics.")

        y_true = merged["sentiment_norm"].astype(str)
        y_pred = merged["sentiment_label"].astype(str)
        label_order = [l for l in ["Negative", "Neutral", "Positive"] if l in set(y_true) | set(y_pred)]

        acc_sent = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", labels=label_order, zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=label_order, zero_division=0)
        st.metric("Sentiment Accuracy", f"{acc_sent:.3f}")
        st.metric("F1 (macro)", f"{f1_macro:.3f}")
        st.metric("F1 (weighted)", f"{f1_weighted:.3f}")

        cm_sent = confusion_matrix(y_true, y_pred, labels=label_order)
        confusion_heatmap(cm_sent, label_order, "Sentiment Confusion Matrix")

        st.markdown("**Classification report**")
        st.code(classification_report(y_true, y_pred, labels=label_order, zero_division=0))
