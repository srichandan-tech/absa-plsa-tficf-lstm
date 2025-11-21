# src/pipeline_train.py
# -----------------------------------------------------------
# ABSA training pipeline (PLSA + TF-ICF + per-aspect LSTM)
# - Builds aspect TF-ICF using PLSA topics + seed terms
# - Assigns aspects to documents
# - NEW: Weak-label rules for sentiment (fallback to TextBlob only if uncertain)
# - Trains one LSTM (positive vs. negative) per aspect
# - Handles class imbalance (class_weight)
# - EarlyStopping + ReduceLROnPlateau
# - Saves tokenizer and models (.keras and .h5 for compatibility)
#
# Usage:
#   conda activate absa310
#   cd /path/to/absa-plsa-lstm
#   PYTHONPATH=. python -m src.pipeline_train --csv data/raw/your.csv --epochs 12
# -----------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .config import (
    PLSA_TOPICS,
    PLSA_MAX_ITERS,
    RANDOM_STATE,
    ASPECTS,
    ASPECT_SEED_TERMS,
    NEGATION_CUES,
)
from .dataset import ReviewDataset
from .plsa import PLSA
from .tf_icf import tf_icf
from .semantic_similarity import aspect_assignment
from .lstm_model import AspectLSTM


# -----------------------------------------------------------
#  Safe helper for train/validation split
# -----------------------------------------------------------
def safe_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a stratified train/validation split if both classes
    have at least 2 samples; otherwise, fall back to non-stratified.
    """
    y_series = pd.Series(y)
    vc = y_series.value_counts(dropna=False)
    if len(vc) >= 2 and vc.min() >= 2:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    print(f"[WARN] Non-stratified split used — class counts: {dict(vc)}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# -----------------------------------------------------------
#  Weak-label rules for sentiment (before TextBlob fallback)
# -----------------------------------------------------------
_POS_LEX = {
    # core polarity words
    "good","great","excellent","amazing","awesome","fantastic","perfect","nice","clean",
    "friendly","helpful","cozy","comfy","welcoming","peaceful","beautiful","love","loved",
    "like","liked","enjoy","enjoyed","recommend","recommended","worth","affordable",
    # intensifiers handled via weighting, but keep a few as strong positives too
    "outstanding","superb","wonderful","brilliant",
}
_NEG_LEX = {
    "bad","terrible","awful","horrible","worst","dirty","filthy","smelly","smell","stain",
    "stained","broken","rude","unhelpful","noisy","noise","disgusting","poor","slow",
    "overpriced","expensive","cold","hot","hard","uncomfortable","issue","problem",
}

_INTENSIFIERS_POS = {"very","really","extremely","so","super","quite","highly","truly"}
_INTENSIFIERS_NEG = {"very","really","extremely","so","super","quite","truly","too"}  # "too noisy"

def _contains_star_score(text: str) -> Optional[int]:
    """Return 1 for likely 4–5/5 patterns, 0 for 1–2/5, else None."""
    t = text.replace(" ", "").lower()
    for pat in ("5/5","4/5","10/10","9/10"):
        if pat in t:
            return 1
    for pat in ("1/5","2/5","1/10","2/10","3/10"):
        if pat in t:
            return 0
    return None

def _apply_negation_window(tokens: list[str]) -> list[tuple[str,bool]]:
    """
    Mark tokens as negated if a negation cue appears within a small backward window.
    Returns list of (token, is_negated).
    """
    negated = []
    last_neg = -999
    for i, tok in enumerate(tokens):
        if tok in NEGATION_CUES:
            last_neg = i
        is_neg = (i - last_neg) <= 3  # 3-token window after a negation cue
        negated.append((tok, is_neg))
    return negated

def weak_label_sentiment(text: str) -> Optional[int]:
    """
    Rule-based weak labeling.
      • Uses polarity lexicons with negation handling + intensifiers
      • Captures numeric star-rating patterns
      • Handles simple 'but' contrast (right clause has priority)
    Returns:
      1 for positive, 0 for negative, None if unsure.
    """
    if not text or not isinstance(text, str):
        return None
    s = text.strip().lower()

    # Obvious numeric rating
    star = _contains_star_score(s)
    if star is not None:
        return star

    # Split on strong contrastive "but"
    # Right clause often expresses overall judgment.
    parts = [p.strip() for p in s.split(" but ") if p.strip()]
    if len(parts) >= 2:
        tail = parts[-1]
        lbl_tail = weak_label_sentiment(" ".join(tail.split()))  # recurse on last clause
        if lbl_tail is not None:
            return lbl_tail
        # if tail uncertain, fall back to full sentence later

    # Tokenize (simple whitespace; your dataset already normalized)
    toks = s.split()
    marked = _apply_negation_window(toks)

    score = 0.0
    for i, (tok, is_neg) in enumerate(marked):
        # base polarity
        if tok in _POS_LEX:
            w = 1.0
            # look for local intensifier
            if i > 0 and toks[i-1] in _INTENSIFIERS_POS:
                w += 0.5
            score += (-w if is_neg else w)
        elif tok in _NEG_LEX:
            w = 1.0
            if i > 0 and toks[i-1] in _INTENSIFIERS_NEG:
                w += 0.5
            score += (w if is_neg else -w)

    # Exclamation marks give a tiny push
    if "!" in s:
        score += 0.2 if score > 0 else (-0.2 if score < 0 else 0)

    # Decide only when confident enough
    if score >= 0.75:
        return 1
    if score <= -0.75:
        return 0
    return None


# -----------------------------------------------------------
#  Main ABSA pipeline
# -----------------------------------------------------------
def main(csv_path: str, epochs: int, min_samples: int, val_size: float) -> None:
    os.makedirs("models/lstm", exist_ok=True)
    os.makedirs("models/label_encoders", exist_ok=True)

    # ---------------- Load & tokenize ----------------
    ds = ReviewDataset(csv_path)
    docs_tokens = ds.tokenized_docs()

    # ---------------- PLSA topics ----------------
    plsa = PLSA(n_topics=PLSA_TOPICS, max_iter=PLSA_MAX_ITERS, random_state=RANDOM_STATE)
    plsa.fit(docs_tokens)

    topic_activ = plsa.transform(docs_tokens)
    doc_topic = topic_activ.argmax(axis=1)

    clusters = defaultdict(list)
    for k, toks in zip(doc_topic, docs_tokens):
        clusters[int(k)].append(toks)

    # ---------------- TF–ICF per topic ----------------
    tf_icf_scores = tf_icf(clusters)
    vocab = plsa.vocab
    V = len(vocab)
    topic_count = (max(tf_icf_scores.keys()) + 1) if tf_icf_scores else 0

    topic_mat = np.zeros((topic_count, V), dtype="float32")
    term_index = {t: i for i, t in enumerate(vocab)}
    for k, term_scores in tf_icf_scores.items():
        for t, w in term_scores.items():
            j = term_index.get(t)
            if j is not None:
                topic_mat[k, j] = float(w)

    # ---------------- Seeded aspect vectors ----------------
    aspect_mat = np.zeros((len(ASPECTS), V), dtype="float32")
    for a_idx, a in enumerate(ASPECTS):
        for t in ASPECT_SEED_TERMS.get(a, []):
            j = term_index.get(t)
            if j is not None:
                aspect_mat[a_idx, j] += 1.0

    # Normalize
    if aspect_mat.size:
        aspect_mat = aspect_mat / (np.linalg.norm(aspect_mat, axis=1, keepdims=True) + 1e-12)
    if topic_mat.size:
        topic_mat_n = topic_mat / (np.linalg.norm(topic_mat, axis=1, keepdims=True) + 1e-12)
    else:
        topic_mat_n = np.zeros_like(topic_mat)

    # Similarity aspect↔topic (A×K)
    sims = aspect_mat @ topic_mat_n.T if topic_count > 0 else np.zeros((len(ASPECTS), 0), dtype="float32")

    # ---------------- Merge topic TF–ICF → aspect TF–ICF ----------------
    aspect_tf_icf = {a: {} for a in ASPECTS}
    for a_idx, a in enumerate(ASPECTS):
        if topic_count == 0:
            continue
        weights = sims[a_idx]  # size K
        for k, term_scores in tf_icf_scores.items():
            w_k = float(weights[k]) if k < len(weights) else 0.0
            for t, v in term_scores.items():
                aspect_tf_icf[a][t] = aspect_tf_icf[a].get(t, 0.0) + w_k * float(v)

    with open("models/aspect_vocab.json", "w") as f:
        json.dump({"vocab": list(vocab), "aspect_tf_icf": aspect_tf_icf}, f)

    # ---------------- Aspect assignment for docs ----------------
    preds, _sims_doc = aspect_assignment(vocab, aspect_tf_icf, docs_tokens)
    df = ds.df.copy()
    df["aspect_pred"] = preds

    # Quick visibility: distribution after assignment
    counts = df["aspect_pred"].astype(str).value_counts()
    print("\n[INFO] Aspect assignment counts:")
    for a in ASPECTS:
        print(f"  - {a:14s}: {int(counts.get(a, 0))}")
    print()

    # ---------------- Sentiment labeling with WEAK RULES then TextBlob fallback ----------------
    weak_hits = 0
    try:
        from textblob import TextBlob

        def _tb_label(x: str) -> int:
            try:
                return 1 if TextBlob(str(x)).sentiment.polarity >= 0 else 0
            except Exception:
                return 1
    except Exception:
        def _tb_label(x: str) -> int:
            # tiny fallback if TextBlob missing
            s = str(x).lower()
            pos_hits = sum(w in s for w in ["good", "great", "amazing", "clean", "friendly", "nice", "love"])
            neg_hits = sum(w in s for w in ["bad", "dirty", "terrible", "rude", "worst", "hate", "noisy", "smelly"])
            return 1 if (pos_hits - neg_hits) >= 0 else 0

    def _label_with_rules(text: str) -> int:
        nonlocal weak_hits
        lbl = weak_label_sentiment(text)
        if lbl is not None:
            weak_hits += 1
            return int(lbl)
        return _tb_label(text)

    if "sentiment" in df.columns:
        s_num = pd.to_numeric(df["sentiment"], errors="coerce")
        inferred = df["review_text"].apply(_label_with_rules)
        df["sentiment"] = s_num.fillna(inferred).astype(int)
    else:
        df["sentiment"] = df["review_text"].apply(_label_with_rules)

    print(f"[INFO] Weak-label rules assigned {weak_hits} rows; the rest used TextBlob fallback.\n")

    # ---------------- Tokenizer & embeddings (global) ----------------
    lstm_global = AspectLSTM()
    lstm_global.fit_tokenizer(df["review_text"].astype(str).tolist())
    lstm_global.build_embedding_matrix()
    dump(lstm_global.tokenizer, "models/tokenizer.pkl")

    # If your AspectLSTM supports trainable embeddings, enable it
    if hasattr(lstm_global, "embed_trainable"):
        try:
            lstm_global.embed_trainable = True
        except Exception:
            pass

    # ---------------- Train one LSTM per aspect ----------------
    for aspect in ASPECTS:
        sub = df[df["aspect_pred"].astype(str).str.lower() == aspect.lower()].copy()
        n = len(sub)
        if n < min_samples:
            print(f"[SKIP] Aspect '{aspect}' has too few samples ({n} < {min_samples}); skipping.")
            continue

        # Must have both classes to train a binary classifier
        if pd.Series(sub["sentiment"]).nunique() < 2:
            print(f"[SKIP] Aspect '{aspect}' has only one sentiment class; skipping.")
            continue

        X = lstm_global.texts_to_seq(sub["review_text"].astype(str).tolist())
        y = sub["sentiment"].astype(int).to_numpy()

        Xtr, Xva, ytr, yva = safe_train_val_split(X, y, test_size=val_size, random_state=RANDOM_STATE)

        # Class weights for imbalance
        classes = np.unique(ytr)
        class_weight = None
        if len(classes) == 2:
            cw_values = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
            class_weight = {int(c): float(w) for c, w in zip(classes, cw_values)}

        # Build model
        model = lstm_global.build_model()

        # Callbacks: early stop + LR scheduler
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        rl = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=2e-5)

        print(f"[TRAIN] Aspect '{aspect}': {n} samples | class_weight={class_weight}")
        model.fit(
            Xtr,
            ytr,
            epochs=epochs,
            batch_size=32,
            validation_data=(Xva, yva),
            verbose=2,
            class_weight=class_weight,
            callbacks=[es, rl],
        )

        # Save in both formats for compatibility with different loaders
        keras_path = f"models/lstm/{aspect}.keras"
        h5_path = f"models/lstm/{aspect}.h5"
        try:
            model.save(keras_path)
        except Exception as e:
            print(f"[WARN] Saving .keras failed for '{aspect}': {e}")
        try:
            model.save(h5_path)
        except Exception as e:
            print(f"[WARN] Saving .h5 failed for '{aspect}': {e}")

        print(f"✅ Saved LSTM for aspect '{aspect}' with {n} samples.")

    print("All eligible aspects have been trained and saved successfully.")


# -----------------------------------------------------------
#  CLI entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aspect-Based Sentiment Trainer (PLSA + TF-ICF + per-aspect LSTM)"
    )
    parser.add_argument("--csv", required=True, help="Path to the review CSV file")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs per aspect (default: 12)")
    parser.add_argument("--min_samples", type=int, default=20, help="Min samples per aspect (default: 20)")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation split size (default: 0.2)")
    args = parser.parse_args()

    main(args.csv, epochs=args.epochs, min_samples=args.min_samples, val_size=args.val_size)
