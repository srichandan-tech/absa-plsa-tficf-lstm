# src/pipeline_infer.py
# -----------------------------------------------------------
# Inference engine for ABSA:
# - Aspect mapping (TF-ICF/PLSA similarity) + per-aspect LSTM sentiment
# - Prefer aspect chosen by LSTM max-confidence, fallback to similarity
# - Stronger lexicon fallback (incl. bigrams & cleanliness/service/location cues)
# - Cue-boost in similarity to reduce “LIKE” dominating specific aspects
# - Multi-aspect prediction (Top-K or AUTO by confidence)
# - predict_all() -> returns all aspects with tri-sentiment scores + conclusion
# - Works with per-aspect models or a mono 'sentiment_lstm.h5'
# -----------------------------------------------------------

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

from joblib import load as joblib_load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Local config / similarity
from .config import (
    MAX_SEQ_LEN,
    LIKE_SIMILARITY_PENALTY,
    ASPECTS as CANONICAL_ASPECTS,
)
from .semantic_similarity import aspect_assignment

# Optional cleaner (fallback to a minimal cleaner if not available)
try:
    from .text_cleaner import clean_text  # type: ignore
except Exception:
    def clean_text(s: str) -> str:
        return " ".join(str(s).lower().strip().split())


# --------------------------
# Lightweight lexicons
# --------------------------
POS_UNI = {
    "good","great","excellent","amazing","awesome","fantastic","perfect","nice",
    "clean","friendly","helpful","cozy","comfy","welcoming","peaceful","beautiful",
    "love","loved","like","liked","enjoy","enjoyed","recommend","recommended","worth",
    "affordable","superb","wonderful","brilliant","ideal"
}
NEG_UNI = {
    "bad","terrible","awful","horrible","worst","dirty","filthy","smelly","smell",
    "stain","stained","broken","rude","unhelpful","noisy","noise","disgusting",
    "poor","slow","overpriced","expensive","cold","hot","hard","uncomfortable",
    "issue","problem"
}
# High-signal bigrams (space-joined)
POS_BI = {
    "very clean","spotlessly clean","extremely helpful","super friendly","well located",
    "great location","convenient location","comfortable bed","quiet room","fast checkin",
}
NEG_BI = {
    "not helpful","very dirty","smelled bad","smell bad","stained carpet","rude staff",
    "poor service","too noisy","paper thin","thin walls","bathroom dirty","moldy bathroom",
    "no hot","no cold","aircon broken","ac broken","staff unhelpful",
    # extra robustness for common phrasings
    "was dirty","room dirty","rooms dirty","dirty room"
}

# Negatives that belong to cleanliness/service (to suppress when aspect == location)
CLEAN_SERVICE_NEG_UNI = {
    "dirty","filthy","smelly","smell","stain","stained","mold","unhelpful","rude","poor"
}
CLEAN_SERVICE_NEG_BI = {
    "bathroom dirty","stained carpet","smell bad","smelled bad","moldy bathroom",
    "not helpful","rude staff","poor service","room dirty","rooms dirty","dirty room","was dirty"
}

# Aspect cue words (unigrams) and bigrams to boost similarity
ASPECT_CUES: Dict[str, Dict[str, set[str]]] = {
    "cleanliness": {
        "uni": {
            "clean","dirty","spotless","filthy","hygiene","tidy","stain","stained",
            "odor","smell","musty","dust","bathroom","toilet","shower","mold",
            "sanitary","messy","trash","garbage","bin","soap","tissue","bath","wipe",
            "hair","mirror","restroom","toiletries","bathrobe","drain",
        },
        "bi": {
            "very clean","spotlessly clean","bathroom dirty","stained carpet",
            "smell bad","smelled bad","moldy bathroom","was dirty","room dirty","dirty room","rooms dirty",
        },
    },
    "service": {
        "uni": {
            "staff","reception","front","checkin","check-in","checkout","check-out",
            "manager","housekeeping","helpful","friendly","polite","rude","unhelpful",
            "concierge","luggage","doorman","maid","request","employee","bartender",
            "wifi","internet","parking","gym","lobby",
        },
        "bi": {"rude staff","poor service","fast checkin","quick checkin","helpful staff","not helpful"},
    },
    "sleep quality": {
        "uni": {
            "bed","pillow","mattress","blanket","sheet","sleep","sleeping","quiet",
            "noise","noisy","night","rest","snore","aircon","ac","heater","hard","soft",
            "warm","cool","curtain","lamp","dark","bright",
        },
        "bi": {"quiet room","paper thin","thin walls","comfortable bed","too noisy"},
    },
    "location": {
        "uni": {
            "location","near","close","walk","walking","distance","central","downtown",
            "station","train","metro","airport","view","access","convenient","beach",
            "market","restaurant","landmark","attraction","connected","accessible",
            "safe","perfect","great","excellent","ideal"
        },
        "bi": {"great location","convenient location","well located"},
    },
    "mark": {  # overall judgement words
        "uni": {"rating","score","overall","experience","value","worth","recommend"},
        "bi": set(),
    },
    "like": {  # intentionally very light; we’ll penalize it below when scoring
        "uni": {"like","love","enjoy","favorite","satisfied","pleasant","lovely"},
        "bi": set(),
    },
}


# -----------------------------------------------------------
# Utility: map model outputs to positive probability
# Supports: [N,1] sigmoid OR [N,2] softmax
# -----------------------------------------------------------
def _to_positive_prob(y: np.ndarray) -> float:
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        return float(y[0, 0])
    if y.ndim == 2 and y.shape[1] == 2:
        return float(y[0, 1])  # class-1 = Positive
    if y.ndim == 1 and y.size == 1:
        return float(y[0])
    return float(np.ravel(y)[0])


class InferenceEngine:
    """
    PLSA/TF-ICF → aspect mapping + per-aspect LSTM sentiment classifier.

    Public methods:
      - predict(text) -> dict
      - predict_multi(text, aspect_top_k=None, aspect_min_conf=0.2) -> dict
      - predict_all(text, pos_thr=0.70, neg_thr=0.30) -> dict
      - predict_all_aspects(text) -> {aspect: prob}
      - batch_predict(texts) -> pd.DataFrame
    """

    def __init__(
        self,
        model_dir: str = "models/lstm",
        tokenizer_path: str = "models/tokenizer.pkl",
        aspect_vocab_path: str = "models/aspect_vocab.json",
        threshold: float = 0.5,
    ):
        self.model_dir = model_dir
        self.tokenizer_path = tokenizer_path
        self.aspect_vocab_path = aspect_vocab_path
        self.threshold = float(threshold)

        # ---- Tokenizer
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found at {self.tokenizer_path}. Train the pipeline first."
            )
        self.tokenizer = joblib_load(self.tokenizer_path)

        # ---- Aspect vocab + TF-ICF map
        if not os.path.exists(self.aspect_vocab_path):
            raise FileNotFoundError(
                f"Aspect vocab file not found at {self.aspect_vocab_path}. "
                f"Run training to generate it."
            )
        with open(self.aspect_vocab_path, "r") as f:
            blob = json.load(f)

        self.vocab: List[str] = blob.get("vocab", []) or []
        self.aspect_tf_icf: Dict[str, Dict[str, float]] = blob.get("aspect_tf_icf", {}) or {}
        if not self.vocab or not self.aspect_tf_icf:
            raise ValueError("Invalid aspect_vocab.json: missing 'vocab' or 'aspect_tf_icf'.")

        # Canonical aspect order (prefer config order)
        self.aspects: List[str] = [a for a in CANONICAL_ASPECTS if a in self.aspect_tf_icf] or \
                                  sorted(list(self.aspect_tf_icf.keys()))

        # ---- Load LSTM models
        self.models: Dict[str, Any] = {}
        self._load_models()

        # Info only
        self.available_aspects = sorted(set(self.aspects) | set(self.models.keys()))
        print(f"[INFO] Loaded {len(self.models)} LSTM models. Keys: {sorted(self.models.keys())}")

        # ---- Tiny warm-up to reduce tf.function retracing warnings
        try:
            dummy = np.zeros((1, MAX_SEQ_LEN), dtype="int32")
            for _a, m in self.models.items():
                _ = m.predict(dummy, verbose=0)
        except Exception as _e:
            print(f"[WARN] Warm-up failed: {_e}")

    # ---------------------------
    # Model loading
    # ---------------------------
    def _load_models(self) -> None:
        """Load per-aspect models or a mono model if present."""
        if not os.path.isdir(self.model_dir):
            print(f"[WARN] LSTM model directory not found: {self.model_dir}")
            return

        # Mono model?
        mono = None
        for base in ("sentiment_lstm.h5", "sentiment_lstm.keras"):
            p = os.path.join(self.model_dir, base)
            if os.path.exists(p):
                mono = p
                break

        if mono:
            try:
                m = load_model(mono, compile=False)
                # Use the same mono model for every aspect (lightweight dict of refs)
                for a in CANONICAL_ASPECTS:
                    self.models[a] = m
                return
            except Exception as e:
                print(f"[WARN] Failed to load mono model '{mono}': {e}")

        # Otherwise load per-aspect models; map common filename variants
        files = [f for f in os.listdir(self.model_dir) if f.endswith((".h5", ".keras"))]
        for f in files:
            path = os.path.join(self.model_dir, f)
            name = os.path.splitext(f)[0]  # e.g., 'sleep quality' or 'sleep_quality'
            # Normalize to canonical aspect key
            candidates = {
                name,
                name.replace("_", " "),
                name.replace(" ", "_"),
                name.lower(),
                name.lower().replace("_", " "),
            }
            target = None
            for a in CANONICAL_ASPECTS:
                if a in candidates or a.replace(" ", "_") in candidates:
                    target = a
                    break
            if target is None:
                target = name
            try:
                self.models[target] = load_model(path, compile=False)
            except Exception as e:
                print(f"[WARN] Failed to load model '{path}': {e}")

    # ---------------------------
    # Internals
    # ---------------------------
    def _texts_to_seq(self, texts: List[str]) -> np.ndarray:
        seq = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

    def _safe_first(self, seq: Any) -> Any:
        if isinstance(seq, (list, tuple)):
            return seq[0] if len(seq) > 0 else None
        if isinstance(seq, np.ndarray):
            return seq[0] if seq.size > 0 else None
        return seq

    def _scores_to_dict(self, scores: Any) -> Dict[str, float]:
        if scores is None:
            return {}
        if isinstance(scores, dict):
            return {str(k): float(v) for k, v in scores.items()}
        if isinstance(scores, (list, tuple, np.ndarray)):
            arr = np.asarray(scores).astype(float)
            if arr.size == 0:
                return {}
            aspects = self.aspects if self.aspects else list(self.models.keys())
            return {a: float(arr[i]) for i, a in enumerate(aspects) if i < arr.shape[0]}
        return {}

    @staticmethod
    def _normalize_0_1(d: Dict[str, float]) -> Dict[str, float]:
        if not d:
            return {}
        vals = np.array([v for v in d.values() if np.isfinite(v)], dtype=float)
        if vals.size == 0:
            return {k: 0.5 for k in d}
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax - vmin < 1e-9:
            return {k: 0.5 for k in d}
        return {k: (float(v) - vmin) / (vmax - vmin) for k, v in d.items()}

    # ---------------------------
    # LSTM-based aspect chooser (preferred)
    # ---------------------------
    def _best_aspect_via_lstm(self, text: str) -> Tuple[Optional[str], Dict[str, float]]:
        if not self.models:
            return None, {}

        cleaned = clean_text(text)
        X = self._texts_to_seq([cleaned])

        probs: Dict[str, float] = {}
        confidences: Dict[str, float] = {}

        for aspect, model in self.models.items():
            try:
                raw = model.predict(X, verbose=0)
                p = _to_positive_prob(raw)
            except Exception as e:
                print(f"[WARN] Inference failed for aspect '{aspect}': {e}")
                p = float("nan")
            probs[aspect] = p
            confidences[aspect] = max(p, 1.0 - p) if np.isfinite(p) else 0.0

        if not confidences:
            return None, probs

        best_aspect = max(confidences.items(), key=lambda kv: kv[1])[0]
        return best_aspect, probs

    # ---------------------------
    # Similarity-based aspect assignment with cue-boost
    # ---------------------------
    def _assign_aspect_for_text(self, text: str) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Start with TF-ICF/PLSA similarity, then apply targeted boosts so
        obviously-mentioned aspects win their slot (e.g., 'location', 'dirty', 'not helpful').
        """
        cleaned = clean_text(text)
        tokens = cleaned.split()
        preds, sims = aspect_assignment(self.vocab, self.aspect_tf_icf, [tokens])

        # Base scores from similarity model
        first = self._safe_first(sims)
        scores = self._scores_to_dict(first)

        # N-grams
        unis = set(tokens)
        bigrams = {" ".join(pair) for pair in zip(tokens, tokens[1:])}

        # Generic cue boosts
        for aspect in self.aspects:
            cues = ASPECT_CUES.get(aspect, {"uni": set(), "bi": set()})
            uni_hits = len(unis & cues.get("uni", set()))
            bi_hits = len(bigrams & cues.get("bi", set()))
            if uni_hits or bi_hits:
                boost = 0.08 * uni_hits + 0.15 * bi_hits
                scores[aspect] = float(scores.get(aspect, 0.0) + boost)

        # Targeted strong boosts (make the right aspect jump)
        # LOCATION
        if "great location" in bigrams or "convenient location" in bigrams or "well located" in bigrams:
            scores["location"] = float(scores.get("location", 0.0) + 0.55)
        if ("location" in unis) and ({"perfect","great","excellent","ideal"} & unis):
            scores["location"] = float(scores.get("location", 0.0) + 0.35)

        # CLEANLINESS
        if ({"dirty","filthy","stain","stained","smell","mold"} & unis) \
            or ({"bathroom dirty","stained carpet","moldy bathroom","smell bad","smelled bad","was dirty","dirty room","room dirty","rooms dirty"} & bigrams):
            scores["cleanliness"] = float(scores.get("cleanliness", 0.0) + 0.45)

        # SERVICE
        if ("not helpful" in bigrams) or ("rude staff" in bigrams) or ("poor service" in bigrams) or ("unhelpful" in unis):
            scores["service"] = float(scores.get("service", 0.0) + 0.45)

        # Softly penalize LIKE so it doesn't overshadow specific aspects
        if "like" in scores and isinstance(LIKE_SIMILARITY_PENALTY, (int, float)):
            scores["like"] = float(scores["like"] * float(LIKE_SIMILARITY_PENALTY))

        # Best label after boosts
        best_pred: Optional[str] = None
        if isinstance(preds, (list, tuple, np.ndarray)) and len(preds) > 0:
            best_pred = preds[0]
        elif isinstance(preds, str):
            best_pred = preds

        if scores:
            best_pred = max(scores.items(), key=lambda kv: kv[1])[0]

        return best_pred, scores

    # ---------------------------
    # Tiny helper: does text contain cues for a given aspect?
    # ---------------------------
    def _has_aspect_cues(self, text: str, aspect: str) -> bool:
        cleaned = clean_text(text)
        toks = cleaned.split()
        unis = set(toks)
        bigrams = {" ".join(pair) for pair in zip(toks, toks[1:])}
        cues = ASPECT_CUES.get(aspect, {"uni": set(), "bi": set()})
        if (unis & cues.get("uni", set())):
            return True
        if (bigrams & cues.get("bi", set())):
            return True
        # extra hard-coded triggers
        if aspect == "cleanliness":
            if unis & {"dirty","filthy","stain","stained","smell","mold"}:
                return True
            if bigrams & {"bathroom dirty","stained carpet","smell bad","smelled bad","moldy bathroom","was dirty","dirty room","room dirty","rooms dirty"}:
                return True
        if aspect == "service":
            if "unhelpful" in unis or ("not helpful" in bigrams) or ("rude staff" in bigrams) or ("poor service" in bigrams):
                return True
        if aspect == "location":
            if ("location" in unis and ({"perfect","great","excellent","ideal"} & unis)) \
               or ({"great location","convenient location","well located"} & bigrams):
                return True
        return False

    # ---------------------------
    # Sentiment for a chosen aspect (model ⨉ lexicon blend)
    # ---------------------------
    def _predict_sentiment_for_aspect(self, text: str, aspect: str) -> Tuple[int, float]:
        """
        Blend the model’s probability with a lexicon-based probability.
        Decisive phrases strongly steer the score for the relevant aspect.
        """
        cleaned = clean_text(text)
        X = self._texts_to_seq([cleaned])

        # Build n-grams once
        toks = cleaned.split()
        unis = set(toks)
        bigrams = {" ".join(pair) for pair in zip(toks, toks[1:])}

        # ---- Lexicon score (sigmoid over pos-neg)
        pos = sum(1 for w in toks if w in POS_UNI) + 1.6 * sum(1 for b in bigrams if b in POS_BI)
        neg = sum(1 for w in toks if w in NEG_UNI) + 1.8 * sum(1 for b in bigrams if b in NEG_BI)

        # Aspect-specific decisive cues and bleed control
        if aspect == "cleanliness":
            neg += 0.7 * len(unis & {"dirty","stain","stained","smell","filthy","mold"})
            neg += 1.0 * len(bigrams & {"bathroom dirty","stained carpet","smell bad","smelled bad","moldy bathroom","was dirty","dirty room","room dirty","rooms dirty"})
        elif aspect == "service":
            neg += 1.3 * len(bigrams & {"not helpful"})
            neg += 1.0 * len(bigrams & {"rude staff","poor service"})
            neg += 0.7 * len(unis & {"unhelpful","rude"})
        elif aspect == "location":
            # Strong positive cues for location quality
            pos += 1.6 * len(bigrams & {"great location","convenient location","well located"})
            if "location" in unis and ({"perfect","great","excellent","ideal"} & unis):
                pos += 1.4

            # Suppress negatives that belong to cleanliness/service (avoid cross-aspect bleed)
            cs_unis = len(unis & CLEAN_SERVICE_NEG_UNI)
            cs_bi   = len(bigrams & CLEAN_SERVICE_NEG_BI)
            if cs_unis or cs_bi:
                neg = max(0.0, neg - (0.9 * cs_unis + 1.2 * cs_bi))

        lex_score = pos - neg
        lex_prob = 1.0 / (1.0 + np.exp(-lex_score))

        # ---- Model probability (if available)
        model = self.models.get(aspect)
        if model is not None:
            try:
                raw = model.predict(X, verbose=0)
                model_prob = _to_positive_prob(raw)
            except Exception:
                model_prob = 0.5
        else:
            model_prob = 0.5  # no model for this aspect

        # ---- Blend (slightly trust lexicon more when strong location cue detected)
        strong_loc_cue = (aspect == "location") and (
            ("location" in unis and ({"perfect","great","excellent","ideal"} & unis)) or
            ({"great location","convenient location","well located"} & bigrams)
        )
        model_conf = abs(model_prob - 0.5)          # 0..0.5
        base_w_model = min(0.9, max(0.4, 0.4 + 1.0 * model_conf))
        if strong_loc_cue:
            w_model = max(0.35, base_w_model - 0.10)  # nudge toward lexicon on strong cue
        else:
            w_model = base_w_model
        w_lex = 1.0 - w_model

        prob = float(w_model * model_prob + w_lex * lex_prob)

        # Positive floor when extremely explicit location praise
        if strong_loc_cue:
            prob = max(prob, 0.68)

        label = int(prob >= self.threshold)
        return label, prob

    # ---------------------------
    # Public API
    # ---------------------------
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Single-text inference:
          1) Choose aspect via LSTM max-confidence (preferred)
          2) If confidence is flat (~0.5) or models missing, fall back to similarity
          3) Predict sentiment for the chosen aspect
        """
        if not isinstance(text, str):
            text = str(text)

        # 1) LSTM vote across aspects
        aspect_lstm, probs_all = self._best_aspect_via_lstm(text)

        # 2) Similarity fallback / tiebreaker (+ cue-boost)
        aspect_sim, sim_scores = self._assign_aspect_for_text(text)

        # Decide which aspect to use
        if aspect_lstm is not None:
            max_conf = 0.0
            if probs_all:
                max_conf = max((max(p, 1 - p) for p in probs_all.values() if np.isfinite(p)), default=0.0)
            # If LSTM is indecisive (~0.5 everywhere), let similarity break ties
            aspect = aspect_sim if (max_conf < 0.55 and aspect_sim is not None) else aspect_lstm
        elif aspect_sim is not None:
            aspect = aspect_sim
        else:
            aspect = next(iter(self.models.keys()), "service")

        # Sentiment for chosen aspect
        sent_label, sent_prob = self._predict_sentiment_for_aspect(text, aspect)

        # Scores for the UI bar chart → normalize to [0,1]
        scores_for_plot = probs_all if probs_all else sim_scores
        scores_for_plot = self._normalize_0_1(scores_for_plot)

        return {
            "aspect": aspect,
            "aspect_scores": scores_for_plot,
            "sentiment": int(sent_label),
            "prob": float(sent_prob),
            "cleaned_text": clean_text(text),
        }

    def predict_multi(
        self,
        text: str,
        aspect_top_k: Optional[int] = None,
        aspect_min_conf: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Multi-aspect inference:
          - AUTO mode: if aspect_top_k in (0, None) → select **all** aspects whose
            normalized similarity ≥ aspect_min_conf (after cue-boost).
          - Top-K mode: if aspect_top_k > 0 → take top-K by similarity, then
            filter by aspect_min_conf.
          - Force-include 'cleanliness' if its cues are present.
          - For each selected aspect, runs the aspect LSTM and returns probs.
        """
        if not isinstance(text, str):
            text = str(text)

        # similarity map (already cue-boosted)
        best_aspect, sim_scores = self._assign_aspect_for_text(text)
        sim_scores_norm = self._normalize_0_1(sim_scores)

        # choose candidate aspects
        ranked = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
        if aspect_top_k is not None and aspect_top_k > 0:
            ranked = ranked[:aspect_top_k]
        # filter by min confidence
        selected = [a for a, _s in ranked if sim_scores_norm.get(a, 0.0) >= aspect_min_conf]
        # AUTO fallback: if no cap and filter removed all, take any with nonzero sim
        if (aspect_top_k in (0, None)) and not selected:
            selected = [a for a, _ in ranked] or list(sim_scores.keys())

        # normal fallback if still empty
        if not selected:
            selected = [best_aspect] if best_aspect else ([self.available_aspects[0]] if self.available_aspects else [])

        # ---- Force-include cleanliness when its cues appear ----
        cleanliness_cued = self._has_aspect_cues(text, "cleanliness")
        if cleanliness_cued and "cleanliness" not in selected:
            if aspect_top_k and aspect_top_k > 0:
                ranked_full = [a for a, _ in sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)]
                picked = ["cleanliness"] + [a for a in ranked_full if a != "cleanliness"]
                picked = list(dict.fromkeys(picked))[:aspect_top_k]
                selected = picked
            else:
                selected.append("cleanliness")

        # run per-aspect LSTM
        results = []
        for a in selected:
            y, p = self._predict_sentiment_for_aspect(text, a)
            results.append({"aspect": a, "sentiment": int(y), "prob": float(p)})

        # diagnostics: all aspect probabilities from LSTM
        all_probs = self.predict_all_aspects(text)

        return {
            "aspects": results,
            "aspect_scores": self._normalize_0_1(sim_scores or all_probs),
            "all_probs": all_probs,
            "cleaned_text": clean_text(text),
        }

    def predict_all(self, text: str, pos_thr: float = 0.70, neg_thr: float = 0.30) -> Dict[str, Any]:
        """
        Return every aspect with:
          - prob_positive, prob_negative, prob_neutral
          - sentiment_label (lean) using thresholds
          - a short conclusion
          - normalized similarity scores for plotting
        """
        if not isinstance(text, str):
            text = str(text)

        _, sim_scores = self._assign_aspect_for_text(text)
        sim_scores_norm = self._normalize_0_1(sim_scores)

        rows = []
        order = self.aspects or self.available_aspects
        for a in order:
            y, ppos = self._predict_sentiment_for_aspect(text, a)
            pneg = float(1.0 - ppos)
            pneu = float(max(0.0, 1.0 - abs(ppos - 0.5) * 2.0))  # 1 near 0.5, 0 near edges

            if ppos >= pos_thr:
                lean = "Positive"
            elif ppos <= neg_thr:
                lean = "Negative"
            else:
                lean = "Neutral"

            concl = f"{a} leans {lean.lower()} (pos={ppos:.3f}, neg={pneg:.3f}, neu={pneu:.3f})."

            rows.append({
                "aspect": a,
                "sentiment": int(y),
                "sentiment_label": lean,
                "prob_positive": float(round(ppos, 6)),
                "prob_negative": float(round(pneg, 6)),
                "prob_neutral": float(round(pneu, 6)),
                "leans_to": lean,
                "sim": float(sim_scores_norm.get(a, 0.0)),
                "conclusion": concl,
            })

        return {
            "aspects": rows,
            "aspect_scores": sim_scores_norm,
            "cleaned_text": clean_text(text),
        }

    def predict_all_aspects(self, text: str) -> Dict[str, float]:
        """Probability of positive sentiment for ALL aspect models."""
        _, probs = self._best_aspect_via_lstm(text)
        return probs or {}

    def batch_predict(self, texts: List[str]) -> pd.DataFrame:
        """Batch inference for a list of texts."""
        aspects, sents, probs, cleaned_texts = [], [], [], []
        for t in texts:
            res = self.predict(t)
            aspects.append(res.get("aspect"))
            sents.append(int(res.get("sentiment", 0)))
            probs.append(float(res.get("prob", 0.0)))
            cleaned_texts.append(res.get("cleaned_text", ""))
        return pd.DataFrame(
            {
                "review_text": list(texts),
                "cleaned_text": cleaned_texts,
                "aspect_pred": aspects,
                "sentiment_pred": sents,
                "score": probs,
            }
        )
