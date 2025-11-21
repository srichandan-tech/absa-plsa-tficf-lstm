import argparse
import os
import json
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model

# --- Flexible imports: works with both "python -m src.batch_infer" and "python src/batch_infer.py"
try:
    # Preferred (package) imports
    from src.lstm_model import AspectLSTM
    from src.config import ASPECTS
except ModuleNotFoundError:
    # Fallback when running as a standalone script
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from lstm_model import AspectLSTM
    from config import ASPECTS


class InferenceEngine:
    """
    Unified inference engine for aspect-based sentiment analysis.
    Uses pre-trained aspect TF–ICF vocabulary and aspect-specific LSTM models.
    """

    def __init__(self):
        # Load tokenizer (built during training)
        tok_path = "models/tokenizer.pkl"
        if not os.path.exists(tok_path):
            raise FileNotFoundError("Missing models/tokenizer.pkl. Run training first.")
        self.tokenizer = load(tok_path)

        # Load aspect vocabulary (TF–ICF weights)
        vocab_file = "models/aspect_vocab.json"
        if not os.path.exists(vocab_file):
            raise FileNotFoundError("Missing models/aspect_vocab.json. Run training first.")
        with open(vocab_file, "r") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.aspect_tf_icf = data["aspect_tf_icf"]

        # Load available LSTM models per aspect
        self.models = {}
        for aspect in ASPECTS:
            path = f"models/lstm/{aspect}.h5"
            if os.path.exists(path):
                self.models[aspect] = load_model(path)

        print(f"[INFO] Loaded {len(self.models)} LSTM models for aspects: {list(self.models.keys())}")

        # Helper for tokenization/padding
        self.lstm_helper = AspectLSTM()
        self.lstm_helper.tokenizer = self.tokenizer

    # -----------------------------------------------------------
    #  Determine aspect using simple TF–ICF sum over tokens
    # -----------------------------------------------------------
    def predict_aspect(self, text_tokens):
        """Infer most likely aspect based on TF–ICF similarity (sum of weights)."""
        scores = {}
        for aspect, weights in self.aspect_tf_icf.items():
            s = 0.0
            for w in text_tokens:
                s += weights.get(w, 0.0)
            scores[aspect] = s
        if not scores:
            return "unknown"
        return max(scores, key=scores.get)

    # -----------------------------------------------------------
    #  Predict sentiment for a single review
    # -----------------------------------------------------------
    def predict(self, text: str):
        """
        Return dict: { 'aspect': str, 'sentiment': int(0/1), 'prob': float }
        """
        tokens = text.lower().split()
        aspect = self.predict_aspect(tokens)

        if aspect not in self.models:
            # No trained model for this aspect — default negative with zero confidence
            return {"aspect": aspect, "sentiment": 0, "prob": 0.0}

        seq = self.lstm_helper.texts_to_seq([text])
        prob = float(self.models[aspect].predict(seq, verbose=0)[0][0])
        label = int(prob >= 0.5)
        return {"aspect": aspect, "sentiment": label, "prob": prob}


# -----------------------------------------------------------
#  CSV Batch Inference Pipeline
# -----------------------------------------------------------
def ensure_review_text(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe has a column named 'review_text'."""
    if "review_text" in df.columns:
        return df
    for cand in df.columns:
        if cand.lower() in ("review", "text", "ulasan", "content", "comment", "review text"):
            df = df.copy()
            df.insert(0, "review_text", df[cand].astype(str))
            return df
    raise ValueError("No 'review_text' column found or suitable fallback (e.g., 'Review').")


def main(in_csv: str, out_csv: str, limit: int = 0):
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    df = ensure_review_text(df)
    if limit and limit > 0:
        df = df.head(limit).copy()

    ie = InferenceEngine()
    aspects, sents, scores = [], [], []

    for text in df["review_text"].astype(str).tolist():
        res = ie.predict(text)
        aspects.append(res["aspect"])
        sents.append(int(res["sentiment"]))
        scores.append(float(res["prob"]))

    df["aspect_pred"] = aspects
    df["sentiment_pred"] = sents
    df["score"] = scores

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {len(df)} rows with predictions → {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch aspect-based sentiment inference")
    p.add_argument("--csv", required=True, help="Path to input CSV file")
    p.add_argument("--out", required=True, help="Path to output CSV file")
    p.add_argument("--limit", type=int, default=0, help="Optional limit of rows to process")
    args = p.parse_args()
    main(args.csv, args.out, args.limit)
