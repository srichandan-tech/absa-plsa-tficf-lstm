#!/usr/bin/env python3
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# -------------------------------
# Helpers
# -------------------------------
CAND_REVIEW = ["review_text", "Review", "review", "text", "content", "comment", "Review Text"]
CAND_ASPECT = ["aspect", "Aspect", "category", "Category", "target", "Target"]
CAND_SENT   = ["sentiment", "Sentiment", "label", "Label", "polarity", "Polarity"]

def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def norm_sent(x):
    if pd.isna(x): return None
    s = str(x).strip().lower()
    if s in {"1","pos","positive","+","p","true"}: return "Positive"
    if s in {"0","neg","negative","-","n","false"}: return "Negative"
    if s in {"neutral","neu","2"}:  return "Neutral"
    return x  # leave as-is

def build_sentiment_label_from_pred(df_pred: pd.DataFrame, pos_thr=0.60, neg_thr=0.40):
    """Create a 'sentiment_label' column if only prob/0-1 columns exist."""
    if "sentiment_label" in df_pred.columns:
        return df_pred

    if "prob" in df_pred.columns:
        def map_prob(p):
            p = float(p)
            if p >= pos_thr: return "Positive"
            if p <= neg_thr: return "Negative"
            return "Neutral"
        df_pred["sentiment_label"] = df_pred["prob"].apply(map_prob)
        return df_pred

    # common case: binary prediction 0/1
    bin_col = None
    for c in ["sentiment_pred", "pred", "y_pred", "prediction"]:
        if c in df_pred.columns:
            bin_col = c; break
    if bin_col:
        df_pred["sentiment_label"] = df_pred[bin_col].map({1:"Positive", 0:"Negative"}).fillna("Neutral")
        return df_pred

    raise ValueError(
        "Predictions file must contain one of: 'sentiment_label', 'prob', or a 0/1 column like 'sentiment_pred'."
    )

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate ABSA predictions vs ground truth.")
    ap.add_argument("--pred", default="data/output/reviews_predicted.csv",
                    help="Predictions CSV (default: data/output/reviews_predicted.csv)")
    ap.add_argument("--gt", default="data/raw/reviews.csv",
                    help="Ground-truth CSV (default: data/raw/reviews.csv)")
    ap.add_argument("--pos_thr", type=float, default=0.60, help="Positive threshold if building labels from prob")
    ap.add_argument("--neg_thr", type=float, default=0.40, help="Negative threshold if building labels from prob")
    args = ap.parse_args()

    # --- Load
    dfp = pd.read_csv(args.pred)
    dft = pd.read_csv(args.gt)

    print("== Columns in predictions ==", list(dfp.columns))
    print("== Columns in ground truth ==", list(dft.columns))

    # --- Harmonize review_text
    pred_review_col = find_col(dfp, CAND_REVIEW)
    gt_review_col   = find_col(dft, CAND_REVIEW)
    if not pred_review_col or not gt_review_col:
        raise ValueError("Could not find a review_text-like column in one of the files.")
    if pred_review_col != "review_text":
        dfp = dfp.rename(columns={pred_review_col: "review_text"})
    if gt_review_col != "review_text":
        dft = dft.rename(columns={gt_review_col: "review_text"})

    # drop duplicates on text to avoid many-to-many merges
    dfp = dfp.drop_duplicates(subset=["review_text"])
    dft = dft.drop_duplicates(subset=["review_text"])

    # --- Build / normalize sentiment labels
    dfp = build_sentiment_label_from_pred(dfp, pos_thr=args.pos_thr, neg_thr=args.neg_thr)

    gt_sent_col = find_col(dft, CAND_SENT)
    if not gt_sent_col:
        raise ValueError(
            "Ground-truth file has no sentiment column. "
            "Expected one of: 'sentiment', 'label', 'polarity' (case-insensitive)."
        )
    if gt_sent_col != "sentiment":
        dft = dft.rename(columns={gt_sent_col: "sentiment"})
    dft["sentiment_norm"] = dft["sentiment"].apply(norm_sent)

    # --- Aspect columns (optional)
    gt_aspect_col = find_col(dft, CAND_ASPECT)
    if gt_aspect_col and gt_aspect_col != "aspect":
        dft = dft.rename(columns={gt_aspect_col: "aspect"})

    pred_aspect_col = "aspect_pred" if "aspect_pred" in dfp.columns else None

    # --- Merge
    merged = dft.merge(dfp, on="review_text", how="inner", suffixes=("_gt","_pred"))
    if merged.empty:
        raise ValueError("No rows matched on 'review_text'. Make sure both files use the same texts.")
    print(f"Merged rows: {len(merged)}")

    # ----------------------------------------------------
    # Aspect metrics (only if both sides available)
    # ----------------------------------------------------
    if "aspect" in merged.columns and pred_aspect_col:
        acc_aspect = accuracy_score(merged["aspect"], merged[pred_aspect_col])
        print(f"\nAspect accuracy: {acc_aspect:.3f}")

        labels_aspect = sorted(list(set(merged["aspect"].dropna()) | set(merged[pred_aspect_col].dropna())))
        cm_aspect = confusion_matrix(merged["aspect"], merged[pred_aspect_col], labels=labels_aspect)
        print("\nAspect confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cm_aspect, index=labels_aspect, columns=labels_aspect))
    else:
        print("\n[Info] Aspect column missing on one side — skipping aspect accuracy.")

    # ----------------------------------------------------
    # Sentiment metrics
    # ----------------------------------------------------
    y_true = merged["sentiment_norm"].astype(str)
    y_pred = merged["sentiment_label"].astype(str)

    # ensure consistent label order if present
    label_order = [l for l in ["Negative", "Neutral", "Positive"] if l in set(y_true) | set(y_pred)]

    acc_sent = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=label_order, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=label_order, zero_division=0)

    print(f"\nSentiment accuracy: {acc_sent:.3f}")
    print(f"F1 (macro):        {f1_macro:.3f}")
    print(f"F1 (weighted):     {f1_weighted:.3f}")

    print("\nSentiment confusion matrix (rows=true, cols=pred):")
    cm_sent = confusion_matrix(y_true, y_pred, labels=label_order)
    print(pd.DataFrame(cm_sent, index=label_order, columns=label_order))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=label_order, zero_division=0))

if __name__ == "__main__":
    main()
