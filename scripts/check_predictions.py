import pandas as pd

# === Load the predictions ===
df = pd.read_csv("data/output/reviews_predicted.csv")

print("=== Columns ===")
print(df.columns.tolist())
print("\n=== Sample rows ===")
print(df.head())

print("\n=== Sentiment Distribution ===")
if "sentiment_label" in df.columns:
    print(df["sentiment_label"].value_counts())
elif "sentiment_pred" in df.columns:
    print(df["sentiment_pred"].value_counts())
else:
    print("[WARN] No sentiment column found")

print("\n=== Aspect Predictions ===")
if "aspect_pred" in df.columns:
    print(df["aspect_pred"].value_counts())
else:
    print("[WARN] No aspect column found")
