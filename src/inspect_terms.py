"""
inspect_terms.py
Extract and summarize top-N aspect terms after training.
Usage:
    python -m src.inspect_terms --top 10
"""

import os
import json
import argparse
import pandas as pd

def main(top_n: int = 10):
    path = "models/aspect_terms.json"
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return

    with open(path, "r") as f:
        data = json.load(f)

    rows = []
    print("\n=== TOP TERMS PER ASPECT ===")
    for aspect, terms in data.items():
        selected = terms[:top_n]
        print(f"\n[{aspect.upper()}]")
        print(", ".join(selected))
        rows.append({"Aspect": aspect, "Top_Terms": ", ".join(selected)})

    out_path = "models/term_list_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n✅ Saved summary to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10, help="number of terms per aspect")
    args = parser.parse_args()
    main(args.top)
