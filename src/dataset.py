import pandas as pd
from typing import List
from .preprocessing import tokenize

REVIEW_COL = "review_text"

class ReviewDataset:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        if REVIEW_COL not in df.columns:
            raise ValueError(f"CSV must contain column '{REVIEW_COL}'")
        self.df = df

    def tokenized_docs(self) -> List[List[str]]:
        return [tokenize(x) for x in self.df[REVIEW_COL].astype(str).tolist()]
