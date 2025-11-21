import re, unicodedata
from typing import List

def normalize_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
