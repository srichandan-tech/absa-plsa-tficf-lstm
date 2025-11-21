"""
text_cleaner.py
---------------
Safe, configurable text preprocessing for ABSA.

Pipeline (in order):
  1) Normalize (lowercase, URL removal, common hyphenated variants)
  2) Remove punctuation/symbols (but keep spaces & digits)
  3) Optional SymSpell correction (if package + dictionary exist)
  4) Optional lemmatization (NLTK if available; safe fallback otherwise)
  5) Optional stopword removal (keeps negation + emphasis cues)

Reads toggles from src.config when available:
  - LOWERCASE, USE_SYMSPELL, LEMMATIZE, USE_STOPWORDS, NEGATION_CUES
Falls back to safe defaults if config import fails.

Design goals for bigram safety:
  - We DO NOT remove spaces, so bigrams remain formable.
  - We KEEP negations and emphasis words (e.g., "very", "too", "so", "really"),
    so patterns like "very clean", "too noisy", "smell bad" survive cleaning.
  - SymSpell is optional and guarded to avoid over-correction of domain terms.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional, Set, List

# ---------------------------
# Optional deps (lazy/defensive)
# ---------------------------
try:
    from symspellpy import SymSpell
except Exception:
    SymSpell = None  # type: ignore

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords as nltk_stopwords
except Exception:
    nltk = None  # type: ignore
    WordNetLemmatizer = None  # type: ignore
    nltk_stopwords = None  # type: ignore

# ---------------------------
# Config (import if present)
# ---------------------------
DEFAULTS = {
    "LOWERCASE": True,
    "USE_SYMSPELL": True,
    "LEMMATIZE": True,
    "USE_STOPWORDS": True,
    "NEGATION_CUES": [
        "not", "no", "never", "hardly", "barely", "without",
        "isn't", "wasn't", "don't", "didn't", "can't", "couldn't", "won't",
    ],
}

try:
    # Local import (no heavy objects)
    from .config import (
        LOWERCASE as CFG_LOWERCASE,
        USE_SYMSPELL as CFG_USE_SYMSPELL,
        LEMMATIZE as CFG_LEMMATIZE,
        USE_STOPWORDS as CFG_USE_STOPWORDS,
        NEGATION_CUES as CFG_NEGATION_CUES,
    )
except Exception:
    CFG_LOWERCASE = DEFAULTS["LOWERCASE"]
    CFG_USE_SYMSPELL = DEFAULTS["USE_SYMSPELL"]
    CFG_LEMMATIZE = DEFAULTS["LEMMATIZE"]
    CFG_USE_STOPWORDS = DEFAULTS["USE_STOPWORDS"]
    CFG_NEGATION_CUES = DEFAULTS["NEGATION_CUES"]

# ---------------------------
# Domain-specific protections
# (prevent over-correction by spellchecker)
# ---------------------------
PROTECTED_TOKENS: Set[str] = {
    # Transport/landmarks
    "rockefeller", "times", "square",
    # Hotel & infra terms
    "frontdesk", "front_desk", "checkin", "checkout", "wi", "wifi", "wi-fi",
    "uber", "grab", "housekeeping", "doorman", "concierge",
    "soundproof", "mattress", "restroom",
    # Climate/control abbreviations
    "ac", "aircon",
}

# Map common hyphenated/space variants BEFORE punctuation removal
VARIANT_NORMALIZATIONS = {
    r"\bcheck\-in\b": "checkin",
    r"\bcheck\sin\b": "checkin",
    r"\bcheck\-out\b": "checkout",
    r"\bfront\-desk\b": "frontdesk",
    r"\bwi\-fi\b": "wifi",
}

URL_PATTERN = re.compile(r"http\S+")
NON_ALNUM = re.compile(r"[^a-z0-9\s]")  # after lowercasing
MULTISPACE = re.compile(r"\s+")

# For simple, no-POS lemmatization fallback (very light)
_SIMPLE_LEMMA_RULES = (
    (r"(.*)ies$", r"\1y"),
    (r"(.*)sses$", r"\1ss"),
    (r"(.*)s$", r"\1"),
)

# Emphasis words we MUST keep (so bigrams like "very clean" survive)
EMPHASIS_TOKENS: Set[str] = {
    "very", "too", "so", "quite", "really", "extremely", "super",
}

# ---------------------------
# Stopwords (with negation + emphasis kept)
# ---------------------------
def _build_stopword_set(negations: Iterable[str]) -> Set[str]:
    keep_neg = set(w.lower() for w in negations)
    keep = keep_neg | EMPHASIS_TOKENS  # keep negations AND emphasis words
    sw: Set[str] = set()
    # Try NLTK list
    if nltk_stopwords is not None:
        try:
            sw = set(nltk_stopwords.words("english"))
        except Exception:
            # Fallback minimal list
            sw = {
                "the","a","an","and","or","but","if","then","than","so","to","of","in",
                "on","for","with","at","by","from","as","it","this","that","these",
                "those","i","you","he","she","we","they","me","him","her","us","them",
                "my","your","his","her","our","their","is","am","are","was","were","be",
                "been","being","do","does","did","have","has","had","will","would","can",
                "could","should","shall","may","might","must","there","here",
            }
    # Ensure negations & emphasis are NOT removed
    return {w for w in sw if w not in keep}

# ---------------------------
# SymSpell finder
# ---------------------------
def _find_symspell_dictionary() -> Optional[Path]:
    candidates = [
        # repo relative (recommended)
        Path(__file__).resolve().parents[1] / "data" / "external" / "frequency_en_82_765.txt",
        # project root
        Path("data/external/frequency_en_82_765.txt").resolve(),
        # cwd
        Path("frequency_en_82_765.txt").resolve(),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# ---------------------------
# TextCleaner
# ---------------------------
class TextCleaner:
    def __init__(
        self,
        use_lower: bool = CFG_LOWERCASE,
        use_spell: bool = CFG_USE_SYMSPELL,
        use_lemma: bool = CFG_LEMMATIZE,
        use_stop: bool = CFG_USE_STOPWORDS,
        negation_cues: Optional[List[str]] = None,
    ):
        self.use_lower = bool(use_lower)

        # Allow disabling SymSpell via env
        env_use = os.environ.get("USE_SYMSPELL", "").strip().lower()
        if env_use in {"0", "false", "no"}:
            use_spell = False
        self.use_spell = bool(use_spell)
        self.use_lemma = bool(use_lemma)
        self.use_stop = bool(use_stop)

        self.negations = list(negation_cues) if negation_cues else list(CFG_NEGATION_CUES)

        # Stopwords (negations + emphasis preserved)
        self.stopwords = _build_stopword_set(self.negations) if self.use_stop else set()

        # Optional lemmatizer
        self._wnl = None
        if self.use_lemma and WordNetLemmatizer is not None:
            try:
                self._wnl = WordNetLemmatizer()
            except Exception:
                self._wnl = None

        # Optional SymSpell
        self.symspell = None
        self.use_symspell = False
        if self.use_spell and SymSpell is not None:
            dict_path = _find_symspell_dictionary()
            if dict_path:
                try:
                    sp = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                    sp.load_dictionary(str(dict_path), term_index=0, count_index=1)
                    self.symspell = sp
                    self.use_symspell = True
                    print(f"[INFO] SymSpell loaded: {dict_path}")
                except Exception as e:
                    print(f"[WARN] SymSpell init failed ({e}); spell correction disabled.")
            else:
                print("[INFO] SymSpell dictionary not found; spell correction disabled.")
        else:
            if self.use_spell:
                print("[INFO] SymSpell package not available; spell correction disabled.")

    # ---------------------------
    # Steps
    # ---------------------------
    def _normalize(self, text: str) -> str:
        t = str(text)
        t = URL_PATTERN.sub(" ", t)
        if self.use_lower:
            t = t.lower()

        # Normalize common hyphen/space variants *before* punctuation strip
        for pat, repl in VARIANT_NORMALIZATIONS.items():
            t = re.sub(pat, repl, t)

        # Remove non-alphanumeric (keep space)
        t = NON_ALNUM.sub(" ", t)

        # Collapse whitespace (keep single spaces → bigrams are safe)
        t = MULTISPACE.sub(" ", t).strip()
        return t

    def _correct_spelling(self, text: str) -> str:
        if not (self.use_symspell and self.symspell):
            return text
        # Protect domain tokens: if many protected, skip correction
        tokens = text.split()
        protected_ratio = sum(tok in PROTECTED_TOKENS for tok in tokens) / max(1, len(tokens))
        if protected_ratio >= 0.5:
            return text
        try:
            suggestions = self.symspell.lookup_compound(text, max_edit_distance=2)
            return suggestions[0].term if suggestions else text
        except Exception:
            return text

    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        if not self.use_lemma:
            return tokens
        if self._wnl is not None:
            try:
                return [self._wnl.lemmatize(tok) for tok in tokens]
            except Exception:
                pass
        # very light rules as a safe fallback
        lemmatized: List[str] = []
        for tok in tokens:
            new_tok = tok
            for pat, repl in _SIMPLE_LEMMA_RULES:
                new_tok = re.sub(pat, repl, new_tok)
            lemmatized.append(new_tok)
        return lemmatized

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        if not self.use_stop or not self.stopwords:
            return tokens
        # Keep negation cues, emphasis words, and protected domain tokens
        keep = set(self.negations) | EMPHASIS_TOKENS | PROTECTED_TOKENS
        return [t for t in tokens if (t not in self.stopwords) or (t in keep)]

    # ---------------------------
    # Public API
    # ---------------------------
    def clean(self, text: str) -> str:
        """
        Full pipeline:
          normalize -> spell correction -> tokenize -> lemmatize -> stopword filter
        Notes:
          - Spaces are preserved so bigrams can be formed downstream.
          - Negation + emphasis words are kept even if stopword removal is ON.
        """
        t = self._normalize(text)                 # 1) normalize (lc, urls, symbols)
        t = self._correct_spelling(t)             # 2) optional SymSpell
        tokens = t.split()                        # 3) tokenize (whitespace)
        tokens = self._lemmatize_tokens(tokens)   # 4) optional lemmatize
        tokens = self._remove_stopwords(tokens)   # 5) optional stopword removal (neg+emphasis kept)
        return " ".join(tokens)

# ---------------------------
# Module-level singleton + function API
# ---------------------------
_CLEANER = TextCleaner()

def clean_text(text: str) -> str:
    """Convenience function used across the pipeline."""
    return _CLEANER.clean(text)

# ---------------------------
# CLI smoke test
# ---------------------------
if __name__ == "__main__":
    sample = "Thiss hotel ws amazng!!! Very cleann room and freindly staf. Check-in was quick; Wi-Fi worked. Thin walls smelled bad."
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
