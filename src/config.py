# =========================================================
#  Aspect-Based Sentiment Analysis — Configuration (EN)
# =========================================================
#  Optimized for English hotel/service reviews
#  Used by: PLSA (topics), TF-ICF (terms), and LSTM (sentiment)
#  Notes:
#    • Absolute paths + sanity checks to avoid silent fallbacks.
#    • Negation cues preserved end-to-end (so "not helpful" stays negative).
#    • Three-way decision: Pos ≥ POS_PROB_THRESHOLD; Neg ≤ NEG_PROB_THRESHOLD.
# =========================================================

from __future__ import annotations
import os

# -------------------------------
# Language & canonical aspects
# -------------------------------
LANGUAGE = "en"  # "en" for English reviews, "id" for Indonesian

# Keep this exact order for display and tie-breaking logic
ASPECTS = [
    "location",
    "service",
    "cleanliness",
    "sleep quality",
    "mark",
    "like",
]

# If LIKE overwhelms specific aspects in similarity matching,
# a small penalty helps it not dominate topic→aspect alignment.
# 1.0 = no penalty. Use 0.45–0.60 to keep it in check.
LIKE_SIMILARITY_PENALTY = 0.55  # adjust to 0.45 if LIKE still dominates

# =========================================================
#  Aspect Seed Terms for Semantic Expansion (PLSA + TF-ICF)
#  IMPORTANT: "service" is **clean** (no cleanliness terms) to avoid
#  pulling “dirty room” into the service aspect.
# =========================================================
ASPECT_SEED_TERMS = {
    "sleep quality": [
        "sleep","bed","pillow","mattress","blanket","sheet","bedding","quiet","silence",
        "noise","comfortable","comfort","rest","night","sound","relax","dark","light",
        "temperature","cold","hot","peaceful","disturb","disturbed","tired","insomnia",
        "snore","aircon","fan","calm","noisy","hard","soft","warm","cool","window",
        "curtain","lamp","bright","dim","sleeping","asleep","rested","restless","nap",
        "dream","snoring","sleepless","chilly","breeze","vent","ac","heater","draft",
        "vibration","motion","quietness",
    ],

    # --- SERVICE: help/attitude/speed/front-desk only (no cleanliness terms) ---
    "service": [
        "service","staff","polite","helpful","friendly","rude","quick","slow","manager",
        "reception","front-desk","frontdesk","front_desk",
        "check","checkin","check-in","check-out","check out","check in",
        "attitude","request","employee","doorman","concierge","luggage",
        "customer","question","response","respond","assist","assistance","support",
        "welcoming","professional","smile","greeting","courteous","apology","apologize",
        "bartender","waiter","waitress","housekeeping-staff","handling","resolve","resolved",
        "fix","fixed","room service","front office","information","recommendation","suggestion",
        "internet support",
    ],

    # --- CLEANLINESS: all hygiene/dirty/smell/bathroom items live here ---
    "cleanliness": [
        "cleanliness","clean","dirty","smell","odor","stain","stained","spotless","dust",
        "filthy","hygiene","unhygienic","tidy","messy","neat","organized",
        "bathroom","toilet","sink","shower","towel","sheet","pillow","blanket",
        "soap","tissue","trash","garbage","bin","mold","hair","mirror","drain",
        "laundry","carpet","cobweb","ventilation","fragrance","smoke","sanitary",
        "restroom","bathrobe","bathroomfloor","wall","furniture","dirtiness",
    ],

    "mark": [
        "mark","score","rating","overall","experience","stay","hotel","worth","value",
        "satisfaction","impression","excellent","poor","average","good","bad","amazing",
        "terrible","perfect","recommend","avoid","fantastic","awful","great","horrible",
        "wonderful","nice","disappointing","pleasant","unpleasant","superb","mediocre",
        "exceptional","awesome","satisfying","unsatisfying","comfortable","uncomfortable",
        "worthless","overrated","underrated","memorable","forgettable","notrecommend",
    ],

    # Preference-only (trimmed) to avoid drowning out specific aspects.
    "like": [
        "like","liked","likes",
        "love","loved",
        "enjoy","enjoyed",
        "prefer","preferred","preference",
        "favorite","favourite",
    ],

    "location": [
        "railway","view","station","airport","distance","far","close","convenient","train",
        "metro","times","square","central","park","fifth","rockefeller","grab","uber","near",
        "access","location","place","mall","area","city","walk","walking","shopping",
        "beautiful","efficient","downtown","transport","traffic","neighborhood",
        "surroundings","beach","market","restaurant","landmark","attraction","sightseeing",
        "plaza","skyline","river","sea","lake","mountain","hill","connected","accessible",
        "strategic","quiet","safe",
    ],
}

# =========================================================
#  Project paths (absolute, robust to current working dir)
# =========================================================
# This file lives in: <project_root>/src/config.py
BASE_DIR     = os.path.dirname(os.path.dirname(__file__))     # <project_root>
DATA_DIR     = os.path.join(BASE_DIR, "data")
RAW_DIR      = os.path.join(DATA_DIR, "raw")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")
OUTPUT_DIR   = os.path.join(BASE_DIR, "output")
MODELS_DIR   = os.path.join(BASE_DIR, "models")

# External resources (absolute)
GLOVE_PATH    = os.path.join(EXTERNAL_DIR, "glove.6B.100d.txt")
SYMSPELL_DICT = os.path.join(EXTERNAL_DIR, "frequency_en_82_765.txt")

# Fail fast if resources are missing (prevents silent zero-vectors)
assert os.path.exists(GLOVE_PATH), f"GloVe not found: {GLOVE_PATH}"
assert os.path.exists(SYMSPELL_DICT), f"SymSpell dict not found: {SYMSPELL_DICT}"

# =========================================================
#  Text preprocessing configuration
# =========================================================
LOWERCASE     = True
USE_SYMSPELL  = True     # optional correction; guarded in cleaner
LEMMATIZE     = True
USE_STOPWORDS = True

# SymSpell limits (used by cleaner)
SYMSPELL_MAX_EDIT = 1    # distance ≤ 1 only
SYMSPELL_MIN_FREQ = 50   # ignore suggestions with low frequency

# Negation handling (must be preserved in all stages)
NEGATION_CUES = [
    "not","no","never","hardly","barely","without",
    "isn't","wasn't","aren't","don't","doesn't","didn't",
    "can't","couldn't","won't","wouldn't","shouldn't","haven't","hasn't","hadn't","n't"
]

# Optional contractions map for the cleaner
NEGATION_CONTRACTIONS = {
    "can't": "can not", "won't": "will not", "isn't": "is not",
    "aren't": "are not", "doesn't": "does not", "don't": "do not",
    "didn't": "did not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "couldn't": "could not", "wouldn't": "would not",
    "shouldn't": "should not",
}

# =========================================================
#  Embedding & tokenizer parameters
# =========================================================
EMBED_DIM     = 100  # must match the GloVe file (glove.6B.100d.txt)
MAX_SEQ_LEN   = 96
MIN_TERM_FREQ = 3

# =========================================================
#  Probabilistic Latent Semantic Analysis (PLSA)
# =========================================================
PLSA_TOPICS    = 40
PLSA_MAX_ITERS = 120
RANDOM_STATE   = 42

# =========================================================
#  Sentiment thresholds (used by UI & inference)
# =========================================================
# Tip: If you want 'location' to flip Positive more often, set POS_PROB_THRESHOLD = 0.65.
POS_PROB_THRESHOLD = 0.70
NEG_PROB_THRESHOLD = 0.30

# =========================================================
#  Sanity checks
# =========================================================
assert isinstance(ASPECTS, list) and len(ASPECTS) >= 3, "ASPECTS must be a non-empty list."
for _aspect in ASPECTS:
    assert _aspect in ASPECT_SEED_TERMS, f"Missing seed terms for aspect: {_aspect}"
