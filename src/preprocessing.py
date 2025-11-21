import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from symspellpy import SymSpell, Verbosity
from .utils import normalize_text
from .config import LANGUAGE

# Stopwords for EN/ID
try:
    stop_en = set(stopwords.words("english"))
except Exception:
    stop_en = set()
try:
    stop_id = set(stopwords.words("indonesian"))
except Exception:
    stop_id = set()

stemmer_en = PorterStemmer()
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer_id = StemmerFactory().create_stemmer()
except Exception:
    stemmer_id = None

# lightweight spell corrector (English only)
_sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
try:
    _sym.load_dictionary("frequency_en_82_765.txt", term_index=0, count_index=1)
except Exception:
    pass

def _stem(token: str) -> str:
    if LANGUAGE == "id":
        if stemmer_id:
            return stemmer_id.stem(token)
        else:
            return token  # fallback: no stemming
    return stemmer_en.stem(token)

def _stopset():
    return stop_id if LANGUAGE == "id" else stop_en

def tokenize(text: str):
    text = normalize_text(text)
    tokens = text.split()
    out = []
    sw = _stopset()
    for t in tokens:
        if t in sw: 
            continue
        if len(t) <= 2: 
            continue
        if LANGUAGE == "en" and _sym.words:
            sug = _sym.lookup(t, Verbosity.TOP, 2)
            if sug:
                t = sug[0].term
        out.append(_stem(t))
    return out
