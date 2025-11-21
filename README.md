# ABSA (Aspect-Based Sentiment Analysis) — PLSA + TF-ICF + LSTM  
Full Multi-Aspect Pipeline with Streamlit App

This project implements a complete ABSA pipeline inspired by academic work on topic modeling and aspect expansion.  
It combines:

- **PLSA** — latent topic discovery  
- **TF-ICF (100%)** — expand vocabulary for each aspect cluster  
- **Semantic Similarity (AC3)** — map documents to canonical aspects  
- **GloVe Word Embedding + LSTM** — per-aspect (or mono) sentiment classifier  
- **Streamlit Application** — for interactive inference, visualization, and optional training  

The system predicts sentiment for **six canonical aspects**:
- Location  
- Service  
- Cleanliness  
- Sleep Quality  
- Mark (Overall Impression)  
- Like (Preference Indicator)

This repository is created by **Wiqi Lee** (Twitter/X: **@wiqi_lee**).

---

## 🚀 Quick Start

### 1. (Optional) Create & activate a virtual environment, then install dependencies:
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt wordnet omw-1.4 averaged_perceptron_tagger stopwords
```

### 2. Download **GloVe 100d**  
Download from the official Stanford website:  
https://nlp.stanford.edu/data/glove.6B.zip

Extract and place:

```
data/external/glove.6B.100d.txt
```

### 3. Prepare your dataset  
Place your CSV at:

```
data/raw/reviews.csv
```

Required column:  
- `review_text` — the original review text  

Optional ground-truth columns:  
- `aspect` — one of the system aspects (location, cleanliness, service, etc.)  
- `sentiment` — 1=positive, 0=negative  

### 4. Train the full pipeline:
```bash
python -m src.pipeline_train --csv data/raw/reviews.csv
```

This runs:
- Text cleaning  
- PLSA topic modeling  
- TF-ICF aspect vocabulary expansion  
- Semantic similarity mapping  
- LSTM training per aspect (auto-skips if too few samples)

### 5. Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

---

## 🌏 Indonesian Dataset Support
If your reviews are Indonesian:

1. Open:
```
src/config.py
```

2. Set:
```python
LANGUAGE = "id"
```

Indonesian mode includes:
- Sastrawi stemmer support (if installed)  
- NLTK Indonesian stopwords  
- Adapted tokenization  

If Sastrawi is missing, it falls back to a lightweight tokenizer.

---

absa-plsa-tficf-lstm/
├─ data/
│  ├─ external/
│  │  └─ glove.6B.100d.txt        # GloVe embeddings (required)
│  └─ raw/
│     └─ reviews.csv              # Training input
├─ models/
│  └─ lstm/                       # Saved LSTM models (mono or per-aspect)
├─ src/
│  ├─ __init__.py
│  ├─ config.py                   # Global config (aspects, thresholds, paths)
│  ├─ text_cleaner.py             # Cleaner preserving bigrams, negations
│  ├─ plsa.py                     # PLSA topic model (EM)
│  ├─ tf_icf.py                   # TF-ICF expansion per aspect
│  ├─ semantic_similarity.py      # AC3 similarity + boosting
│  ├─ lstm_model.py               # GloVe + LSTM sentiment model
│  ├─ pipeline_train.py           # Full training pipeline
│  └─ pipeline_infer.py           # Final robust inference engine
├─ streamlit_app.py               # Streamlit UI entry point
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ LICENSE

---

## 📁 Main Files Overview

| File | Description |
|------|-------------|
| `src/plsa.py` | Compact EM-based PLSA implementation |
| `src/tf_icf.py` | TF-ICF scoring for aspect expansion |
| `src/semantic_similarity.py` | AC3 cosine similarity & boosting |
| `src/lstm_model.py` | GloVe + LSTM binary classifier |
| `src/pipeline_train.py` | Full training pipeline |
| `src/pipeline_infer.py` | Multi-aspect inference engine (final robust version) |
| `src/text_cleaner.py` | Safe text preprocessor (keeps bigrams, negations) |
| `streamlit_app.py` | Streamlit UI (manual/auto multi-aspect, batch, evaluate) |

---

## 📝 Notes
- If an aspect has **too few samples**, its LSTM model is skipped.  
- You may modify thresholds, PLSA topics, GloVe dimension, Top-K aspect selection, and other behaviors in:  
  ```
  src/config.py
  ```
- The inference engine supports:
  - Auto multi-aspect detection  
  - Cue-based boosting (cleanliness, service, location)  
  - Weighted overall review verdict  
  - Full 6-aspect tri-sentiment computation  

---

## 📬 Contact
Created by **Wiqi Lee**  
Twitter/X: **[@wiqi_lee](https://x.com/wiqi_lee)**  
Feel free to reach out for collaboration or research discussion.
