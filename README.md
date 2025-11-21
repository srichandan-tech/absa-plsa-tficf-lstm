This project is also **inspired by my own published research**, titled:

**“Sentiment Analysis of Hotel Aspect Using Probabilistic Latent Semantic Analysis, Word Embedding and LSTM”**  
International Journal of Intelligent Engineering and Systems (IJIES), 2019  
DOI: **10.22266/ijies2019.0831.26**  
Full text: https://www.inass.org/2019/2019083126.pdf

The methodology in this repository builds upon and significantly extends the core ideas introduced in that paper—namely the integration of **PLSA for aspect discovery**, **word embedding for semantic representation**, and **LSTM for sentiment classification**.  

Where the original research focused on hotel-aspect sentiment at a controlled academic scale, this project evolves the pipeline into a **full production-grade, multi-aspect ABSA system** equipped with:

- Expanded aspect vocabularies through **TF-ICF (100%)**  
- A more expressive **AC3 semantic similarity model**  
- Robust **cue-boosting mechanisms** for cross-aspect disambiguation  
- Automatic multi-aspect selection  
- Weighted global sentiment verdicts  
- A fully interactive **Streamlit interface**  

In essence, this repository represents a **modern, practical, and significantly enhanced continuation** of the research principles established in the 2019 publication—transforming them from theoretical experimentation into a polished, end-to-end tool suitable for real-world multilingual review analysis.


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

```
absa-plsa-tficf-lstm/
├── 📂 data/
│   ├── 📂 external/
│   │   └── 📄 glove.6B.100d.txt          # Required GloVe embeddings
│   └── 📂 raw/
│       └── 📄 reviews.csv                # Training input (user-provided)
│
├── 📂 models/
│   └── 📂 lstm/                          # Saved LSTM sentiment models
│       ├── cleanliness.h5
│       ├── location.h5
│       ├── service.h5
│       ├── sleep_quality.h5
│       ├── like.h5
│       └── mark.h5
│
├── 📂 scripts/
│   ├── 🧪 check_predictions.py           # Sanity-check outputs
│   └── 📊 evaluate_model.py              # Offline evaluation
│
├── 📂 src/
│   ├── 📄 __init__.py
│   ├── ⚙️ config.py                      # Global config (aspects, thresholds, paths)
│   ├── 🧹 text_cleaner.py                # Cleaner preserving bigrams/negations
│   ├── 📘 plsa.py                        # PLSA topic modeling (EM)
│   ├── 🔶 tf_icf.py                      # TF-ICF vocabulary expansion
│   ├── 🛰 semantic_similarity.py         # AC3 similarity + aspect boosting
│   ├── 🤖 lstm_model.py                  # GloVe + LSTM sentiment classifier
│   ├── 🚀 pipeline_train.py              # Full training pipeline
│   └── 🧠 pipeline_infer.py              # Final robust inference engine
│
├── 🖥 streamlit_app.py                   # Streamlit interface
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 LICENSE
└── 📄 .gitignore
```

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
Twitter/X: [**@wiqi_lee**](https://twitter.com/wiqi_lee)  
Feel free to reach out for collaboration or research discussion.

