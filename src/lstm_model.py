import numpy as np
from typing import List
from joblib import dump, load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from .config import MAX_SEQ_LEN, EMBED_DIM, GLOVE_PATH


class AspectLSTM:
    """
    Aspect-specific LSTM model builder and tokenizer manager.
    Handles embedding matrix creation, text tokenization, 
    model construction, saving, loading, and sentiment prediction.
    """

    def __init__(self):
        self.tokenizer = Tokenizer(oov_token="<UNK>")
        self.word_index = None
        self.emb_matrix = None
        self.model = None

    # -----------------------------------------------------------
    #  Tokenizer fitting
    # -----------------------------------------------------------
    def fit_tokenizer(self, texts: List[str]):
        """Fit tokenizer on given text corpus."""
        self.tokenizer.fit_on_texts(texts)
        self.word_index = self.tokenizer.word_index
        print(f"[INFO] Tokenizer fitted on {len(self.word_index)} unique tokens.")

    # -----------------------------------------------------------
    #  Build GloVe embedding matrix
    # -----------------------------------------------------------
    def build_embedding_matrix(self):
        """Create embedding matrix from pre-trained GloVe vectors."""
        if self.word_index is None:
            raise ValueError("Tokenizer not fitted. Call fit_tokenizer() first.")

        vocab_size = len(self.word_index) + 1
        matrix = np.zeros((vocab_size, EMBED_DIM))
        glove = {}

        print(f"[INFO] Loading GloVe embeddings from: {GLOVE_PATH}")
        with open(GLOVE_PATH, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != EMBED_DIM + 1:
                    continue  # skip malformed lines
                word = parts[0]
                vector = np.asarray(parts[1:], dtype=float)
                glove[word] = vector

        print(f"[INFO] Loaded {len(glove):,} GloVe vectors.")
        found = 0

        for word, idx in self.word_index.items():
            vec = glove.get(word)
            if vec is not None:
                matrix[idx] = vec
                found += 1

        self.emb_matrix = matrix
        print(f"[INFO] Embedding matrix built: shape={self.emb_matrix.shape}, matched={found}")

    # -----------------------------------------------------------
    #  Build LSTM model
    # -----------------------------------------------------------
    def build_model(self):
        """Construct and compile the LSTM sentiment classifier."""
        if self.emb_matrix is None:
            raise ValueError("Embedding matrix not built. Call build_embedding_matrix() first.")

        model = Sequential([
            Embedding(
                input_dim=self.emb_matrix.shape[0],
                output_dim=EMBED_DIM,
                weights=[self.emb_matrix],
                trainable=False
            ),
            LSTM(128, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print(model.summary())
        self.model = model
        return model

    # -----------------------------------------------------------
    #  Convert text to padded integer sequences
    # -----------------------------------------------------------
    def texts_to_seq(self, texts: List[str]):
        """Convert list of text strings into padded integer sequences."""
        seq = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

    # -----------------------------------------------------------
    #  Save model and tokenizer
    # -----------------------------------------------------------
    def save_model(self, model_path: str, tokenizer_path: str = "models/tokenizer.pkl"):
        """Save trained model and tokenizer."""
        if self.model is None:
            raise ValueError("No model found. Train a model before saving.")
        self.model.save(model_path)
        dump(self.tokenizer, tokenizer_path)
        print(f"[INFO] Model saved to {model_path}")
        print(f"[INFO] Tokenizer saved to {tokenizer_path}")

    # -----------------------------------------------------------
    #  Load model and tokenizer
    # -----------------------------------------------------------
    def load_model(self, model_path: str, tokenizer_path: str = "models/tokenizer.pkl"):
        """Load a trained LSTM model and tokenizer."""
        self.model = load_model(model_path)
        self.tokenizer = load(tokenizer_path)
        print(f"[INFO] Model loaded from {model_path}")
        print(f"[INFO] Tokenizer loaded from {tokenizer_path}")

    # -----------------------------------------------------------
    #  Predict sentiment for new text inputs
    # -----------------------------------------------------------
    def predict(self, texts: List[str]):
        """
        Predict sentiment (0=negative, 1=positive) for input texts.
        Returns a list of integer labels.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        seq = self.texts_to_seq(texts)
        preds = (self.model.predict(seq) > 0.5).astype(int)
        return preds.flatten().tolist()
