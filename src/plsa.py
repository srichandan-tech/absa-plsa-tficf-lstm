import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class PLSA:
    def __init__(self, n_topics=50, max_iter=100, random_state=42):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.random_state = random_state
        self.P_w_z = None
        self.P_z_d = None
        self.vocab = None
        self.vectorizer = None

    def fit(self, docs):
        self.vectorizer = CountVectorizer(tokenizer=lambda x:x, preprocessor=lambda x:x, lowercase=False)
        X = self.vectorizer.fit_transform(docs)  # (D, V)
        self.vocab = self.vectorizer.get_feature_names_out()
        D, V = X.shape
        Z = self.n_topics
        rng = np.random.default_rng(self.random_state)
        P_w_z = rng.random((Z, V))
        P_w_z /= P_w_z.sum(axis=1, keepdims=True)
        P_z_d = rng.random((D, Z))
        P_z_d /= P_z_d.sum(axis=1, keepdims=True)
        X = X.toarray().astype(float)

        for _ in range(self.max_iter):
            numerator = P_z_d[:, :, None] * P_w_z[None, :, :]
            denom = numerator.sum(axis=1, keepdims=True) + 1e-12
            P_z_dw = numerator / denom            # (D, Z, V)

            P_w_z = (P_z_dw * X[:, None, :]).sum(axis=0)
            P_w_z /= P_w_z.sum(axis=1, keepdims=True)

            P_z_d = (P_z_dw * X[:, None, :]).sum(axis=2)
            P_z_d /= P_z_d.sum(axis=1, keepdims=True)

        self.P_w_z = P_w_z
        self.P_z_d = P_z_d
        return self

    def transform(self, docs):
        X = self.vectorizer.transform(docs).toarray().astype(float)
        D, V = X.shape
        Z = self.n_topics
        P_z_d = np.copy(self.P_z_d[:D])
        numerator = P_z_d[:, :, None] * self.P_w_z[None, :, :]
        denom = numerator.sum(axis=1, keepdims=True) + 1e-12
        P_z_dw = numerator / denom
        return (P_z_dw * X[:, None, :]).sum(axis=2)  # topic activations per doc
