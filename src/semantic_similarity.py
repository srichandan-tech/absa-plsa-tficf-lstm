import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .config import ASPECTS, ASPECT_SEED_TERMS

def aspect_assignment(plsa_vocab, tf_icf_expanded, doc_terms_list, use_tf_icf_weight=True):
    vocab_index = {t:i for i,t in enumerate(plsa_vocab)}
    A = len(ASPECTS)
    V = len(plsa_vocab)
    aspect_mat = np.zeros((A, V), dtype=float)

    for a_idx, a in enumerate(ASPECTS):
        for term in ASPECT_SEED_TERMS[a]:
            if term in vocab_index:
                aspect_mat[a_idx, vocab_index[term]] += 1.0
        if isinstance(tf_icf_expanded, dict) and a in tf_icf_expanded:
            for t, w in tf_icf_expanded[a].items():
                if t in vocab_index:
                    aspect_mat[a_idx, vocab_index[t]] += w if use_tf_icf_weight else 1.0

    aspect_mat = aspect_mat / (np.linalg.norm(aspect_mat, axis=1, keepdims=True) + 1e-12)

    doc_mat = np.zeros((len(doc_terms_list), V))
    for i, terms in enumerate(doc_terms_list):
        for t in terms:
            j = vocab_index.get(t)
            if j is not None:
                doc_mat[i, j] += 1
    doc_mat = doc_mat / (np.linalg.norm(doc_mat, axis=1, keepdims=True) + 1e-12)

    sims = cosine_similarity(doc_mat, aspect_mat)  # (D, A)
    pred_idx = sims.argmax(axis=1)
    preds = [ASPECTS[i] for i in pred_idx]
    return preds, sims
