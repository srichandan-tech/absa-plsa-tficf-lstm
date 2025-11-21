import math
from collections import Counter, defaultdict
from typing import Dict, List

def tf_icf(clusters: Dict[int, List[List[str]]]):
    class_term_counts = {}
    class_sizes = {}
    vocab = set()
    for cid, docs in clusters.items():
        cnt = Counter()
        for doc in docs:
            cnt.update(set(doc))  # presence-based TF per class
            vocab.update(doc)
        class_term_counts[cid] = cnt
        class_sizes[cid] = len(docs)

    cf = Counter()
    for t in vocab:
        for cid, cnt in class_term_counts.items():
            if t in cnt:
                cf[t] += 1
    N = len(clusters)

    tf_icf_scores = defaultdict(dict)
    for cid, cnt in class_term_counts.items():
        for t,c in cnt.items():
            icf = math.log(N / cf[t]) if cf[t] > 0 else 0.0
            tf_icf_scores[cid][t] = c * icf
    return tf_icf_scores
