"""Toy datasets (seeded, deterministic) for the gap-filling analysis methods.
Each generator uses a real data-generating process with *known* parameters so the
corresponding method can be shown to recover them in tests and notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "load_rdd", "load_survival", "load_spatial", "load_irt", "load_qca",
    "load_demography", "load_multilevel", "load_stylometry", "load_network",
]


def load_rdd(n: int = 500, cutoff: float = 0.0, tau: float = 3.0, seed: int = 0) -> pd.DataFrame:
    """Sharp RDD: treatment turns on when running var >= cutoff; true jump = ``tau``.
    Columns: running (running variable), treat, y (outcome), x (covariate)."""
    rng = np.random.default_rng(seed)
    running = rng.uniform(-1, 1, n)
    treat = (running >= cutoff).astype(int)
    x = rng.normal(0, 1, n)
    y = 2.0 + 1.5 * running - 0.8 * running**2 + tau * treat + 0.3 * x + rng.normal(0, 0.5, n)
    return pd.DataFrame({"running": running.round(4), "treat": treat,
                         "y": y.round(4), "x": x.round(4)})


def load_survival(n: int = 400, beta: float = 0.8, seed: int = 0) -> pd.DataFrame:
    """Right-censored survival data from an exponential PH model; true log-HR = ``beta``.
    Columns: time (duration), event (1=observed, 0=censored), x (covariate), group."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    group = rng.integers(0, 2, n)
    lin = beta * x + 0.5 * group
    rate = np.exp(lin) * 0.1
    t_event = rng.exponential(1.0 / rate)
    t_cens = rng.exponential(1.0 / (0.05), n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(int)
    return pd.DataFrame({"time": time.round(4), "event": event,
                         "x": x.round(4), "group": group})


def load_spatial(side: int = 8, rho: float = 0.5, seed: int = 0):
    """Spatial autoregressive (SAR) data on a ``side`` x ``side`` rook-contiguity grid;
    true spatial lag = ``rho``. Returns (DataFrame, W) where W is the row-normalized
    n x n weights matrix. DataFrame cols: id, row, col, y, x."""
    rng = np.random.default_rng(seed)
    n = side * side
    coords = [(r, c) for r in range(side) for c in range(side)]
    W = np.zeros((n, n))
    idx = {rc: i for i, rc in enumerate(coords)}
    for (r, c), i in idx.items():
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in idx:
                W[i, idx[nb]] = 1.0
    W = W / W.sum(axis=1, keepdims=True)          # row-normalize
    x = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)
    # y = rho W y + x beta + eps  ->  y = (I - rho W)^-1 (x beta + eps)
    A = np.linalg.inv(np.eye(n) - rho * W)
    y = A @ (1.0 * x + eps)
    df = pd.DataFrame({"id": range(n),
                       "row": [rc[0] for rc in coords], "col": [rc[1] for rc in coords],
                       "y": y.round(4), "x": x.round(4)})
    return df, W


def load_irt(n_persons: int = 400, n_items: int = 10, seed: int = 0):
    """2PL IRT responses. Returns (responses DataFrame [persons x items, 0/1],
    truth DataFrame [item, a, b]) with true discriminations a and difficulties b."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_persons)
    a = rng.uniform(0.7, 2.0, n_items)             # discrimination
    b = np.linspace(-2, 2, n_items)                # difficulty
    logits = a[None, :] * (theta[:, None] - b[None, :])
    p = 1 / (1 + np.exp(-logits))
    resp = (rng.uniform(size=p.shape) < p).astype(int)
    responses = pd.DataFrame(resp, columns=[f"item{j+1}" for j in range(n_items)])
    truth = pd.DataFrame({"item": responses.columns, "a": a.round(3), "b": b.round(3)})
    return responses, truth


def load_qca(seed: int = 0) -> pd.DataFrame:
    """Fuzzy-set QCA data. Outcome Y is (high) when (A AND B) OR C holds — a real
    set-theoretic relation the truth-table minimization should recover.
    Columns: case, A, B, C (fuzzy 0..1 memberships), Y (outcome membership)."""
    rng = np.random.default_rng(seed)
    n = 40
    A = rng.uniform(0, 1, n).round(2)
    B = rng.uniform(0, 1, n).round(2)
    C = rng.uniform(0, 1, n).round(2)
    # fuzzy: AND = min, OR = max; Y sufficient-ish with a little noise
    Y = np.maximum(np.minimum(A, B), C)
    Y = np.clip(Y + rng.normal(0, 0.05, n), 0, 1).round(2)
    return pd.DataFrame({"case": [f"c{i+1}" for i in range(n)],
                         "A": A, "B": B, "C": C, "Y": Y})


def load_demography(seed: int = 0) -> pd.DataFrame:
    """Age-specific mortality for two populations (for life tables + Kitagawa
    decomposition of the crude-rate difference into rate vs age-composition).
    Columns: age_group, n_years, mx_A, mx_B, pop_A, pop_B."""
    ages = ["0", "1-4", "5-14", "15-24", "25-44", "45-64", "65-74", "75-84", "85+"]
    width = [1, 4, 10, 10, 20, 20, 10, 10, 15]
    mx_A = np.array([0.006, 0.0004, 0.0002, 0.0009, 0.0018, 0.008, 0.025, 0.07, 0.18])
    mx_B = mx_A * np.array([1.3, 1.2, 1.1, 1.4, 1.5, 1.3, 1.2, 1.1, 1.05])  # pop B higher mortality
    pop_A = np.array([12, 45, 120, 130, 260, 220, 90, 55, 25], float)       # younger
    pop_B = np.array([8, 30, 95, 110, 230, 240, 130, 90, 55], float)        # older
    return pd.DataFrame({"age_group": ages, "n_years": width,
                         "mx_A": mx_A, "mx_B": mx_B.round(5),
                         "pop_A": pop_A, "pop_B": pop_B})


def load_multilevel(n_groups: int = 30, n_per: int = 20, seed: int = 0) -> pd.DataFrame:
    """Two-level nested data (students within schools) with a random intercept
    (true sd_u ~ 1.0) and a within-group slope (true beta = 2.0).
    Columns: school, student, x, y."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        u = rng.normal(0, 1.0)                       # random intercept
        for s in range(n_per):
            x = rng.normal(0, 1)
            y = 1.0 + u + 2.0 * x + rng.normal(0, 1.0)
            rows.append({"school": g, "student": s, "x": round(x, 4), "y": round(y, 4)})
    return pd.DataFrame(rows)


def load_stylometry(seed: int = 0) -> dict[str, str]:
    """A small corpus of texts by 3 'authors' with distinct function-word habits, so
    Burrows's Delta clusters documents by author. Returns dict doc_id -> text."""
    rng = np.random.default_rng(seed)
    # each author has a characteristic distribution over a shared function-word set
    words = ["the", "and", "of", "to", "a", "in", "that", "it", "is", "was",
             "he", "she", "for", "with", "as", "but", "not", "on", "by", "this"]
    profiles = {
        "austen": np.array([9, 6, 5, 5, 4, 3, 3, 2, 2, 2, 1, 4, 2, 2, 2, 2, 2, 2, 2, 3], float),
        "dickens": np.array([7, 5, 6, 4, 3, 4, 2, 3, 3, 3, 4, 1, 2, 3, 3, 1, 2, 3, 1, 1], float),
        "melville": np.array([8, 4, 7, 5, 3, 3, 3, 2, 3, 4, 3, 1, 3, 2, 4, 2, 1, 3, 3, 1], float),
    }
    corpus = {}
    for author, prof in profiles.items():
        p = prof / prof.sum()
        for k in range(3):                            # 3 docs per author
            counts = rng.multinomial(600, p)
            toks = []
            for w, c in zip(words, counts):
                toks += [w] * int(c)
            rng.shuffle(toks)
            corpus[f"{author}_{k+1}"] = " ".join(toks)
    return corpus


def load_network(n: int = 25, seed: int = 0) -> pd.DataFrame:
    """A directed social network with reciprocity + transitivity built in (so an
    ERGM recovers positive mutual/transitive-triad effects). Returns an edgelist
    DataFrame with columns: source, target."""
    rng = np.random.default_rng(seed)
    # latent homophily positions drive base ties; then add reciprocity & closure
    pos = rng.normal(0, 1, (n, 2))
    edges = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = np.linalg.norm(pos[i] - pos[j])
            if rng.uniform() < np.exp(-1.5 * d) * 0.5:
                edges.add((i, j))
    # inject reciprocity
    for (i, j) in list(edges):
        if rng.uniform() < 0.5:
            edges.add((j, i))
    # inject transitive closure i->j, j->k => i->k
    adj = {}
    for (i, j) in edges:
        adj.setdefault(i, set()).add(j)
    for i in list(adj):
        for j in list(adj.get(i, [])):
            for k in list(adj.get(j, [])):
                if k != i and rng.uniform() < 0.4:
                    edges.add((i, k))
    el = sorted(edges)
    return pd.DataFrame(el, columns=["source", "target"])
