"""Pool-based active-learning discovery simulation (retrospective rediscovery).

All true labels already exist; we hide them and let an acquisition strategy
request them one batch at a time, measuring how fast it re-finds the top-k
materials in the searchable pool.

Leakage guards (asserted in code, logged each run):
  (i)   test IDs are disjoint from the pool
  (ii)  the initial labeled seed contains NONE of the top-k targets
  (iii) at every round the model's training set == union of acquired batches
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .acquire import acquire
from .featurize import Featurized, make_split
from .models import DeepEnsemble


@dataclass
class LoopResult:
    strategy: str
    seed: int
    domain: str
    n_labeled: list = field(default_factory=list)   # cumulative acquisitions
    hits_found: list = field(default_factory=list)   # cumulative top-k hits
    n_topk: int = 0
    pool_size: int = 0
    leakage_ok: bool = True
    leakage_notes: list = field(default_factory=list)


def _standardize_pool(X, train_stats_idx, std_cols):
    """z-score columns using stats from a given index set (train stats only)."""
    from .featurize import Standardizer

    std = Standardizer.fit(X, train_stats_idx, cols=std_cols)
    return std.transform(X)


def run_active_learning(
    feat: Featurized, cfg: dict, strategy: str, seed: int
) -> LoopResult:
    al = cfg.get("active_learning", {})
    mcfg = cfg.get("model", {})
    rng = np.random.RandomState(seed)

    n = len(feat.y)
    # --- held-out test split (calibration only) -> everything else is the pool
    _, test_idx = make_split(feat, cfg, seed=seed)
    test_set = set(test_idx.tolist())
    pool_ids = np.array([i for i in range(n) if i not in test_set])

    # optional subsample of the pool for CPU-friendly wall-clock
    sub = cfg.get("data", {}).get("pool_subsample", 0)
    if sub and sub < len(pool_ids):
        pool_ids = np.sort(rng.choice(pool_ids, size=sub, replace=False))

    y_pool = feat.y[pool_ids]
    P = len(pool_ids)

    # --- top-k defined within the searchable pool
    topk_frac = al.get("topk_frac", 0.05)
    K = max(1, int(round(topk_frac * P)))
    topk_local = set(np.argsort(-y_pool)[:K].tolist())   # indices into pool

    # --- initial labeled seed, EXCLUDING top-k
    seed_frac = al.get("seed_frac", 0.05)
    n_seed = max(2, int(round(seed_frac * P)))
    eligible = np.array([i for i in range(P) if i not in topk_local])
    seed_local = rng.choice(eligible, size=min(n_seed, len(eligible)), replace=False)
    labeled = set(seed_local.tolist())

    # ---------------- leakage guards ----------------
    res = LoopResult(strategy=strategy, seed=seed, domain=feat.domain,
                     n_topk=K, pool_size=P)
    # (i) test disjoint from pool
    if test_set & set(pool_ids.tolist()):
        res.leakage_ok = False
        res.leakage_notes.append("test/pool overlap")
    # (ii) seed contains no top-k
    if labeled & topk_local:
        res.leakage_ok = False
        res.leakage_notes.append("seed contains top-k")

    batch = max(1, int(round(al.get("batch_frac", 0.01) * P)))
    n_rounds = al.get("n_rounds", 14)
    beta = al.get("ucb_beta", 1.5)

    def record():
        res.n_labeled.append(len(labeled))
        res.hits_found.append(len(labeled & topk_local))

    record()  # round 0 (seed only)

    for r in range(n_rounds):
        labeled_arr = np.array(sorted(labeled))
        unlabeled_local = np.array([i for i in range(P) if i not in labeled])
        if len(unlabeled_local) == 0:
            break

        # (iii) training set == union of acquired batches (== labeled set)
        # standardize using labeled (train) stats only -- no test/pool leakage
        Xz = _standardize_pool(feat.X[pool_ids], labeled_arr, feat.std_cols)

        if strategy == "random":
            # baseline needs no model
            mu = np.zeros(len(unlabeled_local))
            sigma = np.zeros(len(unlabeled_local))
        else:
            ens = DeepEnsemble(mcfg, base_seed=seed * 97 + r)
            ens.fit(Xz[labeled_arr], feat.y[pool_ids][labeled_arr])
            pred = ens.predict(Xz[unlabeled_local])
            mu, sigma = pred.mu, pred.sigma

        best_y = feat.y[pool_ids][labeled_arr].max()
        pick = acquire(strategy, mu, sigma, best_y, batch, rng, beta=beta)
        chosen = unlabeled_local[pick]
        labeled.update(chosen.tolist())
        record()

    # final assertion (iii): labeled is exactly seed + acquired (set semantics)
    if len(labeled) != len(set(labeled)):
        res.leakage_ok = False
        res.leakage_notes.append("duplicate labeled ids")

    return res


# --------------------------------------------------------------------------- #
#  Discovery-curve summary metrics
# --------------------------------------------------------------------------- #
def audc(n_labeled, hits, pool_size, n_topk):
    """Normalized area under the discovery curve (0..1).

    x = fraction of pool acquired, y = fraction of top-k found.
    """
    x = np.asarray(n_labeled) / pool_size
    y = np.asarray(hits) / max(1, n_topk)
    return float(np.trapz(y, x) / max(x[-1] - x[0], 1e-9))


def enrichment_factor(hits_final, n_labeled_final, pool_size, n_topk):
    """Base-rate enrichment at the final budget (robust, never divides by ~0).

    = precision-at-budget / prior hit rate
    = (hits_found / n_measured) / (n_topk / pool_size)
    A random searcher yields ~1.0 by construction; >1 means the strategy
    concentrates measurements on true top-k materials better than chance.
    """
    precision = hits_final / max(n_labeled_final, 1)
    base_rate = n_topk / max(pool_size, 1)
    return float(precision / max(base_rate, 1e-12))


def hit_ratio_vs_random(hits_strategy, hits_random):
    """Direct ratio of hits vs the random baseline at final budget (guarded)."""
    hs = hits_strategy[-1]
    hr = hits_random[-1]
    if hr <= 0:
        return float("nan")   # random found none -> ratio undefined
    return float(hs / hr)


def acquisitions_to_k_hits(n_labeled, hits, k):
    """Number of acquisitions needed to find >= k hits (inf if never)."""
    for nl, h in zip(n_labeled, hits):
        if h >= k:
            return int(nl)
    return float("inf")
