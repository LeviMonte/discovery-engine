"""Leakage + behaviour tests for the active-learning loop.

Uses a small synthetic Featurized object so the tests are fast and do not need
network access.
"""
import numpy as np

from discovery.active_loop import run_active_learning
from discovery.featurize import Featurized


def make_synthetic(n=400, d=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    y = X @ w + 0.1 * rng.normal(size=n)
    return Featurized(
        X=X, y=y.astype(np.float64), domain="superconductor",
        std_cols=np.ones(d, dtype=bool),
    )


CFG = {
    "data": {"split": "random", "test_frac": 0.2, "pool_subsample": 0},
    "model": {"hidden": [32, 32], "ensemble_size": 2, "epochs": 12,
              "patience": 4, "lr": 1e-2, "batch_size": 64},
    "active_learning": {"seed_frac": 0.1, "batch_frac": 0.05, "n_rounds": 4,
                        "topk_frac": 0.1, "ucb_beta": 1.5},
}


def test_seed_excludes_topk_and_test_disjoint():
    feat = make_synthetic()
    r = run_active_learning(feat, CFG, "random", seed=0)
    assert r.leakage_ok, r.leakage_notes
    assert r.hits_found[0] == 0     # seed must contain zero top-k hits


def test_hits_monotonic_nondecreasing():
    feat = make_synthetic()
    for strat in ["random", "greedy", "ucb"]:
        r = run_active_learning(feat, CFG, strat, seed=1)
        h = r.hits_found
        assert all(h[i + 1] >= h[i] for i in range(len(h) - 1)), (strat, h)


def test_labeled_grows_by_batch():
    feat = make_synthetic()
    r = run_active_learning(feat, CFG, "greedy", seed=2)
    nl = r.n_labeled
    # strictly increasing acquisitions (training set = union of acquired)
    assert all(nl[i + 1] > nl[i] for i in range(len(nl) - 1))


def test_determinism_same_seed():
    feat = make_synthetic()
    r1 = run_active_learning(feat, CFG, "ucb", seed=3)
    r2 = run_active_learning(feat, CFG, "ucb", seed=3)
    assert r1.hits_found == r2.hits_found
    assert r1.n_labeled == r2.n_labeled
