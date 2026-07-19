import numpy as np

from discovery.models import DeepEnsemble
from discovery.uncertainty import (
    calibration_report, coverage_curve, ece, recalibrate_sigma,
)


def _toy(n=500, d=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    y = X @ w + 0.2 * rng.normal(size=n)
    return X, y.astype(np.float64)


CFG = {"hidden": [32, 32], "ensemble_size": 3, "epochs": 40,
       "patience": 8, "lr": 1e-2, "batch_size": 64}


def test_ensemble_learns_and_decomposes():
    X, y = _toy()
    ens = DeepEnsemble(CFG, base_seed=0).fit(X[:400], y[:400])
    pred = ens.predict(X[400:])
    # positive R^2 on held-out
    ss_res = np.sum((y[400:] - pred.mu) ** 2)
    ss_tot = np.sum((y[400:] - y[400:].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.5, r2
    # decomposition identity
    assert np.allclose(pred.total_var, pred.aleatoric + pred.epistemic)
    assert np.all(pred.epistemic >= -1e-9)


def test_single_has_no_epistemic():
    X, y = _toy()
    ens = DeepEnsemble(CFG, base_seed=0).fit(X[:400], y[:400])
    ps = ens.predict_single(X[400:])
    assert np.allclose(ps.epistemic, 0.0)


def test_coverage_curve_monotone():
    rng = np.random.RandomState(0)
    y = rng.normal(size=2000)
    mu = np.zeros(2000)
    sigma = np.ones(2000)
    ps, emp = coverage_curve(y, mu, sigma)
    # well-specified gaussian -> empirical ~ nominal, low ECE
    assert ece(y, mu, sigma) < 0.05
    assert emp[-1] >= emp[0]


def test_recalibration_reduces_ece_when_miscalibrated():
    rng = np.random.RandomState(0)
    y = rng.normal(size=3000)
    mu = np.zeros(3000)
    sigma = 0.4 * np.ones(3000)          # overconfident
    s = recalibrate_sigma(y, mu, sigma)
    assert ece(y, mu, s * sigma) <= ece(y, mu, sigma)
    assert s > 1.0                       # should inflate sigma
