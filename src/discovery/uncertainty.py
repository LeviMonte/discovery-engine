"""Calibration metrics for regression predictive uncertainty.

Given predictive mean mu and std sigma on a held-out set:
  * reliability / coverage curve: for a nominal central-interval coverage p,
    the empirical fraction of points with |y - mu| <= z_p * sigma.
  * ECE / miscalibration area: mean |empirical - nominal| over a coverage grid.
  * Spearman rho between sigma and |error|: does uncertainty *rank* error?
Optional post-hoc variance recalibration finds a scalar s so that s*sigma is
better calibrated on a validation split.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import norm


def coverage_curve(y, mu, sigma, ps=None):
    """Return (nominal_ps, empirical_coverage) for central intervals."""
    if ps is None:
        ps = np.linspace(0.05, 0.95, 19)
    sigma = np.maximum(sigma, 1e-9)
    z = norm.ppf(0.5 + ps / 2.0)          # half-width multiplier
    resid = np.abs(y - mu)
    emp = np.array([(resid <= zz * sigma).mean() for zz in z])
    return np.asarray(ps), emp


def ece(y, mu, sigma, ps=None):
    """Expected calibration error = mean |empirical - nominal| coverage."""
    ps, emp = coverage_curve(y, mu, sigma, ps)
    return float(np.mean(np.abs(emp - ps)))


def miscalibration_area(y, mu, sigma, ps=None):
    """Area between the reliability curve and the diagonal (trapezoid)."""
    ps, emp = coverage_curve(y, mu, sigma, ps)
    return float(np.trapz(np.abs(emp - ps), ps))


def spearman_sigma_error(y, mu, sigma):
    """Spearman rho between predicted sigma and actual |error|."""
    resid = np.abs(y - mu)
    if np.allclose(sigma, sigma[0]):
        return 0.0
    rho, _ = stats.spearmanr(sigma, resid)
    return float(rho)


def recalibrate_sigma(y_val, mu_val, sigma_val, grid=None):
    """Pick scalar s minimizing ECE of s*sigma on validation. Returns s."""
    if grid is None:
        grid = np.linspace(0.25, 4.0, 61)
    best_s, best_e = 1.0, float("inf")
    for s in grid:
        e = ece(y_val, mu_val, s * sigma_val)
        if e < best_e:
            best_e, best_s = e, s
    return float(best_s)


def regression_metrics(y, mu):
    """R^2 and MAE."""
    ss_res = float(np.sum((y - mu) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(y - mu)))
    return {"r2": r2, "mae": mae}


def calibration_report(y, mu, sigma, ps=None):
    return {
        "ece": ece(y, mu, sigma, ps),
        "miscalibration_area": miscalibration_area(y, mu, sigma, ps),
        "spearman_sigma_error": spearman_sigma_error(y, mu, sigma),
        **regression_metrics(y, mu),
    }
