"""Multi-seed orchestration: calibration analysis + discovery simulation.

Produces results/metrics.json with, per domain:
  * calibration: single-model vs ensemble ECE / miscalibration area / Spearman,
    plus reliability curves and a post-hoc recalibration factor.
  * discovery: per-strategy mean +/- 95% CI discovery curves, AUDC,
    enrichment-vs-random, and acquisitions-to-K-hits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import uncertainty as unc
from .acquire import STRATEGIES
from .active_loop import (
    acquisitions_to_k_hits, audc, enrichment_factor, hit_ratio_vs_random,
    run_active_learning,
)
from .data import load_domain
from .featurize import featurize, make_split
from .models import DeepEnsemble
from .utils import RESULTS_DIR, set_all_seeds


def _ci95(rows):
    """rows: (n_seeds, T). Return mean, half-width of 95% CI per column."""
    a = np.asarray(rows, dtype=float)
    mean = a.mean(axis=0)
    if a.shape[0] < 2:
        return mean, np.zeros_like(mean)
    sd = a.std(axis=0, ddof=1)
    hw = 1.96 * sd / np.sqrt(a.shape[0])
    return mean, hw


# --------------------------------------------------------------------------- #
#  Calibration analysis (single vs ensemble)
# --------------------------------------------------------------------------- #
def run_calibration(feat, cfg, seed=0):
    """One-seed calibration comparison.

    The 'single' baseline is a FAIR standalone heteroscedastic net trained on the
    full fit set (no bootstrap) -- not an ensemble member -- so the single-vs-
    ensemble comparison is not biased in the ensemble's favour. BOTH the single
    net and the ensemble are recalibrated symmetrically on a held-out validation
    slice, so any calibration gain is not an artefact of only recalibrating one.
    """
    from .models import TargetScaler, HeteroscedasticMLP, train_one
    import torch

    set_all_seeds(seed)
    train_idx, test_idx = make_split(feat, cfg, seed=seed)
    # held-out validation slice out of TRAIN for post-hoc recalibration
    # (never touches test; neither model trains on it).
    rng = np.random.RandomState(seed + 777)
    perm = rng.permutation(train_idx)
    n_val = max(1, int(round(0.15 * len(perm))))
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]

    Xz, _ = feat.standardized(fit_idx)
    mcfg = cfg.get("model", {})
    ps = np.linspace(0.05, 0.95, 19)
    y_test = feat.y[test_idx]

    # --- ensemble (M members, bootstrap for diversity) ---
    ens = DeepEnsemble(mcfg, base_seed=seed)
    ens.fit(Xz[fit_idx], feat.y[fit_idx])
    pred_e = ens.predict(Xz[test_idx])
    pred_e_val = ens.predict(Xz[val_idx])

    # --- fair single net: trained on the FULL fit set, no bootstrap ---
    single, sc = train_one(Xz[fit_idx], feat.y[fit_idx], mcfg, seed=seed)

    def _single_pred(idx):
        with torch.no_grad():
            mu, lv = single(torch.tensor(Xz[idx], dtype=torch.float32))
        mu = sc.inv_mean(mu.numpy())
        var = sc.inv_var(np.exp(lv.numpy()))
        return mu, np.sqrt(var)

    mu_s, sig_s = _single_pred(test_idx)
    mu_s_val, sig_s_val = _single_pred(val_idx)

    rep_e = unc.calibration_report(y_test, pred_e.mu, pred_e.sigma, ps)
    rep_s = unc.calibration_report(y_test, mu_s, sig_s, ps)

    # symmetric recalibration on the validation slice
    s_e = unc.recalibrate_sigma(feat.y[val_idx], pred_e_val.mu, pred_e_val.sigma)
    s_s = unc.recalibrate_sigma(feat.y[val_idx], mu_s_val, sig_s_val)
    rep_e_recal = unc.calibration_report(y_test, pred_e.mu, s_e * pred_e.sigma, ps)
    rep_s_recal = unc.calibration_report(y_test, mu_s, s_s * sig_s, ps)

    _, emp_e = unc.coverage_curve(y_test, pred_e.mu, pred_e.sigma, ps)
    _, emp_s = unc.coverage_curve(y_test, mu_s, sig_s, ps)

    return {
        "n_test": int(len(test_idx)),
        "single": rep_s,
        "single_recal": rep_s_recal,
        "ensemble": rep_e,
        "ensemble_recal": rep_e_recal,
        "recal_factor": s_e,
        "recal_factor_single": s_s,
        "reliability": {
            "nominal": ps.tolist(),
            "empirical_single": emp_s.tolist(),
            "empirical_ensemble": emp_e.tolist(),
        },
        "mean_aleatoric": float(pred_e.aleatoric.mean()),
        "mean_epistemic": float(pred_e.epistemic.mean()),
    }


def aggregate_calibration(per_seed):
    """Aggregate single-seed calibration dicts into mean +/- 95% CI.

    per_seed: list of run_calibration outputs (one per seed). The reliability
    curve for the figure is taken from the first seed; scalar metrics get CIs.
    """
    keys = [("single", "ece"), ("single", "r2"), ("single", "spearman_sigma_error"),
            ("ensemble", "ece"), ("ensemble", "r2"), ("ensemble", "spearman_sigma_error"),
            ("single_recal", "ece"), ("ensemble_recal", "ece"),
            ("ensemble", "mae"), ("single", "mae")]
    out = {k: dict(v) for k, v in per_seed[0].items()
           if k in ("single", "ensemble", "single_recal", "ensemble_recal")}
    out["n_test"] = per_seed[0]["n_test"]
    out["reliability"] = per_seed[0]["reliability"]
    out["n_seeds"] = len(per_seed)
    stats = {}
    for grp, met in keys:
        vals = [d[grp][met] for d in per_seed]
        m = float(np.mean(vals))
        ci = float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        stats[f"{grp}.{met}"] = {"mean": m, "ci95": ci}
        out[grp][met] = m          # overwrite point value with seed-mean
    out["stats"] = stats
    for fld in ("recal_factor", "recal_factor_single", "mean_aleatoric", "mean_epistemic"):
        out[fld] = float(np.mean([d[fld] for d in per_seed]))
    return out


# --------------------------------------------------------------------------- #
#  Discovery simulation across strategies x seeds
# --------------------------------------------------------------------------- #
def aggregate_discovery(curves):
    """Aggregate per-seed loop curves into per-strategy stats + derived metrics.

    curves: {strategy: [ {hits, n_labeled, pool_size, n_topk, leakage_ok}, ... ]}
    """
    per_strategy = {}
    for strat, seed_runs in curves.items():
        curves_hits = [r["hits"] for r in seed_runs]
        curves_nl = [r["n_labeled"] for r in seed_runs]
        pool_size = seed_runs[0]["pool_size"]
        n_topk = seed_runs[0]["n_topk"]
        leak = all(r["leakage_ok"] for r in seed_runs)
        L = min(len(c) for c in curves_hits)
        hits = [c[:L] for c in curves_hits]
        nls = [c[:L] for c in curves_nl]
        mean_h, hw_h = _ci95(hits)
        mean_nl = np.mean(nls, axis=0)
        per_strategy[strat] = {
            "n_labeled_mean": mean_nl.tolist(),
            "hits_mean": mean_h.tolist(),
            "hits_ci95": hw_h.tolist(),
            "hits_per_seed": [list(map(int, c)) for c in hits],
            "pool_size": int(pool_size),
            "n_topk": int(n_topk),
            "leakage_ok": bool(leak),
            "audc_mean": float(np.mean([audc(nl, h, pool_size, n_topk)
                                        for nl, h in zip(nls, hits)])),
        }
    # derived: enrichment (base-rate) + hit-ratio-vs-random + acquisitions-to-K
    rand_hits = np.array(per_strategy["random"]["hits_per_seed"])
    for strat, d in per_strategy.items():
        sh = np.array(d["hits_per_seed"])
        nl_mean = d["n_labeled_mean"]
        pool, ntopk = d["pool_size"], d["n_topk"]
        enr = [enrichment_factor(sh[i][-1], nl_mean[-1], pool, ntopk)
               for i in range(len(sh))]
        d["enrichment_mean"] = float(np.mean(enr))
        d["enrichment_ci95"] = float(
            1.96 * np.std(enr, ddof=1) / np.sqrt(len(enr)) if len(enr) > 1 else 0.0
        )
        ratios = [hit_ratio_vs_random(sh[i], rand_hits[i]) for i in range(len(sh))]
        ratios = [r for r in ratios if np.isfinite(r)]
        d["hit_ratio_vs_random_mean"] = float(np.mean(ratios)) if ratios else None
        K_target = max(1, int(round(0.5 * ntopk)))  # find half the hits
        a2k = [acquisitions_to_k_hits(nl_mean, sh[i], K_target)
               for i in range(len(sh))]
        finite = [v for v in a2k if np.isfinite(v)]
        d["acq_to_half_hits_mean"] = float(np.mean(finite)) if finite else None
        d["acq_to_half_hits_target"] = K_target
    return per_strategy


def simulate_curve(feat, cfg, strategy, seed):
    """Run one (strategy, seed) loop and return a serializable curve dict."""
    set_all_seeds(seed)
    r = run_active_learning(feat, cfg, strategy, seed)
    return {
        "hits": [int(x) for x in r.hits_found],
        "n_labeled": [int(x) for x in r.n_labeled],
        "pool_size": int(r.pool_size),
        "n_topk": int(r.n_topk),
        "leakage_ok": bool(r.leakage_ok),
        "leakage_notes": r.leakage_notes,
    }


def run_discovery(feat, cfg, seeds):
    curves = {strat: [simulate_curve(feat, cfg, strat, sd) for sd in seeds]
              for strat in STRATEGIES}
    return aggregate_discovery(curves)


# --------------------------------------------------------------------------- #
#  Top-level per-domain experiment
# --------------------------------------------------------------------------- #
def run_experiment(domain, cfg, seeds, results_dir: Path = RESULTS_DIR):
    raw = load_domain(domain, cfg)
    feat = featurize(raw, cfg)
    out = {
        "domain": domain,
        "dataset": raw.name,
        "target": raw.target_name,
        "units": raw.units,
        "n_rows_total": int(raw.n),
        "n_rows_featurized": int(len(feat.y)),
        "seeds": list(seeds),
        "config": cfg,
        "calibration": aggregate_calibration(
            [run_calibration(feat, cfg, seed=sd) for sd in seeds]),
        "discovery": run_discovery(feat, cfg, seeds),
    }
    return out


def save_metrics(all_results: dict, results_dir: Path = RESULTS_DIR,
                 fname: str = "metrics.json"):
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / fname
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    return path
