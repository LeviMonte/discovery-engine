"""Figures: discovery curves (mean +/- 95% CI) and calibration diagrams."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .utils import FIG_DIR  # noqa: E402

_COLORS = {
    "random": "#888888", "greedy": "#1f77b4", "ucb": "#d62728",
    "thompson": "#2ca02c", "ei": "#9467bd", "max_var": "#ff7f0e",
}


def plot_discovery(domain, disc, out_dir: Path = FIG_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    any_strat = next(iter(disc.values()))
    pool = any_strat["pool_size"]
    ntopk = any_strat["n_topk"]
    for strat, d in disc.items():
        x = np.array(d["n_labeled_mean"])
        y = np.array(d["hits_mean"])
        hw = np.array(d["hits_ci95"])
        c = _COLORS.get(strat, None)
        ax.plot(x, y, label=strat, color=c, lw=2)
        ax.fill_between(x, y - hw, y + hw, color=c, alpha=0.15)
    ax.set_xlabel("# materials measured (acquisitions)")
    ax.set_ylabel(f"top-{ntopk} hits found")
    ax.set_title(f"Discovery efficiency -- {domain}\n(pool={pool}, mean +/- 95% CI)")
    ax.axhline(ntopk, ls=":", c="k", alpha=0.4, label="all hits")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / f"discovery_{domain}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_calibration(domain, calib, out_dir: Path = FIG_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    rel = calib["reliability"]
    ps = np.array(rel["nominal"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.plot(ps, rel["empirical_single"], "o-", color="#ff7f0e",
            label=f"single (ECE={calib['single']['ece']:.3f})")
    ax.plot(ps, rel["empirical_ensemble"], "s-", color="#1f77b4",
            label=f"ensemble (ECE={calib['ensemble']['ece']:.3f})")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title(f"Reliability -- {domain}")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")
    fig.tight_layout()
    path = out_dir / f"calibration_{domain}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_all_figures(all_results, out_dir: Path = FIG_DIR):
    paths = []
    for domain, res in all_results.items():
        if domain == "_meta":
            continue
        paths.append(plot_discovery(domain, res["discovery"], out_dir))
        paths.append(plot_calibration(domain, res["calibration"], out_dir))
    return paths
