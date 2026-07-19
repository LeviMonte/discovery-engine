#!/usr/bin/env python3
"""CLI for the Uncertainty-Driven Materials Discovery Engine.

  python run.py --domain {superconductor,molecular,all} --seeds 5 [--smoke]

Runs the calibration analysis + active-learning discovery simulation for each
domain, writes results/metrics.json and results/figures/*.png.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from discovery.experiments import run_experiment, save_metrics  # noqa: E402
from discovery.plots import make_all_figures  # noqa: E402
from discovery.utils import (  # noqa: E402
    REPO_ROOT, apply_smoke, load_config,
)

CONFIGS = {
    "superconductor": REPO_ROOT / "config" / "superconductor.yaml",
    "molecular": REPO_ROOT / "config" / "molecular.yaml",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["superconductor", "molecular", "all"],
                    default="all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    domains = (["superconductor", "molecular"] if args.domain == "all"
               else [args.domain])

    all_results = {}
    for dom in domains:
        cfg = load_config(CONFIGS[dom])
        if args.smoke:
            cfg = apply_smoke(cfg)
            n_seeds = cfg.get("_smoke_seeds", 2)
        else:
            n_seeds = args.seeds
        seeds = list(range(n_seeds))
        t0 = time.time()
        print(f"\n=== {dom} | seeds={seeds} | smoke={args.smoke} ===", flush=True)
        res = run_experiment(dom, cfg, seeds)
        all_results[dom] = res
        cal = res["calibration"]
        print(f"  [{dom}] test R2 single={cal['single']['r2']:.3f} "
              f"ensemble={cal['ensemble']['r2']:.3f} | "
              f"ECE single={cal['single']['ece']:.3f} "
              f"ensemble={cal['ensemble']['ece']:.3f}", flush=True)
        for strat, d in res["discovery"].items():
            print(f"    {strat:9s} AUDC={d['audc_mean']:.3f} "
                  f"enrich={d['enrichment_mean']:.2f} "
                  f"leak_ok={d['leakage_ok']}", flush=True)
        print(f"  [{dom}] done in {time.time()-t0:.1f}s", flush=True)

    all_results["_meta"] = {"smoke": args.smoke, "seeds": args.seeds}
    # smoke writes to a separate file so it never clobbers a full run's outputs
    fname = "metrics_smoke.json" if args.smoke else "metrics.json"
    path = save_metrics(all_results, fname=fname)
    figs = [] if args.smoke else make_all_figures(all_results)
    print(f"\nSaved metrics -> {path}")
    for f in figs:
        print(f"Saved figure -> {f}")


if __name__ == "__main__":
    main()
