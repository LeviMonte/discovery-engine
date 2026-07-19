#!/usr/bin/env python3
"""Checkpointed runner: processes the experiment as a persisted work queue.

Every task (a calibration analysis, or one (domain, strategy, seed) discovery
loop) is executed and its result written to results/partial/. Each invocation
works until a wall-clock budget is reached, then exits. Re-invoke until it
prints ALL_DONE, at which point metrics.json + figures are assembled.

This exists because the sandbox freezes background processes between calls, so
long runs must be split across many short foreground invocations.

  python run_chunked.py [--smoke] [--seeds N] [--budget SECONDS]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from discovery.acquire import STRATEGIES  # noqa: E402
from discovery.data import load_domain  # noqa: E402
from discovery.experiments import (  # noqa: E402
    aggregate_calibration, aggregate_discovery, run_calibration, simulate_curve,
)
from discovery.featurize import Featurized, featurize  # noqa: E402
from discovery.plots import make_all_figures  # noqa: E402
from discovery.utils import (  # noqa: E402
    REPO_ROOT, RESULTS_DIR, apply_smoke, load_config, set_all_seeds,
)

CONFIGS = {
    "superconductor": REPO_ROOT / "config" / "superconductor.yaml",
    "molecular": REPO_ROOT / "config" / "molecular.yaml",
}
CACHE = RESULTS_DIR / "cache"
PARTIAL = RESULTS_DIR / "partial"


def get_cfg(domain, smoke, seeds):
    cfg = load_config(CONFIGS[domain])
    if smoke:
        cfg = apply_smoke(cfg)
        n = cfg.get("_smoke_seeds", 2)
    else:
        n = seeds
    return cfg, list(range(n))


def ensure_featurized(domain, cfg):
    CACHE.mkdir(parents=True, exist_ok=True)
    npz = CACHE / f"{domain}.npz"
    meta = CACHE / f"{domain}.meta.json"
    if npz.exists() and meta.exists():
        z = np.load(npz, allow_pickle=True)
        scaf = z["scaffolds"] if "scaffolds" in z.files and z["scaffolds"].size else None
        feat = Featurized(X=z["X"], y=z["y"], domain=domain,
                          std_cols=z["std_cols"], scaffolds=scaf)
        m = json.load(open(meta))
        return feat, m
    raw = load_domain(domain, cfg)
    feat = featurize(raw, cfg)
    np.savez_compressed(
        npz, X=feat.X, y=feat.y, std_cols=feat.std_cols,
        scaffolds=feat.scaffolds if feat.scaffolds is not None else np.array([]),
    )
    m = {"dataset": raw.name, "target": raw.target_name, "units": raw.units,
         "n_rows_total": int(raw.n), "n_rows_featurized": int(len(feat.y))}
    json.dump(m, open(meta, "w"))
    return feat, m


def task_list(domains, cfgs):
    tasks = []
    for dom in domains:
        for sd in cfgs[dom][1]:
            tasks.append(("calib", dom, None, sd))
        for strat in STRATEGIES:
            for sd in cfgs[dom][1]:
                tasks.append(("disc", dom, strat, sd))
    return tasks


def partial_path(kind, dom, strat, sd):
    if kind == "calib":
        return PARTIAL / f"calib_{dom}_{sd}.json"
    return PARTIAL / f"disc_{dom}_{strat}_{sd}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=float, default=34.0)
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    PARTIAL.mkdir(parents=True, exist_ok=True)

    domains = ["superconductor", "molecular"]
    cfgs = {d: get_cfg(d, args.smoke, args.seeds) for d in domains}
    tasks = task_list(domains, cfgs)

    t0 = time.time()
    done_now = 0
    total = len(tasks)
    already = sum(1 for t in tasks if partial_path(*t).exists())

    if not args.assemble_only:
        for kind, dom, strat, sd in tasks:
            pp = partial_path(kind, dom, strat, sd)
            if pp.exists():
                continue
            if time.time() - t0 > args.budget:
                print(f"PROGRESS {already + done_now}/{total} "
                      f"(elapsed {time.time()-t0:.1f}s) -- re-invoke to continue")
                return
            cfg, seeds = cfgs[dom]
            feat, _ = ensure_featurized(dom, cfg)
            if kind == "calib":
                res = run_calibration(feat, cfg, seed=sd)
            else:
                res = simulate_curve(feat, cfg, strat, sd)
            json.dump(res, open(pp, "w"), default=float)
            done_now += 1
            print(f"  done {kind} {dom} {strat} seed={sd} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    # all tasks present? assemble
    missing = [t for t in tasks if not partial_path(*t).exists()]
    if missing:
        print(f"PROGRESS {total-len(missing)}/{total} -- re-invoke to continue")
        return

    all_results = {"_meta": {"smoke": args.smoke, "seeds": args.seeds}}
    for dom in domains:
        cfg, seeds = cfgs[dom]
        _, meta = ensure_featurized(dom, cfg)
        calib = aggregate_calibration(
            [json.load(open(partial_path("calib", dom, None, sd))) for sd in seeds])
        curves = {strat: [json.load(open(partial_path("disc", dom, strat, sd)))
                          for sd in seeds] for strat in STRATEGIES}
        disc = aggregate_discovery(curves)
        all_results[dom] = {
            "domain": dom, **meta, "seeds": seeds,
            "config": cfg, "calibration": calib, "discovery": disc,
        }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    figs = make_all_figures(all_results)
    print("ALL_DONE")
    print(f"metrics -> {RESULTS_DIR/'metrics.json'}")
    for f in figs:
        print(f"figure -> {f}")


if __name__ == "__main__":
    main()
