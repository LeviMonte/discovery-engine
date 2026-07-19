#!/usr/bin/env python3
"""Phase 8.1 -- web/replay/data/discovery_runs.json from results/metrics.json.

Pure transform: no new experiments, no new randomness. The curves, CI bands, and
headline metrics are copied straight from the Phase-5 metrics.json so the replay
page is numerically identical to the saved static figures.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "metrics.json"
OUT = ROOT / "web" / "replay" / "data" / "discovery_runs.json"

STRATEGIES = ["random", "greedy", "ucb", "thompson", "ei", "max_var"]


def main():
    m = json.loads(SRC.read_text())
    out = {"_meta": {"source": "results/metrics.json",
                     "note": "Retrospective benchmark on public data where the top "
                             "materials were already known; this replays saved runs, "
                             "it does not run new discovery."},
           "domains": {}}
    for dom in ("superconductor", "molecular"):
        d = m[dom]
        disc = d["discovery"]
        cal = d["calibration"]
        strategies = {}
        for s in STRATEGIES:
            v = disc[s]
            strategies[s] = {
                "n_labeled": v["n_labeled_mean"],
                "hits_mean": v["hits_mean"],
                "hits_ci95": v["hits_ci95"],
                "hits_per_seed": v["hits_per_seed"],
                "enrichment": round(v["enrichment_mean"], 3),
                "enrichment_ci95": round(v["enrichment_ci95"], 3),
                "audc": round(v["audc_mean"], 3),
                "acq_to_half_hits": v["acq_to_half_hits_mean"],
                "hit_ratio_vs_random": (round(v["hit_ratio_vs_random_mean"], 3)
                                        if v["hit_ratio_vs_random_mean"] else None),
            }
        def cget(grp, met):
            s = cal.get("stats", {}).get(f"{grp}.{met}")
            return s["mean"] if s else cal.get(grp, {}).get(met)
        calibration = {
            "reliability": cal["reliability"],
            "single": {"ece": cget("single", "ece"), "r2": cget("single", "r2"),
                       "spearman": cget("single", "spearman_sigma_error")},
            "ensemble": {"ece": cget("ensemble", "ece"), "r2": cget("ensemble", "r2"),
                         "spearman": cget("ensemble", "spearman_sigma_error")},
            "single_recal_ece": cget("single_recal", "ece"),
            "ensemble_recal_ece": cget("ensemble_recal", "ece"),
            "mean_aleatoric": cal.get("mean_aleatoric"),
            "mean_epistemic": cal.get("mean_epistemic"),
            "n_test": cal.get("n_test"), "n_seeds": cal.get("n_seeds"),
        }
        out["domains"][dom] = {
            "dataset": d["dataset"],
            "target": d["target"],
            "units": d["units"],
            "pool_size": disc["greedy"]["pool_size"],
            "n_topk": disc["greedy"]["n_topk"],
            "n_seeds": len(d["seeds"]),
            "acq_to_half_target": disc["greedy"]["acq_to_half_hits_target"],
            "strategies": strategies,
            "calibration": calibration,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    # verification: re-read and confirm final hits match metrics.json
    chk = json.loads(OUT.read_text())
    for dom in ("superconductor", "molecular"):
        for s in STRATEGIES:
            a = chk["domains"][dom]["strategies"][s]["hits_mean"][-1]
            b = m[dom]["discovery"][s]["hits_mean"][-1]
            assert abs(a - b) < 1e-9, (dom, s, a, b)
    print(f"wrote {OUT} (verified final hits match metrics.json for all "
          f"{len(STRATEGIES)} strategies x 2 domains)")


if __name__ == "__main__":
    main()
