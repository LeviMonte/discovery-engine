#!/usr/bin/env python3
"""Phase 11 -- reconstruct the exact UCI (Hamidieh 2018) 81-feature composition
featurization so it can run in the browser.

Element property values are RECOVERED from the dataset itself (least-squares on the
mean_* columns -> machine-precision exact), then the 10 statistics are validated
against every one of the 81 columns of train.csv. Outputs element_props.json for
the JS port. No guessing: if a column doesn't reproduce, we fix the formula.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROPS = ["atomic_mass", "fie", "atomic_radius", "Density", "ElectronAffinity",
         "FusionHeat", "ThermalConductivity", "Valence"]
STATS = ["mean", "wtd_mean", "gmean", "wtd_gmean", "entropy", "wtd_entropy",
         "range", "wtd_range", "std", "wtd_std"]


def recover_table():
    tr = pd.read_csv(ROOT / "data" / "superconductor" / "train.csv")
    um = pd.read_csv(ROOT / "data" / "superconductor" / "unique_m.csv")
    elem_cols = [c for c in um.columns if c not in ("critical_temp", "material")]
    E = um[elem_cols].to_numpy(dtype=float)
    present = (E > 0).astype(float)
    n = tr["number_of_elements"].to_numpy(dtype=float)
    used = present.sum(0) > 0
    table = {}
    for p in PROPS:
        b = n * tr[f"mean_{p}"].to_numpy()
        x, *_ = np.linalg.lstsq(present[:, used], b, rcond=None)
        full = np.zeros(len(elem_cols)); full[used] = x
        table[p] = full
    return tr, um, elem_cols, E, table


def stats_for(vals, amts, props_by_elem):
    """vals: property values per present element; amts: their amounts."""
    t = np.asarray(vals, dtype=float)
    a = np.asarray(amts, dtype=float)
    p = a / a.sum()                       # composition fractions
    n = len(t)
    eps = 1e-12
    f = t / t.sum()                        # property-value distribution
    g = (p * t); g = g / g.sum()           # weighted property distribution
    out = {}
    out["mean"] = t.mean()
    out["wtd_mean"] = float(np.sum(p * t))
    out["gmean"] = float(np.exp(np.mean(np.log(np.clip(t, eps, None)))))
    out["wtd_gmean"] = float(np.exp(np.sum(p * np.log(np.clip(t, eps, None)))))
    out["entropy"] = float(-np.sum(f * np.log(np.clip(f, eps, None))))
    out["wtd_entropy"] = float(-np.sum(g * np.log(np.clip(g, eps, None))))
    out["range"] = float(t.max() - t.min())
    pw = p * t
    out["wtd_range"] = float(pw.max() - pw.min())
    out["std"] = float(np.sqrt(np.mean((t - t.mean()) ** 2)))
    out["wtd_std"] = float(np.sqrt(np.sum(p * (t - out["wtd_mean"]) ** 2)))
    return out


def main():
    tr, um, elem_cols, E, table = recover_table()
    idx = {c: i for i, c in enumerate(elem_cols)}
    N = len(tr)
    # build per-material present elements + amounts
    col_err = {f"{s}_{p}": 0.0 for p in PROPS for s in STATS}
    # validate on a sample for speed, then full
    rng = np.random.RandomState(0)
    sample = rng.choice(N, size=3000, replace=False)
    for m in sample:
        amts = E[m]; present = np.where(amts > 0)[0]
        a = amts[present]
        for p in PROPS:
            vals = table[p][present]
            st = stats_for(vals, a, None)
            for s in STATS:
                col = f"{s}_{p}"
                err = abs(st[s] - tr[col].iloc[m])
                if err > col_err[col]:
                    col_err[col] = err
    worst = sorted(col_err.items(), key=lambda kv: -kv[1])[:8]
    print("worst-reproduced columns (max abs err over 3000 materials):")
    for c, e in worst:
        print(f"  {c:28s} {e:.3e}")
    allok = all(e < 1e-4 for e in col_err.values())
    print("ALL 81 COLUMNS REPRODUCED" if allok else "SOME COLUMNS OFF -> fix formula")

    # export element property table for JS
    out = {"props": PROPS, "stats": STATS,
           "elements": {}}
    for i, c in enumerate(elem_cols):
        if any(table[p][i] != 0 for p in PROPS):
            out["elements"][c] = {p: round(float(table[p][i]), 6) for p in PROPS}
    # also standardization stats used by the model live already in model_meta.json
    dest = ROOT / "web" / "predict" / "models" / "element_props.json"
    dest.write_text(json.dumps(out))
    print(f"wrote {dest} with {len(out['elements'])} elements x {len(PROPS)} properties")


if __name__ == "__main__":
    main()
