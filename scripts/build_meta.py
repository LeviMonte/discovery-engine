#!/usr/bin/env python3
"""Phase 8.0 (cont.) -- finish web/predict/models/model_meta.json.

Reads the fast cache written by export_onnx.py (target/feature stats) and adds the
things that need the raw data but NOT the trained network: permutation-free
RandomForest feature importances (to choose the handful of sliders) and real-row
presets (YBCO-like, MgB2-like, a random pool point). Kept separate so the heavy
ensemble training in export_onnx.py doesn't have to fit in one shell budget.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discovery.data import load_domain          # noqa: E402
from discovery.featurize import featurize, make_split  # noqa: E402
from discovery.utils import load_config          # noqa: E402

OUT = ROOT / "web" / "predict" / "models"


def _find_preset(table, um, elements):
    """Highest-Tc real row whose formula contains all `elements`; else None."""
    mask = np.ones(len(um), dtype=bool)
    for el in elements:
        if el in um.columns:
            mask &= um[el].to_numpy() > 0
        else:
            mask &= False
    idx = np.where(mask)[0]
    if not len(idx):
        return None
    tc = um["critical_temp"].to_numpy()
    best = int(idx[np.argmax(tc[idx])])
    return {"values": table.iloc[best].to_numpy(dtype=float).tolist(),
            "true_tc": float(tc[best]), "formula": str(um["material"].to_numpy()[best])}


def main():
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    cache = json.loads((OUT / "_export_cache.json").read_text())
    cfg = load_config(ROOT / "config" / "superconductor.yaml")
    raw = load_domain("superconductor", cfg)
    feat = featurize(raw, cfg)
    train_idx, _ = make_split(feat, cfg, seed=0)
    Xtr, ytr = feat.X[train_idx], feat.y[train_idx]
    names = feat.feature_names

    rf = RandomForestRegressor(n_estimators=60, max_depth=12, n_jobs=-1,
                               random_state=0)
    rf.fit(Xtr, ytr)
    imp = rf.feature_importances_
    slider_idx = [int(i) for i in np.argsort(-imp)[:8]]

    def _mk(i):
        col = Xtr[:, i]
        lo, hi = float(np.percentile(col, 1)), float(np.percentile(col, 99))
        if hi <= lo:
            hi = lo + 1.0
        return {"index": i, "name": names[i], "min": round(lo, 4),
                "max": round(hi, 4), "step": round((hi - lo) / 100.0, 5),
                "importance": float(imp[i])}

    sliders = [_mk(i) for i in slider_idx]

    um = pd.read_csv(ROOT / "data" / "superconductor" / "unique_m.csv")
    presets = {}
    ybco = _find_preset(raw.table, um, ["Y", "Ba", "Cu", "O"])
    mgb2 = _find_preset(raw.table, um, ["Mg", "B"])
    if ybco:
        presets["YBCO-like (cuprate)"] = ybco
    if mgb2:
        presets["MgB2-like"] = mgb2
    rng = np.random.RandomState(7)
    rp = int(rng.choice(len(train_idx)))
    presets["Random training point"] = {
        "values": Xtr[rp].astype(float).tolist(),
        "true_tc": float(ytr[rp]), "formula": "random training composition"}

    meta = {
        "domain": "superconductor",
        "inference": "live-onnx",
        "onnx": "superconductor_ensemble.onnx",
        "target": {
            "name": "critical temperature (Tc)", "units": "K",
            "note": "This is Tc *assuming the compound superconducts*. The model "
                    "was trained only on known superconductors and CANNOT tell you "
                    "whether an arbitrary composition superconducts at all.",
        },
        "n_features": len(names),
        "feature_names": names,
        "feat_mean": cache["feat_mean"],
        "feat_std": cache["feat_std"],
        "feat_min": cache["feat_min"],
        "feat_max": cache["feat_max"],
        "feat_median": cache["feat_median"],
        "target_means": cache["target_means"],
        "target_stds": cache["target_stds"],
        "n_members": cache["n_members"],
        "train_tc_range": cache["train_tc_range"],
        "sliders": sliders,
        "presets": presets,
    }
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote model_meta.json")
    print("  sliders:", [s["name"] for s in sliders])
    print("  presets:", {k: round(v["true_tc"], 1) for k, v in presets.items()})


if __name__ == "__main__":
    main()
