#!/usr/bin/env python3
"""Phase 8.0 -- export the trained superconductor deep ensemble to ONNX.

The Phase 0-7 pipeline trained ensembles in-memory; here we retrain one ensemble
per exportable domain on the SAME (seed-0) train split used for calibration, wrap
the M=5 members into a single module that returns per-member (mu, logvar) in the
standardized target space, and export to ONNX for genuine in-browser inference via
onnxruntime-web. Inverse target-scaling + aleatoric/epistemic decomposition are
done in the frontend from the stats saved in model_meta.json.

Parity: PyTorch vs onnxruntime outputs are checked to < 1e-4 relative error on
held-out points, per member.

Superconductor uses this live-ONNX path. The molecular (fingerprint) domain uses
the precomputed curated library from build_grid.py instead -- see WEBSITE.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discovery.data import load_domain          # noqa: E402
from discovery.featurize import Standardizer, featurize, make_split  # noqa: E402
from discovery.models import DeepEnsemble        # noqa: E402
from discovery.utils import load_config, set_all_seeds  # noqa: E402

OUT = ROOT / "web" / "predict" / "models"


class EnsembleExport(nn.Module):
    """Wraps M trained members; returns stacked (mu, logvar) in standardized space.

    Input:  x  (batch, n_features)  -- already standardized
    Output: mu (batch, M), logvar (batch, M)
    """

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList([m for m, _ in members])

    def forward(self, x):
        mus, lvs = [], []
        for m in self.members:
            mu, lv = m(x)
            mus.append(mu.unsqueeze(-1))
            lvs.append(lv.unsqueeze(-1))
        return torch.cat(mus, dim=-1), torch.cat(lvs, dim=-1)


def _slider_features(X_raw, y, feature_names, k=8):
    """Pick the k most important composition features (RandomForest importances)."""
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=120, max_depth=14, n_jobs=-1,
                               random_state=0)
    rf.fit(X_raw, y)
    order = np.argsort(-rf.feature_importances_)[:k]
    return [int(i) for i in order], rf.feature_importances_


def _find_preset(raw, feature_names, feat_median, must_contain):
    """Return a real dataset row (81 raw feats) whose formula contains all
    `must_contain` element symbols; fall back to median if none found."""
    try:
        import pandas as pd
        um = pd.read_csv(ROOT / "data" / "superconductor" / "unique_m.csv")
        mask = np.ones(len(um), dtype=bool)
        for el in must_contain:
            # element column present and > 0
            if el in um.columns:
                mask &= um[el].to_numpy() > 0
        idx = np.where(mask)[0]
        if len(idx):
            # pick the highest-Tc match
            best = idx[np.argmax(um["critical_temp"].to_numpy()[idx])]
            return raw.table.iloc[int(best)].to_numpy(dtype=float).tolist(), \
                float(um["critical_temp"].to_numpy()[best]), \
                str(um["material"].to_numpy()[best])
    except Exception as e:  # noqa: BLE001
        print("  preset lookup failed:", e)
    return list(feat_median), None, "median composition"


def export_superconductor():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config" / "superconductor.yaml")
    set_all_seeds(0)
    raw = load_domain("superconductor", cfg)
    feat = featurize(raw, cfg)
    train_idx, test_idx = make_split(feat, cfg, seed=0)

    std = Standardizer.fit(feat.X, train_idx, cols=feat.std_cols)
    Xz = std.transform(feat.X)

    print("training M=5 ensemble for export ...", flush=True)
    ens = DeepEnsemble(cfg.get("model", {}), base_seed=0)
    ens.fit(Xz[train_idx], feat.y[train_idx])

    # target scaler is per-member; they share (almost) identical stats since the
    # target standardization uses the member's bootstrap. Use member-0's scaler
    # stats as the reported target scale, but export raw std-space and let JS use
    # each member's stored scaler for exactness.
    tmeans = [float(sc.mean) for _, sc in ens.members]
    tstds = [float(sc.std) for _, sc in ens.members]

    wrapper = EnsembleExport(ens.members).eval()
    dummy = torch.tensor(Xz[test_idx][:1], dtype=torch.float32)
    onnx_path = OUT / "superconductor_ensemble.onnx"
    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=["x_std"], output_names=["mu_std", "logvar_std"],
        dynamic_axes={"x_std": {0: "batch"}, "mu_std": {0: "batch"},
                      "logvar_std": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print("exported ->", onnx_path)

    # ---- parity check: torch vs onnxruntime on 10 test points ----
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    xp = Xz[test_idx][:10].astype(np.float32)
    with torch.no_grad():
        mu_t, lv_t = wrapper(torch.tensor(xp))
    mu_o, lv_o = sess.run(None, {"x_std": xp})
    rel = lambda a, b: float(np.max(np.abs(a - b) / (np.abs(b) + 1e-6)))
    r_mu, r_lv = rel(mu_o, mu_t.numpy()), rel(lv_o, lv_t.numpy())
    print(f"parity: max rel err  mu={r_mu:.2e}  logvar={r_lv:.2e}")
    assert r_mu < 1e-4 and r_lv < 1e-4, "ONNX parity check FAILED"

    # ---- cache core stats (fast) so build_meta.py can finish metadata without
    #      retraining. Training is deterministic (seed 0) so this matches the ONNX.
    Xtr = feat.X[train_idx]
    cache = {
        "target_means": tmeans, "target_stds": tstds,
        "feat_mean": [float(v) for v in std.mean],
        "feat_std": [float(v) for v in std.std],
        "feat_min": [float(v) for v in np.min(Xtr, axis=0)],
        "feat_max": [float(v) for v in np.max(Xtr, axis=0)],
        "feat_median": [float(v) for v in np.median(Xtr, axis=0)],
        "n_members": len(ens.members),
        "feature_names": feat.feature_names,
        "train_tc_range": [float(feat.y[train_idx].min()),
                           float(feat.y[train_idx].max())],
    }
    (OUT / "_export_cache.json").write_text(json.dumps(cache))
    print("cached core stats -> _export_cache.json")
    print("SUPERCONDUCTOR ONNX + CACHE OK  (run build_meta.py to finish meta)")
    return True


if __name__ == "__main__":
    export_superconductor()
