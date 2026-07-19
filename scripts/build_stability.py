#!/usr/bin/env python3
"""Phase 12 -- the VERIFICATION LAYER: a composition -> formation-energy model.

Trains a heteroscedastic deep ensemble to predict experimental formation enthalpy
(eV/atom) from composition, reusing the EXACT 81-feature UCI featurizer (so it runs
in-browser via scfeat.js). Lower / negative formation energy = more thermodynamically
favourable. This is a real first-pass stability screen -- an honest proxy for
energy-above-hull, not a substitute for it -- that lets generated candidates be
filtered by "is this even plausible to form", the AlphaFold-style verification move.

Data: matminer experimental formation-enthalpy datasets (real measured values).
Output: web/predict/models/stability_ensemble.onnx + stability_meta.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_sc_featurizer import recover_table, stats_for, PROPS, STATS  # noqa: E402
from discovery.featurize import Standardizer                            # noqa: E402
from discovery.models import DeepEnsemble                               # noqa: E402
from discovery.utils import set_all_seeds                               # noqa: E402
from export_onnx import EnsembleExport                                  # noqa: E402

OUT = ROOT / "web" / "predict" / "models"
FEATURE_NAMES = ["number_of_elements"] + [f"{s}_{p}" for p in PROPS for s in STATS]


def load_data(table, idx, elem_set):
    from matminer.datasets import load_dataset
    from pymatgen.core import Composition
    rows = {}
    for name, fcol, ecol in [
        ("expt_formation_enthalpy_kingsbury", "formula", "expt_form_e"),
        ("expt_formation_enthalpy", "formula", "e_form expt"),
    ]:
        try:
            df = load_dataset(name)
        except Exception as e:  # noqa: BLE001
            print("  skip", name, e); continue
        for _, r in df.iterrows():
            try:
                comp = Composition(str(r[fcol])).get_el_amt_dict()
            except Exception:
                continue
            els = list(comp.keys())
            if not els or any(e not in elem_set for e in els):
                continue
            y = float(r[ecol])
            if not np.isfinite(y):
                continue
            key = "".join(sorted(f"{e}{comp[e]:.3f}" for e in els))
            rows[key] = (comp, y)          # dedupe by composition
    X, Y = [], []
    for comp, y in rows.values():
        els = list(comp.keys()); amts = [comp[e] for e in els]
        m = {"number_of_elements": len(els)}
        for p in PROPS:
            st = stats_for([table[p][idx[e]] for e in els], amts, None)
            for s in STATS:
                m[f"{s}_{p}"] = st[s]
        X.append([m[c] for c in FEATURE_NAMES]); Y.append(y)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float64)


def main():
    set_all_seeds(0)
    tr, um, elem_cols, E, table = recover_table()
    idx = {c: i for i, c in enumerate(elem_cols)}
    elem_set = {c for c in elem_cols if any(table[p][idx[c]] != 0 for p in PROPS)}
    print(f"element table covers {len(elem_set)} elements")

    X, Y = load_data(table, idx, elem_set)
    print(f"training set: {len(Y)} compositions | e_form range "
          f"[{Y.min():.2f}, {Y.max():.2f}] eV/atom, mean {Y.mean():.2f}")

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(Y))
    ntest = int(0.15 * len(Y))
    test, train = perm[:ntest], perm[ntest:]
    std = Standardizer.fit(X, train, cols=np.ones(X.shape[1], dtype=bool))
    Xz = std.transform(X)

    cfg = {"hidden": [256, 256, 128], "ensemble_size": 5, "epochs": 200,
           "patience": 20, "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 128}
    ens = DeepEnsemble(cfg, base_seed=0).fit(Xz[train], Y[train])
    pred = ens.predict(Xz[test])
    ss = 1 - np.sum((Y[test] - pred.mu) ** 2) / np.sum((Y[test] - Y[test].mean()) ** 2)
    mae = float(np.mean(np.abs(Y[test] - pred.mu)))
    print(f"held-out formation-energy model: R^2={ss:.3f} MAE={mae:.3f} eV/atom")

    wrapper = EnsembleExport(ens.members).eval()
    dummy = torch.tensor(Xz[test][:1], dtype=torch.float32)
    onnx_path = OUT / "stability_ensemble.onnx"
    torch.onnx.export(wrapper, dummy, str(onnx_path), input_names=["x_std"],
                      output_names=["mu_std", "logvar_std"],
                      dynamic_axes={"x_std": {0: "batch"}, "mu_std": {0: "batch"},
                                    "logvar_std": {0: "batch"}},
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    xp = Xz[test][:10].astype(np.float32)
    with torch.no_grad():
        mu_t, _ = wrapper(torch.tensor(xp))
    mu_o, _ = sess.run(None, {"x_std": xp})
    assert np.max(np.abs(mu_o - mu_t.numpy())) < 1e-4, "parity fail"

    meta = {
        "domain": "stability", "onnx": "stability_ensemble.onnx",
        "target": {"name": "formation energy", "units": "eV/atom",
                   "note": "Experimental formation enthalpy from composition. Lower = "
                           "more favourable to form. A first-pass thermodynamic screen, "
                           "NOT energy-above-hull; predicted, not measured."},
        "feature_names": FEATURE_NAMES,
        "feat_mean": [float(v) for v in std.mean],
        "feat_std": [float(v) for v in std.std],
        "target_means": [float(sc.mean) for _, sc in ens.members],
        "target_stds": [float(sc.std) for _, sc in ens.members],
        "n_members": len(ens.members), "r2": round(float(ss), 3), "mae": round(mae, 3),
        "favourable_threshold": 0.0, "n_train": int(len(train)),
        "train_range": [float(Y.min()), float(Y.max())],
    }
    (OUT / "stability_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote stability_ensemble.onnx + stability_meta.json")


if __name__ == "__main__":
    main()
