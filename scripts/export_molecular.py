#!/usr/bin/env python3
"""Phase 9 -- export the MOLECULAR ensemble to ONNX for LIVE in-browser inference.

The Phase-8 molecular panel used a precomputed library because reproducing the
Python featurization in-browser was deemed fragile. Phase 9 removes that
limitation: RDKit.js reproduces our Morgan fingerprints bit-for-bit and 9 of the
10 descriptors exactly (only NumHAcceptors differs by definition), so we train the
browser model on those 9 JS-reproducible descriptors and export it. The sketcher
can then featurize an arbitrary drawn molecule in the browser and run the real
ensemble live.

This does NOT touch the Phase 0-7 model or results/metrics.json (those keep the
original 10-descriptor featurization). It writes:
  web/predict/models/molecular_ensemble.onnx
  web/predict/models/molecular_meta.json   (descriptor stats, target scalers,
                                             epistemic OOD threshold, examples)
  web/predict/models/grid_molecular.json    (curated library, regenerated from
                                             THIS model so live + library agree)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discovery.data import load_domain              # noqa: E402
from discovery.featurize import featurize, make_split, Standardizer  # noqa: E402
from discovery.models import DeepEnsemble            # noqa: E402
from discovery.utils import load_config, set_all_seeds  # noqa: E402
from export_onnx import EnsembleExport               # noqa: E402

OUT = ROOT / "web" / "predict" / "models"

# 9 descriptors RDKit.js reproduces exactly (NumHAcceptors dropped -- see docstring)
WEB_DESCRIPTORS = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumRotatableBonds",
                   "RingCount", "FractionCSP3", "NumAromaticRings", "HeavyAtomCount"]
N_BITS, RADIUS = 1024, 2


def web_featurize(smiles_list):
    """Morgan(1024,r2) bits + 9 JS-reproducible descriptors (unstandardized)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.DataStructs import ConvertToNumpyArray

    fns = {n: getattr(Descriptors, n) for n in WEB_DESCRIPTORS}
    rows, valid = [], []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            rows.append(np.zeros(N_BITS + len(fns))); valid.append(False); continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=N_BITS)
        arr = np.zeros(N_BITS); ConvertToNumpyArray(fp, arr)
        d = np.array([fns[n](m) for n in WEB_DESCRIPTORS], dtype=float)
        rows.append(np.concatenate([arr, np.nan_to_num(d)])); valid.append(True)
    return np.vstack(rows).astype(np.float32), np.array(valid)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config" / "molecular.yaml")
    set_all_seeds(0)
    raw = load_domain("molecular", cfg)
    feat = featurize(raw, cfg)                 # for scaffolds + identical split
    train_idx, test_idx = make_split(feat, cfg, seed=0)

    # valid SMILES in the same order featurize() kept
    from rdkit import Chem
    valid_mask = [Chem.MolFromSmiles(s) is not None for s in raw.smiles]
    smiles = [s for s, v in zip(raw.smiles, valid_mask) if v]
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "esol.csv").dropna(
        subset=["smiles", "measured log solubility in mols per litre"])
    names = [n for n, v in zip(df["Compound ID"].astype(str).tolist(), valid_mask) if v]
    assert len(smiles) == len(feat.y)

    Xw, _ = web_featurize(smiles)
    # standardize only the 9 descriptor columns using train stats
    std_cols = np.zeros(Xw.shape[1], dtype=bool); std_cols[N_BITS:] = True
    std = Standardizer.fit(Xw, train_idx, cols=std_cols)
    Xz = std.transform(Xw)

    print("training molecular ensemble (9-descriptor, JS-reproducible) ...", flush=True)
    ens = DeepEnsemble(cfg.get("model", {}), base_seed=0)
    ens.fit(Xz[train_idx], feat.y[train_idx])

    # ---- export ONNX ----
    wrapper = EnsembleExport(ens.members).eval()
    dummy = torch.tensor(Xz[test_idx][:1], dtype=torch.float32)
    onnx_path = OUT / "molecular_ensemble.onnx"
    torch.onnx.export(wrapper, dummy, str(onnx_path),
        input_names=["x_std"], output_names=["mu_std", "logvar_std"],
        dynamic_axes={"x_std": {0: "batch"}, "mu_std": {0: "batch"},
                      "logvar_std": {0: "batch"}}, opset_version=17, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    xp = Xz[test_idx][:10].astype(np.float32)
    with torch.no_grad():
        mu_t, lv_t = wrapper(torch.tensor(xp))
    mu_o, lv_o = sess.run(None, {"x_std": xp})
    rel = lambda a, b: float(np.max(np.abs(a - b) / (np.abs(b) + 1e-6)))
    assert rel(mu_o, mu_t.numpy()) < 1e-4 and rel(lv_o, lv_t.numpy()) < 1e-4
    print(f"parity ok: mu={rel(mu_o, mu_t.numpy()):.2e} logvar={rel(lv_o, lv_t.numpy()):.2e}")

    tmeans = [float(sc.mean) for _, sc in ens.members]
    tstds = [float(sc.std) for _, sc in ens.members]

    # ---- epistemic OOD threshold from TRAIN predictions ----
    ptr = ens.predict(Xz[train_idx])
    epi_thresh = float(np.percentile(ptr.epistemic, 90))

    # ---- regenerate curated grid from THIS model (live + library agree) ----
    split = np.array(["train"] * len(feat.y), dtype=object); split[test_idx] = "test"
    train_scaf = set(feat.scaffolds[train_idx])
    order = np.argsort(feat.y)
    lib = list(test_idx)
    ts = [i for i in order if split[i] == "train"]
    step = max(1, len(ts) // max(1, 400 - len(lib)))
    lib += ts[::step][:400 - len(lib)]
    lib = sorted(set(int(i) for i in lib))
    pred = ens.predict(Xz[lib])
    entries = []
    for k, i in enumerate(lib):
        entries.append({
            "smiles": smiles[i], "name": names[i][:60], "split": split[i],
            "true": round(float(feat.y[i]), 3), "mu": round(float(pred.mu[k]), 3),
            "aleatoric": round(float(pred.aleatoric[k]), 4),
            "epistemic": round(float(pred.epistemic[k]), 4),
            "sigma": round(float(pred.sigma[k]), 3),
            "novel_scaffold": bool(feat.scaffolds[i] not in train_scaf),
        })
    entries.sort(key=lambda e: e["mu"], reverse=True)

    target = {"name": "aqueous solubility (log S)", "units": "log(mol/L)",
              "note": "ESOL solubility model (the intended polymer-Tg data was "
                      "unreachable). Live in-browser inference via RDKit.js + ONNX."}
    meta = {
        "domain": "molecular", "inference": "live-onnx-rdkitjs",
        "onnx": "molecular_ensemble.onnx", "target": target,
        "n_bits": N_BITS, "radius": RADIUS,
        "descriptors": WEB_DESCRIPTORS,
        "desc_mean": [float(std.mean[N_BITS + j]) for j in range(len(WEB_DESCRIPTORS))],
        "desc_std": [float(std.std[N_BITS + j]) for j in range(len(WEB_DESCRIPTORS))],
        "target_means": tmeans, "target_stds": tstds, "n_members": len(ens.members),
        "epistemic_ood_threshold": epi_thresh,
        "true_range": [round(float(feat.y.min()), 2), round(float(feat.y.max()), 2)],
        "allowed_elements": ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"],
    }
    (OUT / "molecular_meta.json").write_text(json.dumps(meta, indent=2))
    grid = {"domain": "molecular", "inference": "live-onnx-rdkitjs",
            "dataset": raw.name, "target": target, "n_members": len(ens.members),
            "n_train": int((split == "train").sum()), "n_test": int((split == "test").sum()),
            "true_range": meta["true_range"], "library": entries}
    (OUT / "grid_molecular.json").write_text(json.dumps(grid))
    corr = np.corrcoef([e["mu"] for e in entries], [e["true"] for e in entries])[0, 1]
    print(f"wrote molecular_ensemble.onnx, molecular_meta.json, grid_molecular.json")
    print(f"  library {len(entries)} mols | corr(pred,true)={corr:.3f} | "
          f"epi OOD thresh={epi_thresh:.4f}")


if __name__ == "__main__":
    main()
