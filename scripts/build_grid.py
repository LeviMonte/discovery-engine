#!/usr/bin/env python3
"""Phase 8.0 fallback -- precomputed prediction library for the MOLECULAR domain.

The molecular model consumes Morgan-fingerprint + descriptor features. Reproducing
that exact featurization in-browser (RDKit-WASM parity with the Python pipeline) is
fragile, so -- per the build spec's explicit per-domain fallback -- the molecular
panel ships a *precomputed curated library* instead of live ONNX: we train the
ensemble here, predict on a representative set of real ESOL molecules (spanning the
train + test distribution), and write those predictions as a static JSON lookup.

Every entry carries the true measured value, the train/test membership, and a
novel-scaffold (out-of-distribution) flag, so the UI stays honest.
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
from discovery.models import DeepEnsemble        # noqa: E402
from discovery.utils import load_config, set_all_seeds  # noqa: E402

OUT = ROOT / "web" / "predict" / "models"


def main(n_library=400):
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config" / "molecular.yaml")
    set_all_seeds(0)
    raw = load_domain("molecular", cfg)
    feat = featurize(raw, cfg)           # order preserved for valid SMILES
    # names/true values from the source csv, aligned to the valid rows
    df = pd.read_csv(ROOT / "data" / "esol.csv")
    df = df.dropna(subset=["smiles", "measured log solubility in mols per litre"])
    names = df["Compound ID"].astype(str).tolist()
    # feat dropped invalid SMILES; rebuild the valid mask the same way
    from rdkit import Chem
    valid = [Chem.MolFromSmiles(s) is not None for s in raw.smiles]
    names = [n for n, v in zip(names, valid) if v]
    smiles = [s for s, v in zip(raw.smiles, valid) if v]
    assert len(names) == len(feat.y), (len(names), len(feat.y))

    train_idx, test_idx = make_split(feat, cfg, seed=0)
    Xz, _ = feat.standardized(train_idx)

    print("training molecular ensemble for library predictions ...", flush=True)
    ens = DeepEnsemble(cfg.get("model", {}), base_seed=0)
    ens.fit(Xz[train_idx], feat.y[train_idx])

    split = np.array(["train"] * len(feat.y), dtype=object)
    split[test_idx] = "test"
    train_scaffolds = set(feat.scaffolds[train_idx])

    # ---- choose a representative curated library ----
    # all test molecules (novel scaffolds -> good OOD demos) + a stratified
    # sample of train molecules spanning the solubility range.
    order = np.argsort(feat.y)
    lib = list(test_idx)
    remaining = n_library - len(lib)
    if remaining > 0:
        train_sorted = [i for i in order if split[i] == "train"]
        step = max(1, len(train_sorted) // remaining)
        lib += train_sorted[::step][:remaining]
    lib = sorted(set(int(i) for i in lib))

    pred = ens.predict(Xz[lib])
    tmean = raw.name  # unused

    entries = []
    for k, i in enumerate(lib):
        novel = feat.scaffolds[i] not in train_scaffolds
        entries.append({
            "smiles": smiles[i],
            "name": names[i][:60],
            "split": split[i],
            "true": round(float(feat.y[i]), 3),
            "mu": round(float(pred.mu[k]), 3),
            "aleatoric": round(float(pred.aleatoric[k]), 4),
            "epistemic": round(float(pred.epistemic[k]), 4),
            "sigma": round(float(pred.sigma[k]), 3),
            "novel_scaffold": bool(novel),
        })
    entries.sort(key=lambda e: e["mu"], reverse=True)

    grid = {
        "domain": "molecular",
        "inference": "precomputed-library",
        "dataset": raw.name,
        "target": {
            "name": "aqueous solubility (log S)", "units": "log(mol/L)",
            "note": "This is the ESOL solubility model actually trained in Phases "
                    "0-7 (the intended polymer-Tg data was unreachable). Predictions "
                    "here are precomputed, not live in-browser inference.",
        },
        "n_members": cfg.get("model", {}).get("ensemble_size", 5),
        "n_train": int((split == "train").sum()),
        "n_test": int((split == "test").sum()),
        "true_range": [round(float(feat.y.min()), 2), round(float(feat.y.max()), 2)],
        "library": entries,
    }
    (OUT / "grid_molecular.json").write_text(json.dumps(grid))
    n_novel = sum(e["novel_scaffold"] for e in entries)
    print(f"wrote grid_molecular.json | {len(entries)} molecules "
          f"({n_novel} novel-scaffold / OOD)")
    # sanity: model tracks truth
    mu = np.array([e["mu"] for e in entries]); tr = np.array([e["true"] for e in entries])
    print(f"library corr(pred,true)={np.corrcoef(mu, tr)[0,1]:.3f}")


if __name__ == "__main__":
    main()
