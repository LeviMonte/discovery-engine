#!/usr/bin/env python3
"""Phase 10 -- synthesizability (SAscore) + precomputed 3D conformers.

- Real SAscore (Ertl & Schuffenhauer 2009, RDKit Contrib sascorer) for every
  library molecule -> added to grid_molecular.json.
- Precomputed 3D conformers (ETKDG + MMFF) as molblocks -> mol3d.json, rendered by
  3Dmol.js. RDKit.js has no 3D embedding, so novel/sketched molecules fall back to
  a 2D depiction (RDKit.js SVG) in the UI.
- A calibrated IN-BROWSER SAscore estimator for novel/generated molecules: a folded
  fragment-score table + a small linear model over browser-computable features,
  fit to reproduce reference SAscore. We ship it only with its validation R^2 so
  the UI can be honest about it.
"""
from __future__ import annotations

import gzip
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "predict" / "models"
from rdkit import Chem                                    # noqa: E402
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, RDConfig  # noqa: E402
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer                                           # noqa: E402

NBITS = 32768


def load_fpscores():
    p = os.path.join(RDConfig.RDContribDir, "SA_Score", "fpscores.pkl.gz")
    data = pickle.load(gzip.open(p))
    d = {}
    for e in data:
        for k in e[1:]:
            d[k] = float(e[0])
    return d


def folded_table(fpscores):
    """Average reference score of fragments folding to each of NBITS buckets."""
    acc = np.zeros(NBITS); cnt = np.zeros(NBITS)
    for h, s in fpscores.items():
        b = h % NBITS
        acc[b] += s; cnt[b] += 1
    default = float(np.mean(list(fpscores.values())))
    T = np.where(cnt > 0, acc / np.maximum(cnt, 1), default)
    return T, default


FEAT_ORDER = ["folded_frag", "heavy_atoms", "num_rings", "fraction_csp3", "molwt",
              "num_arom_rings", "num_rot", "num_hbd", "tpsa", "clogp"]


def browser_feats(mol, T, default):
    """Features the browser (RDKit.js) can compute: folded Morgan presence + descriptors."""
    bits = list(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=NBITS).GetOnBits())
    frag = float(np.mean([T[b] for b in bits])) if bits else default
    from rdkit.Chem import Crippen
    return [frag,
            float(Descriptors.HeavyAtomCount(mol)),
            float(Descriptors.RingCount(mol)),
            float(Descriptors.FractionCSP3(mol)),
            float(Descriptors.MolWt(mol)),
            float(Descriptors.NumAromaticRings(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            float(Descriptors.NumHDonors(mol)),
            float(Descriptors.TPSA(mol)),
            float(Crippen.MolLogP(mol))]


def main():
    grid_path = OUT / "grid_molecular.json"
    grid = json.loads(grid_path.read_text())
    fpscores = load_fpscores()
    T, default = folded_table(fpscores)

    # real SAscore for the library entries
    mols = []
    for e in grid["library"]:
        m = Chem.MolFromSmiles(e["smiles"])
        if m is None:
            continue
        e["sascore"] = round(float(sascorer.calculateScore(m)), 2)
        mols.append((e["smiles"], m))

    # fit the in-browser estimator on ALL ESOL molecules (more data), held-out R^2
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "esol.csv").dropna(subset=["smiles"])
    X, y = [], []
    for smi in df["smiles"].astype(str):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        X.append(browser_feats(m, T, default)); y.append(float(sascorer.calculateScore(m)))
    X = np.array(X); y = np.array(y)
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_predict
    reg = LinearRegression().fit(X, y)
    cvp = cross_val_predict(LinearRegression(), X, y, cv=5)
    r2_cv = float(1 - np.sum((y - cvp) ** 2) / np.sum((y - y.mean()) ** 2))
    mae_cv = float(np.mean(np.abs(cvp - y)))
    print(f"in-browser SAscore estimator: 5-fold R^2={r2_cv:.3f} MAE={mae_cv:.2f} over {len(y)} mols")

    (OUT / "sascore_model.json").write_text(json.dumps({
        "nbits": NBITS, "table": [round(float(t), 4) for t in T],
        "default": round(default, 4),
        "coef": [float(c) for c in reg.coef_], "intercept": float(reg.intercept_),
        "feat_order": FEAT_ORDER,
        "r2": round(r2_cv, 3), "mae": round(mae_cv, 2),
        "note": "Calibrated in-browser estimate of Ertl-Schuffenhauer SAscore (1=easy, "
                "10=hard). Reference SAscore is precomputed exactly for the library; "
                "for novel molecules this estimator reproduces it with the stated "
                "5-fold R^2 — treat it as a rough guide, not the exact score.",
    }))

    # 3D conformers for the library
    mol3d = {}
    for smi, m in mols:
        try:
            mh = Chem.AddHs(m)
            if AllChem.EmbedMolecule(mh, randomSeed=0xf00d) != 0:
                continue
            try:
                AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
            except Exception:
                pass
            mol3d[smi] = Chem.MolToMolBlock(mh)
        except Exception:
            continue
    (OUT / "mol3d.json").write_text(json.dumps(mol3d))
    grid_path.write_text(json.dumps(grid))
    print(f"wrote sascore into grid ({len(y)} mols), sascore_model.json, "
          f"mol3d.json ({len(mol3d)} conformers)")
    print(f"  SAscore range in library: {min(e.get('sascore',9) for e in grid['library'])}"
          f"–{max(e.get('sascore',0) for e in grid['library'])}")


if __name__ == "__main__":
    main()
