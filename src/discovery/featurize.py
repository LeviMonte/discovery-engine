"""Featurization + leakage-safe splits.

- Superconductors: precomputed 81 composition features, z-scored on TRAIN stats.
- Molecules: RDKit Morgan fingerprints (+ a few continuous descriptors),
  with a Bemis-Murcko scaffold split so no scaffold is shared train/test.

Standardization statistics are always computed on the training indices only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import RawData


# --------------------------------------------------------------------------- #
#  Standardizer (train-stats only)
# --------------------------------------------------------------------------- #
@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    cols: np.ndarray  # boolean mask of columns to standardize

    @classmethod
    def fit(cls, X: np.ndarray, train_idx: np.ndarray, cols: np.ndarray | None = None):
        Xt = X[train_idx]
        if cols is None:
            cols = np.ones(X.shape[1], dtype=bool)
        mean = np.zeros(X.shape[1])
        std = np.ones(X.shape[1])
        mean[cols] = Xt[:, cols].mean(axis=0)
        s = Xt[:, cols].std(axis=0)
        s[s < 1e-8] = 1.0
        std[cols] = s
        return cls(mean=mean, std=std, cols=cols)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Molecular featurization
# --------------------------------------------------------------------------- #
_DESCRIPTORS = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "FractionCSP3", "NumAromaticRings",
    "HeavyAtomCount",
]


def _mol_from_smiles(smiles: str):
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles)


def morgan_and_descriptors(smiles_list, n_bits=1024, radius=2, use_descriptors=True):
    """Return (X, valid_mask). Invalid SMILES yield a zero row + mask=False."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    desc_fns = {n: getattr(Descriptors, n) for n in _DESCRIPTORS} if use_descriptors else {}
    n_desc = len(desc_fns)
    rows = []
    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(n_bits + n_desc, dtype=np.float64))
            valid.append(False)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float64)
        from rdkit.DataStructs import ConvertToNumpyArray

        ConvertToNumpyArray(fp, arr)
        if n_desc:
            d = np.array([fn(mol) for fn in desc_fns.values()], dtype=np.float64)
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            arr = np.concatenate([arr, d])
        rows.append(arr)
        valid.append(True)
    X = np.vstack(rows)
    return X, np.array(valid, dtype=bool), n_bits, n_desc


def bemis_murcko_scaffold(smiles: str) -> str:
    """Canonical Bemis-Murcko scaffold SMILES ('' if it cannot be computed)."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf)
    except Exception:  # noqa: BLE001
        return ""


def scaffold_split(smiles_list, test_frac=0.15, seed=0):
    """Split indices so that no Bemis-Murcko scaffold spans train and test.

    Scaffolds are assigned greedily (largest groups first) to the train side
    until it is full, guaranteeing disjoint scaffold sets.
    """
    scaffolds: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        scaffolds.setdefault(bemis_murcko_scaffold(smi), []).append(i)
    # deterministic ordering: by group size desc, then scaffold string
    groups = sorted(scaffolds.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    n = len(smiles_list)
    n_test_target = int(round(test_frac * n))
    train_idx: list[int] = []
    test_idx: list[int] = []
    # Fill the test set from the *smaller* scaffold groups first so large
    # common scaffolds stay in train (standard MoleculeNet behaviour).
    rng = np.random.RandomState(seed)
    small_first = sorted(scaffolds.items(), key=lambda kv: (len(kv[1]), kv[0]))
    for _, idxs in small_first:
        if len(test_idx) + len(idxs) <= n_test_target:
            test_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    # anything unassigned -> train (safety)
    assigned = set(train_idx) | set(test_idx)
    for i in range(n):
        if i not in assigned:
            train_idx.append(i)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def random_split(n, test_frac=0.15, seed=0):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = int(round(test_frac * n))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx


# --------------------------------------------------------------------------- #
#  Unified entry
# --------------------------------------------------------------------------- #
@dataclass
class Featurized:
    X: np.ndarray            # raw (unstandardized) feature matrix, float32
    y: np.ndarray
    domain: str
    std_cols: np.ndarray     # which columns to z-score
    scaffolds: np.ndarray | None = None  # per-row scaffold id (molecular)
    feature_names: list | None = None

    def standardized(self, train_idx: np.ndarray):
        std = Standardizer.fit(self.X, train_idx, cols=self.std_cols)
        return std.transform(self.X), std


def featurize(raw: RawData, cfg: dict) -> Featurized:
    if raw.domain == "superconductor":
        X = raw.table.to_numpy(dtype=np.float32)
        std_cols = np.ones(X.shape[1], dtype=bool)  # all continuous
        return Featurized(
            X=X, y=raw.y.astype(np.float64), domain=raw.domain,
            std_cols=std_cols, feature_names=list(raw.feature_cols),
        )
    if raw.domain == "molecular":
        fc = cfg.get("featurize", {})
        X, valid, n_bits, n_desc = morgan_and_descriptors(
            raw.smiles,
            n_bits=fc.get("morgan_bits", 1024),
            radius=fc.get("morgan_radius", 2),
            use_descriptors=fc.get("use_descriptors", True),
        )
        # standardize only the continuous descriptor tail, leave bits as 0/1
        std_cols = np.zeros(X.shape[1], dtype=bool)
        if n_desc:
            std_cols[n_bits:] = True
        scaffolds = np.array([bemis_murcko_scaffold(s) for s in raw.smiles])
        # drop invalid molecules entirely (no fabricated features)
        X = X[valid]
        y = raw.y[valid].astype(np.float64)
        scaffolds = scaffolds[valid]
        return Featurized(
            X=X.astype(np.float32), y=y, domain=raw.domain,
            std_cols=std_cols, scaffolds=scaffolds,
        )
    raise ValueError(raw.domain)


def make_split(feat: Featurized, cfg: dict, seed: int):
    """Held-out test split (calibration only). Scaffold for molecules."""
    split = cfg.get("data", {}).get("split", "random")
    if split == "scaffold" and feat.scaffolds is not None:
        # rebuild scaffold split on the (already-valid) rows
        train_idx, test_idx = _scaffold_split_from_ids(
            feat.scaffolds, cfg.get("data", {}).get("test_frac", 0.15), seed
        )
    else:
        train_idx, test_idx = random_split(
            len(feat.y), cfg.get("data", {}).get("test_frac", 0.15), seed
        )
    return train_idx, test_idx


def _scaffold_split_from_ids(scaffolds: np.ndarray, test_frac: float, seed: int):
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(scaffolds):
        groups.setdefault(s, []).append(i)
    n = len(scaffolds)
    n_test_target = int(round(test_frac * n))
    test_idx: list[int] = []
    train_idx: list[int] = []
    small_first = sorted(groups.items(), key=lambda kv: (len(kv[1]), kv[0]))
    for _, idxs in small_first:
        if len(test_idx) + len(idxs) <= n_test_target:
            test_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    assigned = set(train_idx) | set(test_idx)
    for i in range(n):
        if i not in assigned:
            train_idx.append(i)
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))
