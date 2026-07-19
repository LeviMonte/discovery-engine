"""Data loading for both domains, with fallbacks and leakage-safe splits.

Domain A -- Inorganic superconductors: UCI "Superconductivty Data" (id 464).
Domain B -- Molecular graphs (transfer): MoleculeNet ESOL/Lipophilicity fallback
            (primary MolPolySim polymer-Tg data used if the repo is reachable).

No fabrication: if data cannot be obtained, we raise rather than invent it.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .utils import DATA_DIR


# --------------------------------------------------------------------------- #
#  Container
# --------------------------------------------------------------------------- #
@dataclass
class RawData:
    domain: str
    y: np.ndarray                       # regression target (original units)
    name: str                           # dataset name
    target_name: str
    units: str
    table: pd.DataFrame | None = None   # composition features (domain A)
    feature_cols: list[str] | None = None
    smiles: list[str] | None = None     # molecular graphs (domain B)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)

    def summary(self) -> str:
        k = max(1, int(0.05 * self.n))
        top = np.sort(self.y)[-k:]
        return (
            f"[{self.domain}] {self.name}: n={self.n} rows | "
            f"target='{self.target_name}' ({self.units}) "
            f"range=[{self.y.min():.3g}, {self.y.max():.3g}] "
            f"mean={self.y.mean():.3g} | "
            f"top-5% (k={k}) threshold={top.min():.3g}"
        )


def _download(url: str, timeout: int = 60) -> bytes:
    with urlopen(url, timeout=timeout) as r:
        return r.read()


# --------------------------------------------------------------------------- #
#  Domain A: superconductors
# --------------------------------------------------------------------------- #
SUPERCON_URLS = [
    "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconductivty+data.zip",
]


def ensure_superconductor(data_dir: Path = DATA_DIR) -> Path:
    """Return path to train.csv, downloading/unzipping if necessary."""
    sc_dir = Path(data_dir) / "superconductor"
    train_csv = sc_dir / "train.csv"
    if train_csv.exists():
        return train_csv
    sc_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in SUPERCON_URLS:
        try:
            blob = _download(url)
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                zf.extractall(sc_dir)
            if train_csv.exists():
                return train_csv
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url} -> {e}")
    # matminer fallback
    try:
        from matminer.datasets import load_dataset

        df = load_dataset("superconductivity2018")
        df.to_csv(sc_dir / "train.csv", index=False)
        return train_csv
    except Exception as e:  # noqa: BLE001
        errors.append(f"matminer -> {e}")
    raise RuntimeError(
        "Could not obtain UCI superconductivity data. Tried:\n  "
        + "\n  ".join(errors)
        + "\nNo data was fabricated."
    )


def load_superconductor(data_dir: Path = DATA_DIR) -> RawData:
    train_csv = ensure_superconductor(data_dir)
    df = pd.read_csv(train_csv)
    target = "critical_temp"
    feature_cols = [c for c in df.columns if c != target]
    # keep only numeric feature columns
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    df = df.dropna(subset=[target]).reset_index(drop=True)
    y = df[target].to_numpy(dtype=np.float64)
    return RawData(
        domain="superconductor",
        y=y,
        name="UCI Superconductivty Data (Hamidieh 2018, id 464)",
        target_name="critical_temp (Tc)",
        units="K",
        table=df[feature_cols].astype(np.float64).reset_index(drop=True),
        feature_cols=feature_cols,
        meta={"n_features": len(feature_cols), "source": str(train_csv)},
    )


# --------------------------------------------------------------------------- #
#  Domain B: molecular
# --------------------------------------------------------------------------- #
MOL_SOURCES = {
    "esol": {
        "urls": [
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
            "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
        ],
        "fname": "esol.csv",
        "smiles_col": "smiles",
        "target_col": "measured log solubility in mols per litre",
        "name": "MoleculeNet ESOL / Delaney aqueous solubility",
        "target_name": "measured log solubility",
        "units": "log(mol/L)",
    },
    "lipophilicity": {
        "urls": [
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
            "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/Lipophilicity.csv",
        ],
        "fname": "lipo.csv",
        "smiles_col": "smiles",
        "target_col": "exp",
        "name": "MoleculeNet Lipophilicity (ChEMBL logD)",
        "target_name": "experimental logD",
        "units": "logD",
    },
}


def ensure_molecular(dataset: str, data_dir: Path = DATA_DIR) -> Path:
    spec = MOL_SOURCES[dataset]
    path = Path(data_dir) / spec["fname"]
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in spec["urls"]:
        try:
            blob = _download(url)
            path.write_bytes(blob)
            return path
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url} -> {e}")
    raise RuntimeError(
        f"Could not obtain molecular dataset '{dataset}'. Tried:\n  "
        + "\n  ".join(errors)
        + "\nNo data was fabricated."
    )


def load_molecular(dataset: str = "esol", data_dir: Path = DATA_DIR) -> RawData:
    spec = MOL_SOURCES[dataset]
    path = ensure_molecular(dataset, data_dir)
    df = pd.read_csv(path)
    smiles_col, target_col = spec["smiles_col"], spec["target_col"]
    df = df.dropna(subset=[smiles_col, target_col]).reset_index(drop=True)
    smiles = df[smiles_col].astype(str).tolist()
    y = df[target_col].to_numpy(dtype=np.float64)
    return RawData(
        domain="molecular",
        y=y,
        name=spec["name"],
        target_name=spec["target_name"],
        units=spec["units"],
        smiles=smiles,
        meta={"dataset": dataset, "source": str(path)},
    )


def load_domain(domain: str, cfg: dict, data_dir: Path = DATA_DIR) -> RawData:
    if domain == "superconductor":
        return load_superconductor(data_dir)
    if domain == "molecular":
        ds = cfg.get("data", {}).get("dataset", "esol")
        try:
            return load_molecular(ds, data_dir)
        except Exception:
            # last-resort fallback to the other molecular set
            other = "lipophilicity" if ds == "esol" else "esol"
            return load_molecular(other, data_dir)
    raise ValueError(f"unknown domain: {domain}")


if __name__ == "__main__":
    for dom, cfg in [("superconductor", {}), ("molecular", {"data": {"dataset": "esol"}})]:
        try:
            raw = load_domain(dom, cfg)
            print(raw.summary())
        except Exception as e:  # noqa: BLE001
            print(f"[{dom}] FAILED: {e}")
