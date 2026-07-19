"""Shared utilities: seeding, config loading, device."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml


def set_all_seeds(seed: int) -> None:
    """Seed python, numpy and torch (incl. cuda) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinism on CPU is the default; keep matmul deterministic.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_device():
    try:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return "cpu"


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_smoke(cfg: dict) -> dict:
    """Overlay the `smoke` block onto the relevant sub-configs for fast runs."""
    smoke = cfg.get("smoke", {})
    if not smoke:
        return cfg
    if "pool_subsample" in smoke:
        cfg.setdefault("data", {})["pool_subsample"] = smoke["pool_subsample"]
    for k in ("ensemble_size", "epochs", "patience"):
        if k in smoke:
            cfg.setdefault("model", {})[k] = smoke[k]
    if "n_rounds" in smoke:
        cfg.setdefault("active_learning", {})["n_rounds"] = smoke["n_rounds"]
    cfg["_smoke_seeds"] = smoke.get("seeds", 2)
    return cfg


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
