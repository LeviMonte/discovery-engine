"""Heteroscedastic networks + deep ensemble.

Base learner: an MLP outputting (mu, log_var), trained with Gaussian NLL so each
model reports an aleatoric uncertainty. A deep ensemble of M such learners lets
us separate:
    predictive mean  mu*      = mean_m mu_m
    aleatoric        = mean_m sigma_m^2
    epistemic        = var_m  mu_m
    total variance   sigma*^2 = aleatoric + epistemic

Active learning needs the epistemic term, so the ensemble is essential -- a
single heteroscedastic net (what MolPolySim shipped) cannot supply it. That
contrast is one of the reported results.

NOTE on architecture: the spec's molecular model is a GINE message-passing
network. torch_geometric is an optional/fragile dependency, so this reference
build uses the documented fallback -- Morgan-fingerprint features into the same
heteroscedastic MLP used for the superconductor domain. The uncertainty +
active-learning framework is identical either way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Target scaler (standardize for training, invert for reporting)
# --------------------------------------------------------------------------- #
@dataclass
class TargetScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, y: np.ndarray):
        std = float(y.std())
        return cls(mean=float(y.mean()), std=std if std > 1e-8 else 1.0)

    def transform(self, y):
        return (y - self.mean) / self.std

    def inv_mean(self, m):
        return m * self.std + self.mean

    def inv_var(self, v):
        return v * (self.std ** 2)


# --------------------------------------------------------------------------- #
#  Base network
# --------------------------------------------------------------------------- #
class HeteroscedasticMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 256, 128), logvar_min=-8.0, logvar_max=6.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(d, 1)
        self.logvar_head = nn.Linear(d, 1)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward(self, x):
        z = self.body(x)
        mu = self.mu_head(z).squeeze(-1)
        logvar = self.logvar_head(z).squeeze(-1)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        return mu, logvar


def gaussian_nll(mu, logvar, y):
    """Mean Gaussian negative log-likelihood (up to a constant)."""
    return (0.5 * (logvar + (y - mu) ** 2 / torch.exp(logvar))).mean()


# --------------------------------------------------------------------------- #
#  Single-model training (early stopping on a val split)
# --------------------------------------------------------------------------- #
def train_one(
    X, y, cfg, seed, val_frac=0.15, verbose=False
):
    """Train one heteroscedastic MLP. Returns (model, target_scaler)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.RandomState(seed)

    n = len(y)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    if len(tr_idx) == 0:  # tiny-data guard
        tr_idx = perm
        val_idx = perm

    scaler = TargetScaler.fit(y[tr_idx])
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(scaler.transform(y), dtype=torch.float32)

    model = HeteroscedasticMLP(
        X.shape[1], tuple(cfg.get("hidden", [256, 256, 128])),
        cfg.get("logvar_min", -8.0), cfg.get("logvar_max", 6.0),
    )
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    bs = cfg.get("batch_size", 256)
    epochs = cfg.get("epochs", 120)
    patience = cfg.get("patience", 12)

    tr = torch.tensor(tr_idx, dtype=torch.long)
    va = torch.tensor(val_idx, dtype=torch.long)
    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        order = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(order), bs):
            b = order[i : i + bs]
            opt.zero_grad()
            mu, lv = model(Xt[b])
            loss = gaussian_nll(mu, lv, yt[b])
            if not torch.isfinite(loss):
                # instability guard: skip bad batch
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            mu, lv = model(Xt[va])
            vloss = gaussian_nll(mu, lv, yt[va]).item()
        if vloss < best_val - 1e-4:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, scaler


# --------------------------------------------------------------------------- #
#  Deep ensemble
# --------------------------------------------------------------------------- #
@dataclass
class EnsemblePrediction:
    mu: np.ndarray          # predictive mean (original units)
    aleatoric: np.ndarray   # mean_m sigma_m^2
    epistemic: np.ndarray   # var_m mu_m
    total_var: np.ndarray   # aleatoric + epistemic

    @property
    def sigma(self):
        return np.sqrt(self.total_var)


class DeepEnsemble:
    def __init__(self, cfg, base_seed=0):
        self.cfg = cfg
        self.base_seed = base_seed
        self.M = cfg.get("ensemble_size", 5)
        self.members = []       # list of (model, scaler)

    def fit(self, X, y):
        self.members = []
        rng = np.random.RandomState(self.base_seed)
        for m in range(self.M):
            seed = self.base_seed * 1000 + m
            # bootstrap resample for diversity (different data shuffles)
            idx = rng.randint(0, len(y), size=len(y))
            model, scaler = train_one(X[idx], y[idx], self.cfg, seed=seed)
            self.members.append((model, scaler))
        return self

    def _member_predict(self, X):
        Xt = torch.tensor(X, dtype=torch.float32)
        mus, var_al = [], []
        with torch.no_grad():
            for model, scaler in self.members:
                mu, lv = model(Xt)
                mu = mu.numpy()
                var = np.exp(lv.numpy())
                mus.append(scaler.inv_mean(mu))
                var_al.append(scaler.inv_var(var))
        return np.stack(mus), np.stack(var_al)  # (M, N) each

    def predict(self, X) -> EnsemblePrediction:
        mus, var_al = self._member_predict(X)
        mu_star = mus.mean(axis=0)
        aleatoric = var_al.mean(axis=0)
        epistemic = mus.var(axis=0)
        return EnsemblePrediction(
            mu=mu_star, aleatoric=aleatoric, epistemic=epistemic,
            total_var=aleatoric + epistemic,
        )

    def predict_single(self, X) -> EnsemblePrediction:
        """First member only -- the 'single heteroscedastic net' baseline.

        Epistemic term is unavailable, so total_var == aleatoric.
        """
        Xt = torch.tensor(X, dtype=torch.float32)
        model, scaler = self.members[0]
        with torch.no_grad():
            mu, lv = model(Xt)
            mu = scaler.inv_mean(mu.numpy())
            var = scaler.inv_var(np.exp(lv.numpy()))
        return EnsemblePrediction(
            mu=mu, aleatoric=var, epistemic=np.zeros_like(var), total_var=var,
        )
