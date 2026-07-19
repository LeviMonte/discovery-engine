"""Acquisition functions for pool-based active discovery (maximize target).

Each function scores the *unlabeled candidate pool* and returns the indices
(into that candidate array) of the next batch to reveal.

Strategies:
  random    -- baseline floor (uncertainty-blind, no model use)
  greedy    -- highest predicted mean mu*        (pure exploitation)
  ucb       -- mu* + beta * sigma*               (uncertainty-aware hypothesis)
  thompson  -- sample ~ N(mu*, sigma*^2), take top (uncertainty-aware)
  ei        -- expected improvement over best labeled-so-far
  max_var   -- highest sigma*                     (pure exploration; sanity)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _topk(scores, batch):
    batch = min(batch, len(scores))
    # argpartition then sort for determinism on ties (stable by index)
    idx = np.argsort(-scores, kind="stable")[:batch]
    return idx


def acquire_random(mu, sigma, best_y, batch, rng):
    n = len(mu)
    return rng.permutation(n)[: min(batch, n)]


def acquire_greedy(mu, sigma, best_y, batch, rng):
    return _topk(mu, batch)


def acquire_ucb(mu, sigma, best_y, batch, rng, beta=1.5):
    return _topk(mu + beta * sigma, batch)


def acquire_thompson(mu, sigma, best_y, batch, rng):
    samples = rng.normal(mu, np.maximum(sigma, 1e-9))
    return _topk(samples, batch)


def acquire_ei(mu, sigma, best_y, batch, rng):
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - best_y) / sigma
    ei = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
    return _topk(ei, batch)


def acquire_max_var(mu, sigma, best_y, batch, rng):
    return _topk(sigma, batch)


REGISTRY = {
    "random": acquire_random,
    "greedy": acquire_greedy,
    "ucb": acquire_ucb,
    "thompson": acquire_thompson,
    "ei": acquire_ei,
    "max_var": acquire_max_var,
}

STRATEGIES = list(REGISTRY.keys())


def acquire(strategy, mu, sigma, best_y, batch, rng, beta=1.5):
    fn = REGISTRY[strategy]
    if strategy == "ucb":
        return fn(mu, sigma, best_y, batch, rng, beta=beta)
    return fn(mu, sigma, best_y, batch, rng)
