# RESULTS — Uncertainty-Driven Materials Discovery Engine

**A controlled, retrospective benchmark of whether calibrated predictive
uncertainty improves sample efficiency in materials screening, tested for
cross-domain transfer.**

> This is a methods / benchmarking contribution. It is **not** a claim to have
> discovered a new material. Every number below is a re-discovery of already-known
> public labels that the simulation hid from itself. Results are reported exactly
> as they came out — including the parts that do not support the tidy hypothesis.

All numbers in this document are read directly from
[`results/metrics.json`](results/metrics.json) and were produced by
`python run.py --domain all --seeds 5` (config in `config/*.yaml`).

---

## 1. The question

Does calibrated predictive uncertainty make materials discovery more
sample-efficient than uncertainty-blind search — and does the same method
transfer across chemistries?

To answer it without any risk of over-claiming, we run **retrospective
rediscovery**: all "true" target values already exist in a public dataset; we
hide them, seed a model with a small random labeled set (which is forced to
contain *none* of the best materials), and let an acquisition strategy request
labels one batch at a time. We measure how quickly each strategy re-finds the
top 5% of materials.

---

## 2. Method (as built)

- **Base learner:** a heteroscedastic MLP that outputs a mean μ and a log-variance,
  trained with **Gaussian negative-log-likelihood** so every prediction carries an
  aleatoric ±σ. (Architecture inherited in spirit from MolPolySim; the molecular
  domain uses the documented Morgan-fingerprint fallback rather than a GINE graph
  network so the pipeline runs on CPU — the uncertainty/AL framework is identical.)
- **Deep ensemble (M = 5):** decomposes predictive variance into
  **aleatoric** = meanₘ σ²ₘ and **epistemic** = varₘ μₘ, with total
  σ*² = aleatoric + epistemic. The epistemic term is what a single heteroscedastic
  net (what MolPolySim shipped) cannot provide.
- **Leakage guards (asserted in code, logged every run):** the held-out test set is
  disjoint from the discovery pool; the initial seed contains none of the top-k;
  at every round the model trains only on points it has acquired. `leak_ok=True`
  on all 60 discovery runs (2 domains × 6 strategies × 5 seeds).
- **Six acquisition strategies:** `random` (floor), `greedy` (μ*),
  `ucb` (μ*+βσ*), `thompson`, `ei`, `max_var` (σ* only).
- **≥5 seeds**, mean ± 95% CI. Superconductor pool sub-sampled to 3,000 rows for
  CPU wall-clock (documented); ESOL uses all 959 pool rows.

---

## 3. Finding 1 — Discovery efficiency

Model-guided search dramatically beats random in **both** domains. Enrichment is
the base-rate enrichment factor at the final budget (precision ÷ prior hit rate).
Random sits at ≈ 0.9 rather than exactly 1.0 because the initial seed is forced to
be hit-free and is counted in the denominator.

### Domain A — Superconductors (pool 3,000; top-k = 150; 5 seeds)

| strategy | enrichment | AUDC | hits vs random | acquisitions to find 75 hits |
|---|---|---|---|---|
| random   | 0.88 ± 0.03 | 0.071 | 1.0× | never (in budget) |
| **greedy**   | **5.04 ± 0.26** | **0.550** | **5.7×** | **300** |
| ucb      | 4.96 ± 0.44 | 0.520 | 5.7× | 306 |
| ei       | 4.67 ± 0.41 | 0.495 | 5.3× | 306 |
| thompson | 4.24 ± 0.29 | 0.380 | 4.8× | 378 |
| max_var  | 2.94 ± 0.47 | 0.249 | 3.4× | 450 |

### Domain B — ESOL solubility (pool 959; top-k = 48; 5 seeds)

| strategy | enrichment | AUDC | hits vs random | acquisitions to find 24 hits |
|---|---|---|---|---|
| random   | 0.93 ± 0.22 | 0.149 | 1.0× | never (in budget) |
| **greedy**   | **2.98 ± 0.08** | 0.586 | **3.4×** | **109** |
| ucb      | 2.98 ± 0.09 | 0.595 | 3.4× | 116 |
| ei       | 2.84 ± 0.10 | 0.603 | 3.2× | 116 |
| thompson | 2.90 ± 0.04 | 0.508 | 3.3× | 147 |
| max_var  | 1.26 ± 0.33 | 0.159 | 1.5× | never (in budget) |

**Honest reading of Finding 1:**

1. **Model-guided acquisition is worth a lot.** In superconductors the best
   strategies find ~5× more top materials per measurement than random and reach
   half the top-k in ~300 measurements vs never for random within budget.
2. **Uncertainty-aware acquisition does *not* clearly beat pure exploitation.**
   `ucb`, `ei`, and `thompson` are statistically indistinguishable from `greedy`
   (overlapping 95% CIs) in both domains — on these two retrospective benchmarks
   the σ* exploration bonus buys essentially nothing over just chasing the
   predicted mean. This is a genuine, reportable negative result for the strong
   form of the hypothesis.
3. **Pure exploration is weak, as expected.** `max_var` (acquire highest σ*
   only) is the worst model strategy in both domains — clearly above random for
   superconductors (3.4×) but barely above random for ESOL (1.5×). This is the
   expected sanity contrast: exploration alone does not efficiently find top-k.

So the defensible claim is the weaker one: **calibrated models make screening far
more sample-efficient than random; the specific act of adding an uncertainty bonus
to the acquisition rule does not, here, beat greedy exploitation.**

See `results/figures/discovery_superconductor.png` and
`results/figures/discovery_molecular.png`.

---

## 4. Finding 2 — Calibration (single net vs deep ensemble)

Measured on the held-out test set over **5 seeds** (mean ± 95% CI). The `single`
baseline is a **fair standalone** heteroscedastic net trained on the full fit set
(no bootstrap — *not* an ensemble member), and **both** models are recalibrated
symmetrically on a held-out validation slice, so any gap is not an artefact of the
comparison. ECE = mean |empirical − nominal| central-interval coverage over a
19-point grid. (n_test = 3,189 superconductor / 169 ESOL.)

| domain | model | test R² | ECE (raw) | ECE (recalibrated) | Spearman(σ,|err|) |
|---|---|---|---|---|---|
| superconductor | single het. net | 0.843 ± 0.029 | 0.022 ± 0.011 | 0.013 ± 0.004 | 0.705 ± 0.015 |
| superconductor | ensemble (M=5)  | **0.859 ± 0.011** | 0.037 ± 0.013 | **0.008 ± 0.002** | **0.745 ± 0.009** |
| ESOL | single het. net | 0.754 ± 0.009 | **0.086 ± 0.034** | 0.066 ± 0.031 | −0.119 ± 0.038 |
| ESOL | ensemble (M=5)  | **0.775 ± 0.007** | 0.117 ± 0.007 | 0.071 ± 0.022 | **−0.048 ± 0.045** |

**Honest reading of Finding 2 (this result was corrected after review):**

- **The deep ensemble does *not* improve raw calibration in either domain — it is
  slightly *worse*.** A fair single heteroscedastic net is already reasonably
  calibrated (ECE 0.022 superconductor, 0.086 ESOL). Adding the epistemic variance
  term makes the ensemble mildly **over-dispersed** (0.037 and 0.117; recalibration
  factors s = 0.89 and 1.18 are both driven away from the single net's ≈1.0). This
  contradicts the naïve expectation that "ensembles are better calibrated", and the
  §7 sanity anchor "ensemble ECE ≤ single ECE" **fails in both domains** on raw σ.
  We report it as a real finding, not a bug: on top of an already-calibrated
  aleatoric estimate, the epistemic term adds variance and over-covers.
  > An earlier version of this document reported a large ESOL "ensemble calibration
  > win" (single ECE 0.27 → ensemble 0.11). An independent review found that the
  > "single net" there was a bootstrap-starved ensemble member. With a fair single
  > net the win disappears. This correction is exactly the kind of leakage-of-
  > advantage the project is meant to catch, so it is documented rather than erased.
- **Post-hoc recalibration equalises them.** A single-parameter σ-scaling fit on a
  validation slice brings both models to good calibration (superconductor ≈0.01,
  ESOL ≈0.07), with the ensemble marginally ahead after recalibration on
  superconductors (0.008 vs 0.013).
- **Where the ensemble genuinely helps:** modestly higher accuracy (R² +0.015
  superconductor, +0.021 ESOL, non-overlapping-ish CIs) and modestly better
  **error-ranking** (Spearman higher for the ensemble in both domains). And only the
  ensemble supplies the epistemic term that exploration-style acquisition needs —
  though Finding 1 showed exploration itself is weak.
- **Does σ rank error?** Strongly on superconductors (Spearman ≈ 0.7–0.75), but
  **not** on ESOL (≈ 0, even slightly negative). That the discovery loop still works
  on ESOL via the predicted mean is consistent with Finding 1: the mean, not the
  uncertainty, is doing the work here.

---

## 5. Finding 3 — Transfer across chemistries

The **same** ensemble + acquisition machinery, changed only at the featurizer
(81 inorganic composition descriptors → Morgan fingerprints of organic molecules),
produces the **same qualitative result in both domains**:

- model-guided strategies achieve large enrichment over random (≈5× superconductors,
  ≈3× ESOL);
- uncertainty-aware acquisitions track pure greedy rather than beating it;
- pure exploration (`max_var`) is the weakest model strategy;
- the deep ensemble decomposes uncertainty into aleatoric + epistemic, gives a
  small accuracy and error-ranking gain, but does not by itself improve raw
  calibration (it over-disperses until recalibrated) — the same pattern in both
  domains.

That the ranking of strategies is preserved across an inorganic-composition domain
and an organic-molecule domain is the transfer result. It is a statement about the
**method's behaviour**, not about any specific material.

---

## 6. Caveats (please read before citing any number)

- **Retrospective, not prospective.** We rediscover known labels; this bounds, but
  does not equal, real prospective discovery performance.
- **Top-k is defined within the searchable pool** (not the full dataset) so that
  the held-out calibration test set can stay disjoint from the discovery pool
  without making some hits unreachable. The superconductor pool is sub-sampled to
  3,000 rows for wall-clock; both choices are configurable.
- **Fingerprint fallback, not a graph network.** Domain B uses Morgan
  fingerprints + descriptors, not the GINE MPNN named in the design. Absolute
  molecular accuracy would likely improve with a graph model; the comparative
  conclusions about *acquisition strategies* should be robust to this.
- **ESOL, not polymer Tg.** The intended MolPolySim Tg data was unreachable in this
  environment (see `DATA_SOURCES.md`); ESOL is the documented fallback.
- **Small ESOL test set (n≈169)** means its calibration numbers have real sampling
  noise; treat them as indicative.
- **UCB β, ensemble size, epochs, and pool size are modest** for CPU budget. The
  qualitative findings were stable across 5 seeds with tight CIs, but we make no
  claim they are the asymptotic values.

---

## 7. One-paragraph summary

Across two very different chemistries, a heteroscedastic model makes retrospective
materials screening **3–5× more sample-efficient than random search**. But two
tidier sub-hypotheses are **not** supported, and we report them plainly: (1) adding
an uncertainty bonus to the acquisition rule (UCB / Thompson / EI) does **not** beat
pure greedy exploitation on either benchmark; and (2) a deep ensemble does **not**
improve raw calibration over a *fair* single heteroscedastic net — in both domains
it slightly over-disperses, and post-hoc recalibration is what delivers good
coverage. The ensemble's real, measurable benefits are a small accuracy gain, a
small error-ranking gain, and supplying the epistemic term that exploration needs
(though exploration proved weak). Predicted σ ranks error well on superconductors
but not on ESOL. The honest headline: **calibrated model *means* — not
uncertainty-guided acquisition, and not ensembling per se — are what buy the sample
efficiency here**, and this pattern transfers across both domains.
