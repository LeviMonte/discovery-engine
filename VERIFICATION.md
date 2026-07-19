# VERIFICATION

A running log of what was checked, what failed, and how it was fixed — per the
build spec's error-reevaluation and self-correction protocol (§7).

## 1. Unit / integration tests (`pytest`)
`19 passed`. Coverage:
- featurizer shapes; invalid-SMILES flagged, not fabricated.
- **scaffold split**: train/test share **zero** Bemis–Murcko scaffolds (asserted).
- **standardization uses train statistics only** (asserted the fitted mean equals
  the train slice mean and *not* the full-array mean).
- acquisition functions: correct top-k selection, no duplicates, `greedy`/`max_var`/
  `ucb` pick the expected items.
- ensemble: positive held-out R², exact `total = aleatoric + epistemic`,
  `epistemic ≥ 0`; single-net prediction has zero epistemic.
- ECE/coverage: well-specified Gaussian gives low ECE; recalibration reduces ECE and
  inflates σ when the input is overconfident.
- **leakage guards** (synthetic data): seed contains zero top-k (hits[0]==0); test
  disjoint from pool; labeled set strictly grows by the acquired batch each round.
- **determinism**: identical curves for the same seed.

## 2. Determinism check
Ran the smoke pipeline components twice per domain. Discovery curves and calibration
metrics were **bit-identical** across repeats:
```
molecular:      disc_identical=True  calib_identical=True
superconductor: disc_identical=True  calib_identical=True
DETERMINISM_OK
```
All seeds set (python/numpy/torch/cuda); CPU; `torch.use_deterministic_algorithms`.

## 3. Leakage audit
`leak_ok=True` on all 60 discovery runs (2 domains × 6 strategies × 5 seeds).
The three guards are asserted **in code** every run (`active_loop.py`):
(i) test IDs ∩ pool IDs = ∅; (ii) seed ∩ top-k = ∅ (confirmed by hits[0]=0);
(iii) at round r the model trains only on the union of acquired batches, and
standardization statistics come only from the labeled set.

## 4. Sanity anchors (§7)
| anchor | result |
|---|---|
| `greedy` + ≥1 uncertainty-aware strategy beat `random` on enrichment in ≥1 domain | **HOLDS** — all model strategies beat random in **both** domains (≈5× SC, ≈3× ESOL). |
| `max_var` (pure exploration) underperforms uncertainty-aware strategies at top-k | **HOLDS** — weakest model strategy in both domains (2.94 SC, 1.26 ESOL). |
| ensemble ECE ≤ single-model ECE | **FAILS in both domains** on raw σ. After honest debugging this is a *real* finding, not a bug: on an already-calibrated aleatoric estimate the epistemic term over-disperses. Recalibration resolves it. Reported in RESULTS.md §4, not hacked. |

No NaN/instability encountered (Gaussian-NLL with clamped log-variance + grad
clipping; a NaN-batch skip guard is present in `train_one`).

## 5. Independent review pass (subagent) — findings and fixes
An independent reviewer re-read RESULTS.md against `metrics.json`, the code, and the
four figures. Verdict on first pass: **FIX-NEEDED**. Findings and resolutions:

1. **(Real bug) Unfair single-net baseline.** `predict_single` returned ensemble
   member 0, which was trained on a bootstrap resample (~63% unique rows). This
   biased the single-vs-ensemble calibration comparison toward the ensemble.
   **Fixed:** `run_calibration` now trains a *fair* standalone heteroscedastic net on
   the full fit set (no bootstrap) as the single baseline. Consequence: the earlier
   large ESOL "ensemble calibration win" **disappeared** — the corrected result is
   that the ensemble does not improve raw calibration in either domain. RESULTS.md §4
   was rewritten to reflect this, with the correction called out explicitly.
2. **(Disclosure) Calibration was single-seed** while prose claimed multi-seed/CI.
   **Fixed:** calibration now runs over all 5 seeds and reports mean ± 95% CI.
3. **(Asymmetry) Only the ensemble was recalibrated.** **Fixed:** both the single net
   and the ensemble are now recalibrated symmetrically on a held-out validation slice.
4. **(Wording) "random ≈ 1.0 by construction"** was inaccurate (it is ≈0.9 because
   the hit-free seed is in the denominator). **Fixed** in RESULTS.md §3.
5. **(Wording) "62 runs"** → corrected to "60 discovery runs". **Fixed.**
6. Reviewer confirmed: no test-into-train leakage; ECE/coverage/decomposition
   formulas correct; all figures consistent with the JSON and monotonic.

After fixes, a programmatic prose-vs-JSON check of every headline number in
RESULTS.md/README.md passes (all 16 calibration + 22 discovery/metadata assertions
match `results/metrics.json`).

## 6. Definition-of-Done status
All boxes checked — see README / RESULTS. `pytest` green, determinism holds, both
domains load from public data (superconductor from UCI; molecular from the
documented MoleculeNet fallback after the MolPolySim repo was unreachable), figures
and `metrics.json` regenerated from the corrected pipeline.
