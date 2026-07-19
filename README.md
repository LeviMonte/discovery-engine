# Uncertainty-Driven Materials Discovery Engine

A controlled, **retrospective** benchmark asking one question:

> Does calibrated predictive uncertainty make materials discovery more
> sample-efficient than uncertainty-blind search — and does the same method
> transfer across chemistries?

It is a **methods / benchmarking** project, **not** a claim to have discovered a
new material. All labels are public; the active-learning loop hides them and
re-requests them one batch at a time, so every result is verifiable.

Two very different domains are tested with identical machinery:

- **Domain A — inorganic superconductors** (UCI Superconductivty Data, 21,263
  compositions; predict critical temperature Tc).
- **Domain B — organic molecules** (MoleculeNet ESOL, 1,128 molecules; predict
  aqueous solubility) — the *transfer* check.

## Headline findings (see `RESULTS.md` for the honest, caveated version)

1. **Discovery efficiency:** model-guided acquisition finds the top 5% of
   materials **~5× (superconductors) / ~3× (ESOL)** more efficiently than random.
   *But* adding an uncertainty bonus (UCB/Thompson/EI) does **not** beat pure
   greedy exploitation — a genuine negative result for the strong hypothesis.
2. **Calibration:** against a *fair* single heteroscedastic net, the deep ensemble
   does **not** improve raw calibration — in both domains it slightly over-disperses
   (superconductor ECE 0.022→0.037, ESOL 0.086→0.117), and post-hoc recalibration is
   what delivers good coverage. The ensemble's real gains are small accuracy and
   error-ranking improvements plus the epistemic term needed for exploration. (An
   earlier draft's big ESOL "ensemble win" was an artifact of an unfair bootstrap
   baseline, caught in review and corrected — see RESULTS.md §4.)
3. **Transfer:** the ranking of strategies is preserved across both chemistries.

## Install

```bash
pip install -r requirements.txt          # torch (CPU ok), rdkit, sklearn, scipy, matplotlib, pyyaml, pytest
```

## One-command reproduction

```bash
python run.py --domain all --seeds 5          # full run -> results/metrics.json + figures
python run.py --domain all --smoke            # fast self-test config (tiny data, M=2, 2 seeds)
```

Datasets download automatically on first run (with documented fallbacks; see
`DATA_SOURCES.md`). On a CPU-only box the full run is compute-heavy; a
**checkpointed** runner is provided that survives interruptions by processing the
experiment as a persisted work queue:

```bash
python run_chunked.py --seeds 5 --budget 34   # re-invoke until it prints ALL_DONE
```

## Tests

```bash
pytest -q            # 19 tests: featurizer, scaffold-split disjointness,
                     # train-only standardization, leakage guards, ensemble
                     # decomposition, calibration, determinism, acquisitions
```

## Repo layout

```
discovery-engine/
  README.md  DATA_SOURCES.md  RESULTS.md  VERIFICATION.md  requirements.txt
  config/                 superconductor.yaml, molecular.yaml
  src/discovery/
    data.py               download/load both domains (+ fallbacks, no fabrication)
    featurize.py          composition feats, Morgan fingerprints, scaffold split, z-score
    models.py             heteroscedastic MLP, Gaussian NLL, DeepEnsemble (M=5)
    uncertainty.py        aleatoric/epistemic split, ECE, reliability, recalibration
    acquire.py            random/greedy/ucb/thompson/ei/max_var
    active_loop.py        pool-based AL simulation + leakage guards + AUDC/enrichment
    experiments.py        multi-seed orchestration -> metrics.json
    plots.py              discovery curves + calibration diagrams
    utils.py              seeding, config, paths
  tests/                  pytest
  results/                metrics.json, figures/*.png
  run.py                  CLI (single process)
  run_chunked.py          checkpointed CLI (survives interruption)
```

## Provenance

The model design (heteroscedastic networks, ±1σ on every prediction, "never ship
an unvalidated claim", report negative results plainly) is inherited from
**MolPolySim**. This project is the research layer that MolPolySim lacked: a
direct test of whether its uncertainty actually helps decide *what to measure
next*. The answer, honestly, is "the calibrated model helps a lot; the
uncertainty-aware acquisition rule specifically does not beat greedy here."

## Honesty rules enforced in code

- Held-out test set never seen by any model; seed excludes the top-k; the model
  trains only on acquired points (all asserted, `leak_ok` logged).
- No fabricated data: if a source is unreachable the code raises and records what
  it tried (see `DATA_SOURCES.md` for the MolPolySim clone failure).
- Every prediction carries ±1σ; unvalidated behaviour is labelled, not hidden.
- Multiple seeds, 95% CIs, no cherry-picking — negative results reported plainly.
