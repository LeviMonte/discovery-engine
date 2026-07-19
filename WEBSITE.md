# WEBSITE.md — public interactive demo (Phase 8)

A zero-backend, zero-API-key static site layered on top of the Phases 0–7
research repo. It turns the trained deep ensembles and the saved active-learning
results into two things a stranger can open and understand in under two minutes.

## Live URL

> **Live:** https://levimonte.github.io/discovery-engine/
> **Repo:** https://github.com/LeviMonte/discovery-engine

Deployed from `web/` via GitHub Actions (`.github/workflows/deploy-pages.yml`) on
every push to `main` that touches `web/`. To publish:

```bash
GH_USER=<your-username> REPO=discovery-engine bash scripts/deploy.sh
```

then (if not auto-enabled) set repo **Settings → Pages → Source: GitHub Actions**.

## What's on it

**Landing (`web/index.html`)** — one-screen framing of the three findings, links
to both experiences.

**A · Predict & explain (`web/predict/`)**
- *Superconductor* — 8 most-important composition sliders (RandomForest
  importance), 3 real-composition presets (YBCO-like, MgB2-like, a random training
  point), live μ ± σ with the aleatoric/epistemic split as a two-tone bar, and a
  visible **out-of-distribution warning** that fires when any slider leaves the
  training range.
- *Molecule* — searchable curated library of real ESOL molecules; shows predicted
  log S ± σ, the split, the **true measured value**, and a **novel-scaffold**
  (OOD) flag for held-out test molecules.

**B · Discovery replay (`web/replay/`)** — overlay any subset of the six
acquisition strategies, scrub/animate the discovery curves growing
measurement-by-measurement with 95% CI bands over 5 seeds, and read live
enrichment / AUDC / acquisitions-to-K metrics that match `results/metrics.json`
exactly at the final frame.

## Pages (tabs)

- **Predict** — live inference for both domains, sketcher, **synthesizability (SAscore)**,
  a **chemical-formula input** for superconductors (type `YBa2Cu3O7` → live Tc), and a
  **molecule→Tc bridge** ("Tc if it superconducted?", heavily OOD-flagged). The 81 UCI
  composition features are reconstructed in-browser (`shared/scfeat.js`) from an element
  table recovered exactly from the dataset — validated to match `train.csv` to <1e-6.
  So superconductor prediction now works from any real formula, not just sliders.
  and a **2D/3D structure viewer** (3Dmol.js; precomputed conformers for library
  molecules, RDKit 2D for novel ones — RDKit.js has no in-browser 3D embedding).
- **Verification layer** — the honest "is this candidate real?" filter for
  superconductors, running live in-browser: (1) a composition→**formation-energy**
  heteroscedastic ensemble (trained on real *experimental* formation enthalpies,
  held-out **R²=0.94, MAE=0.14 eV/atom**; NaCl→−2.06, TiO₂→−3.14, MgB₂→−0.28 eV/atom
  all match reality), and (2) a **SMACT-style charge-neutrality/electronegativity
  screen**. Shown on the Predict formula box and used in the generator to surface
  thermodynamically **favourable** high-Tc candidates first. Honest caveats: formation
  energy is a proxy, **not** energy-above-hull; SMACT legitimately rejects
  metals/intermetallics (charge neutrality is an ionic concept), so a SMACT "fail" is
  informational, not a validity verdict.
- **Generate** — inverse design in BOTH domains now: an in-browser evolutionary search
  either mutates molecules (RDKit-validated) or **mutates chemical formulas** for
  superconductors (live 81-feature + ONNX Tc), scoring each with the ensemble and
  ranking by objective (property, UCB explore, synthesizability). Candidates are
  labelled model hypotheses, not discoveries. Save / synthesis-context per hit.
- **Retrosynthesis** — a *client* for real retrosynthesis backends (AiZynthFinder /
  ASKCOS / IBM RXN). With no backend connected it makes zero external calls and shows
  the honest setup recipe; connect your own endpoint and it renders suggested routes
  (labelled unverified, organic-only). No route is ever fabricated.
- **Replay** — animated discovery curves from the saved runs.
- **Confidence & novelty** — real calibration diagnostics (reliability diagram,
  ECE, does σ rank error, aleatoric vs epistemic) straight from the held-out
  results; explains how out-of-distribution is flagged.
- **Leaderboard** — your predictions auto-save to `localStorage` and rank against
  real benchmark reference materials; click any row for synthesis context; submit
  your best find via a prefilled GitHub issue.

## Architecture: both domains now run LIVE ONNX in-browser

| domain | approach | notes |
|---|---|---|
| **Superconductor** | **Live ONNX.** 5 members → `superconductor_ensemble.onnx`; sliders → 81-vector → standardize in JS → ONNX → decode. | Parity vs PyTorch **1.5e-6**. |
| **Molecule** | **Live ONNX via RDKit.js.** A drawn/typed SMILES is featurized in the browser by RDKit-WASM (Morgan-1024 + 9 descriptors), standardized, and run through `molecular_ensemble.onnx`. | Phase 9 upgrade from the old precomputed grid. RDKit.js reproduces our fingerprints **bit-for-bit** and 9/10 descriptors exactly; the browser model is trained on those 9 JS-reproducible descriptors (only `NumHAcceptors` was dropped, by definition mismatch). Full chain (RDKit.js→standardize→ONNX→decode) verified in Node to match the Python pipeline within **0.0005 log units**. |

**Molecule sketcher.** JSME (vendored, `web/vendor/jsme/`) → SMILES → live predict.
Metals / non-organic atoms are detected and the user is redirected to the
superconductor tab — SMILES + the organic-solubility model genuinely cannot
represent metallic/inorganic structures, and we refuse to fake a number.

**Synthesis context (not recipes).** Clicking a compound shows the *established
synthesis class* with real citations for famous materials (YBCO solid-state route,
MgB₂ elemental reaction, superhydride high-pressure, etc.), and for model
predictions states plainly that no validated route exists + a literature link.
No per-compound procedure is fabricated.

Everything runs from static files. Vendored libraries (`web/vendor/`):
`onnxruntime-web@1.14.0` (single-threaded wasm — no COOP/COEP needed),
`chart.js@4.4.3`, `@rdkit/rdkit` MinimalLib (WASM), and JSME. No runtime CDN calls,
no analytics, no data collection.

## Build / regenerate the assets

From the repo root (needs the Phases 0–7 environment — torch, rdkit, sklearn, onnx):

```bash
python scripts/export_onnx.py        # superconductor ONNX + stats cache + parity check
python scripts/build_meta.py         # slider importances + real-composition presets -> model_meta.json
python scripts/build_grid.py         # molecular curated-library predictions -> grid_molecular.json
python scripts/export_replay_data.py # results/metrics.json -> replay/data/discovery_runs.json
```

## Run locally (for reviewers who'd rather inspect than trust the deploy)

```bash
cd web && python -m http.server 8099
# open http://localhost:8099/
```

## Known limitations

- **Leaderboard is local, not global** — finds save to your browser only; there is
  no shared backend (static hosting). "Submit" opens a GitHub issue for manual
  review. Stated on the page.
- **Synthesis is context, not a recipe** — real referenced routes for known material
  classes; explicit "no validated route" for predictions. Nothing fabricated.
- **Sketcher is organic-only** — metals/inorganic structures are rejected and
  redirected to the superconductor tab (SMILES cannot represent them).
- **Molecular model is ESOL solubility, not polymer Tg** — the intended MolPolySim
  Tg data was unreachable in the build environment (see `DATA_SOURCES.md`). The UI
  says so plainly; no polymer claims are made.
- **Superconductor sliders expose 8 of 81 features**; the other 73 are held at the
  training median, so a preset's prediction reflects sliders + median, not the full
  real composition. Adequate for intuition, not for quantitative use.
- **Uncertainty caveats carried over verbatim**: Tc is "Tc if it superconducts";
  σ ranks error on superconductors but *not* on ESOL; the ensemble does not improve
  raw calibration over a fair single net. All surfaced in the UI, not buried here.
- The demo visualizes an already-validated benchmark; it is **not** a
  model-serving product and makes **no** discovery claims.

## What did NOT need a fallback

The superconductor domain runs genuine live in-browser inference (not a lookup).
Only the molecular domain uses the precomputed-library approach.
