# DATA_SOURCES

Every dataset used, its origin, license, and row count. All labels are public
and pre-existing; the active-learning simulation *hides* them and re-requests
them retrospectively. **No data was fabricated.** Where a source was
unreachable, that is recorded here and a documented fallback was used.

---

## Domain A — Inorganic superconductors (primary target)

| field | value |
|---|---|
| Name | UCI "Superconductivty Data" (Hamidieh 2018), dataset id **464** |
| File used | `data/superconductor/train.csv` |
| Rows | **21,263** |
| Features | 81 precomputed composition descriptors |
| Target | `critical_temp` (Tc), in Kelvin; range 0.00021–185 K, mean 34.4 K |
| Companion | `unique_m.csv` — element-fraction columns + chemical `material` formula |
| URL (used) | `https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip` |
| Fallback URLs | UCI legacy mirror `.../ml/machine-learning-databases/00464/…`; then `matminer.datasets.load_dataset` |
| License | UCI Machine Learning Repository — Creative Commons Attribution 4.0 (CC BY 4.0) |
| Citation | Hamidieh, K. (2018). *A data-driven statistical model for predicting the critical temperature of a superconductor.* Computational Materials Science, 154, 346–354. |

Notes: this domain uses a **descriptor/composition** model (no molecular graph —
superconductors are inorganic compositions). The model consumes the 81 numeric
columns, z-scored on training statistics only.

---

## Domain B — Molecular graphs (transfer result)

### Intended primary (not available in this run)

| field | value |
|---|---|
| Name | MolPolySim polymer glass-transition (Tg) dataset |
| Intended access | `git clone https://github.com/LeviMonte/MolPolySim` then run its `fetch_data.py` |
| **Status in this run** | **UNREACHABLE.** `git clone` returned `could not read Username for 'https://github.com'` (repository private or non-existent from this environment). Per the build spec we did **not** block on it and fell through to the documented MoleculeNet fallback below. |

### Fallback used (primary in practice)

| field | value |
|---|---|
| Name | MoleculeNet **ESOL** (Delaney) aqueous solubility |
| File used | `data/esol.csv` |
| Rows | **1,128** (all with valid RDKit-parseable SMILES) |
| Target | `measured log solubility in mols per litre`, log(mol/L); range −11.6 to 1.58 |
| SMILES column | `smiles` |
| URL (used) | `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv` |
| License | MIT (MoleculeNet / DeepChem) |
| Citation | Delaney, J. S. (2004). *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure.* J. Chem. Inf. Comput. Sci., 44(3), 1000–1005. Wu et al. (2018), *MoleculeNet*, Chem. Sci. |

### Secondary molecular set (downloaded, available, not used as primary)

| field | value |
|---|---|
| Name | MoleculeNet **Lipophilicity** (ChEMBL logD) |
| File | `data/lipo.csv` (4,200 rows) |
| Target | `exp` (experimental logD) |
| Purpose | Available as an alternate Domain-B target (`config: data.dataset: lipophilicity`). |

Notes: this domain uses **molecular-graph-derived** features. The reference build
uses the spec's documented **fingerprint fallback** (RDKit Morgan bits + a small
set of continuous descriptors) rather than a `torch_geometric` GINE network, so
the whole pipeline runs on CPU without the fragile graph stack. The uncertainty +
active-learning framework is identical to a graph model; only the featurizer
differs. A **Bemis–Murcko scaffold split** provides the held-out test set so no
scaffold is shared between train and test.

---

## Why ESOL instead of polymer Tg?

The intended transfer target was MolPolySim's polymer Tg data, to mirror the
provenance project. That repository was not reachable from the build
environment (see above), so we used ESOL: a well-characterised, openly licensed
MoleculeNet regression set with SMILES + a continuous experimental target. The
scientific point of Domain B is unchanged — to show the *method transfers to
graph-structured molecular data from a completely different chemistry* than the
inorganic superconductors. ESOL serves that purpose; the chosen property is
secondary to the transfer demonstration.
