/* Single source of truth for honesty copy, pulled from RESULTS.md.
 * Imported (as a plain <script>) by every page. Do NOT soften this language in
 * the interactive UI -- it was deliberately honest in the writeup. */
window.CAVEATS = {
  project:
    "This is a demo of a benchmarking result, not a materials-discovery product. " +
    "All labels shown already existed in public datasets; nothing here claims to " +
    "have found a new material.",

  superconductor: {
    target: "Critical temperature Tc (K)",
    tc_if:
      "The model predicts Tc *assuming the compound superconducts*. It was trained " +
      "only on known superconductors and CANNOT tell you whether an arbitrary " +
      "composition superconducts at all.",
    composition:
      "Composition-only model (81 numeric descriptors) — it sees no crystal " +
      "structure, no processing, no measurement conditions.",
    ood:
      "This composition is outside the range of compositions the model was trained " +
      "on. Treat this prediction with extra skepticism — the model is extrapolating.",
    sigma_ranks:
      "On this domain the predicted uncertainty does track error reasonably well " +
      "(Spearman σ vs |error| ≈ 0.75), so a larger band is a meaningful 'less sure'.",
  },

  molecular: {
    target: "Aqueous solubility log S (log mol/L)",
    not_tg:
      "Heads up on provenance: the intended transfer target was polymer glass-" +
      "transition (Tg) data from MolPolySim, but that source was unreachable, so " +
      "the model actually trained in Phases 0–7 predicts ESOL aqueous solubility. " +
      "That is what this panel shows — not Tg, and nothing about polymers.",
    fingerprint:
      "This uses the documented Morgan-fingerprint + MLP fallback, not a graph " +
      "neural network. A graph model would likely be more accurate.",
    precomputed:
      "This panel now runs LIVE in your browser: RDKit.js computes the same Morgan " +
      "fingerprint + descriptors the model was trained on (verified bit-for-bit " +
      "against the Python pipeline), then the ONNX ensemble predicts. Organic " +
      "molecules only.",
    ood:
      "Novel scaffold: this molecule's Bemis–Murcko scaffold is not represented in " +
      "the training set (it came from the held-out scaffold-split test set). The " +
      "prediction is a genuine out-of-distribution test.",
    sigma_ranks:
      "Honesty note: on this solubility dataset the predicted uncertainty does NOT " +
      "rank error (Spearman σ vs |error| ≈ 0). So treat the band here as an average " +
      "spread, not a reliable 'this specific one is uncertain' signal.",
  },

  uncertainty:
    "Every prediction carries μ ± σ. Total σ splits into aleatoric (irreducible " +
    "noise the model estimates) and epistemic (disagreement across the 5 ensemble " +
    "members — shrinks with more data). Finding from Phases 0–7: the deep ensemble " +
    "did not improve raw calibration over a fair single net; post-hoc recalibration " +
    "is what delivers good coverage.",

  replay:
    "This replays retrospective active-learning runs on public data where the top " +
    "materials were already known — it is not a live discovery run. Key honest " +
    "finding: uncertainty-aware acquisition (UCB/Thompson/EI) did NOT beat plain " +
    "greedy exploitation; it is the calibrated model mean, not the uncertainty " +
    "bonus, that buys the ~3–5× sample-efficiency over random search.",
};
