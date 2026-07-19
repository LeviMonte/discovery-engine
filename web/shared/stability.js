/* Verification layer: composition -> formation energy (eV/atom) via a heteroscedastic
 * ensemble (R^2~0.94 on held-out experimental data), plus the SMACT plausibility
 * screen. Reuses the exact in-browser 81-feature featurizer (SCFeat). Lower / negative
 * formation energy = more favourable to form. This is a first-pass thermodynamic
 * screen (a proxy for stability), NOT energy-above-hull. */
window.StabScreen = (function () {
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const variance = (a) => { const m = mean(a); return mean(a.map(v => (v - m) ** 2)); };
  let meta = null, session = null;

  async function load(base) {
    base = base || 'models/';
    meta = await (await fetch(base + 'stability_meta.json')).json();
    session = await ort.InferenceSession.create(base + meta.onnx, { executionProviders: ['wasm'] });
    await window.SMACT.load(base);
    return { meta };
  }

  async function assess(EP, formula) {
    const comp = window.SCFeat.parseFormula(formula);
    const { map, missing } = window.SCFeat.features(comp, EP);
    if (missing.length) return { valid: false, missing };
    const vec = window.SCFeat.vector(map, meta.feature_names);
    const n = vec.length, x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = (vec[i] - meta.feat_mean[i]) / meta.feat_std[i];
    const out = await session.run({ x_std: new ort.Tensor('float32', x, [1, n]) });
    const mu_std = out.mu_std.data, lv_std = out.logvar_std.data, M = meta.n_members;
    const mus = [], va = [];
    for (let m = 0; m < M; m++) { const tm = meta.target_means[m], ts = meta.target_stds[m];
      mus.push(mu_std[m] * ts + tm); va.push(Math.exp(lv_std[m]) * ts * ts); }
    const eform = mean(mus), sigma = Math.sqrt(mean(va) + variance(mus));
    const smact = window.SMACT.check(comp);
    return { valid: true, eform, sigma, favourable: eform < meta.favourable_threshold, smact,
      r2: meta.r2 };
  }
  return { load, assess, get meta() { return meta; } };
})();
