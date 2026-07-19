/* Shared superconductor model helpers: chemical formula -> 81 features (SCFeat)
 * -> standardize -> ONNX ensemble -> Tc with aleatoric/epistemic. Used by the
 * Predict formula box, the molecule->Tc bridge, and the generator's SC mode. */
window.SCModel = (function () {
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const variance = (a) => { const m = mean(a); return mean(a.map(v => (v - m) ** 2)); };

  async function load(base) {
    base = base || 'models/';
    ort.env.wasm.wasmPaths = new URL('../vendor/', location.href).href;
    ort.env.wasm.numThreads = 1; ort.env.wasm.simd = true;
    const meta = await (await fetch(base + 'model_meta.json')).json();
    const session = await ort.InferenceSession.create(base + meta.onnx, { executionProviders: ['wasm'] });
    const EP = await (await fetch(base + 'element_props.json')).json();
    return { meta, session, EP };
  }

  async function predictVector(session, meta, vec) {
    const n = meta.n_features;
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = (vec[i] - meta.feat_mean[i]) / meta.feat_std[i];
    const out = await session.run({ x_std: new ort.Tensor('float32', x, [1, n]) });
    const mu_std = out.mu_std.data, lv_std = out.logvar_std.data, M = meta.n_members;
    const mus = [], va = [];
    for (let m = 0; m < M; m++) { const tm = meta.target_means[m], ts = meta.target_stds[m];
      mus.push(mu_std[m] * ts + tm); va.push(Math.exp(lv_std[m]) * ts * ts); }
    const mu = mean(mus), ale = mean(va), epi = variance(mus);
    // OOD: how many of the 81 raw features fall outside the training range
    let oob = 0;
    for (let i = 0; i < n; i++) if (vec[i] < meta.feat_min[i] || vec[i] > meta.feat_max[i]) oob++;
    return { mu, ale, epi, sigma: Math.sqrt(ale + epi), oob };
  }

  async function fromFormula(session, meta, EP, formula) {
    const comp = window.SCFeat.parseFormula(formula);
    if (!Object.keys(comp).length) return { valid: false, reason: 'empty' };
    const { map, missing } = window.SCFeat.features(comp, EP);
    if (missing.length) return { valid: false, reason: 'unsupported', missing, comp };
    const vec = window.SCFeat.vector(map, meta.feature_names);
    if (vec.some(v => v == null || !isFinite(v))) return { valid: false, reason: 'nan', comp };
    const r = await predictVector(session, meta, vec);
    return { valid: true, comp, formula, ...r };
  }

  // composition dict -> canonical formula string
  function formulaStr(comp) {
    return Object.keys(comp).filter(e => comp[e] > 0).sort()
      .map(e => e + (comp[e] === 1 ? '' : (+comp[e].toFixed(3)))).join('');
  }

  return { load, predictVector, fromFormula, formulaStr, mean, variance };
})();
