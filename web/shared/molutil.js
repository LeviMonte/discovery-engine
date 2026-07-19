/* Shared molecular stack: RDKit.js + ONNX ensemble + SAscore + 3D/2D structure.
 * Used by the Predict and Generator pages. All math mirrors the Python pipeline,
 * verified in Node to agree within 5e-4 log units. */
window.MolStack = (function () {
  let promise = null;
  const DKEY = { MolWt:'amw', MolLogP:'CrippenClogP', TPSA:'tpsa', NumHDonors:'NumHBD',
    NumRotatableBonds:'NumRotatableBonds', RingCount:'NumRings', FractionCSP3:'FractionCSP3',
    NumAromaticRings:'NumAromaticRings', HeavyAtomCount:'NumHeavyAtoms' };

  function load(base) {
    if (promise) return promise;
    base = base || 'models/';
    promise = (async () => {
      ort.env.wasm.wasmPaths = new URL('../vendor/', location.href).href;
      ort.env.wasm.numThreads = 1; ort.env.wasm.simd = true;
      const meta = await (await fetch(base + 'molecular_meta.json')).json();
      const RDKit = await initRDKitModule({ locateFile: () => new URL('../vendor/rdkit/RDKit_minimal.wasm', location.href).href });
      const session = await ort.InferenceSession.create(base + meta.onnx, { executionProviders: ['wasm'] });
      const sa = await (await fetch(base + 'sascore_model.json')).json();
      const grid = await (await fetch(base + 'grid_molecular.json')).json();
      let mol3d = {}; try { mol3d = await (await fetch(base + 'mol3d.json')).json(); } catch (e) {}
      const gridBySmiles = {}; grid.library.forEach(e => gridBySmiles[e.smiles] = e);
      return { RDKit, session, meta, sa, grid, gridBySmiles, mol3d };
    })();
    return promise;
  }

  function elementsOf(mol) {
    try {
      const mb = mol.get_molblock().split('\n');
      const na = parseInt(mb[3].substr(0, 3));
      const els = new Set();
      for (let i = 0; i < na; i++) els.add(mb[4 + i].substr(31, 3).trim());
      return els;
    } catch (e) { return new Set(); }
  }

  function descriptors(mol) { return JSON.parse(mol.get_descriptors()); }

  function saScore(S, mol) {
    const m = S.sa;
    const fp = mol.get_morgan_fp(JSON.stringify({ radius: 2, nBits: m.nbits }));
    let sum = 0, n = 0;
    for (let i = 0; i < fp.length; i++) if (fp[i] === '1') { sum += m.table[i]; n++; }
    const frag = n ? sum / n : m.default;
    const d = descriptors(mol);
    const feats = [frag, d.NumHeavyAtoms, d.NumRings, d.FractionCSP3, d.amw,
      d.NumAromaticRings, d.NumRotatableBonds, d.NumHBD, d.tpsa, d.CrippenClogP];
    let s = m.intercept;
    for (let i = 0; i < m.coef.length; i++) s += m.coef[i] * feats[i];
    return Math.max(1, Math.min(10, s));
  }

  function featurize(S, mol) {
    const nb = S.meta.n_bits;
    const fp = mol.get_morgan_fp(JSON.stringify({ radius: S.meta.radius, nBits: nb }));
    const d = descriptors(mol);
    const x = new Float32Array(nb + S.meta.descriptors.length);
    for (let i = 0; i < nb; i++) x[i] = fp[i] === '1' ? 1 : 0;
    S.meta.descriptors.forEach((name, j) => { x[nb + j] = (d[DKEY[name]] - S.meta.desc_mean[j]) / S.meta.desc_std[j]; });
    return x;
  }

  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const variance = (a) => { const m = mean(a); return mean(a.map(v => (v - m) ** 2)); };

  async function predict(S, smiles) {
    const mol = S.RDKit.get_mol(smiles || '');
    if (!mol || !mol.is_valid()) { if (mol) mol.delete(); return { valid: false, reason: 'unparseable' }; }
    const els = elementsOf(mol);
    const allowed = new Set(S.meta.allowed_elements);
    const bad = [...els].filter(e => e && !allowed.has(e));
    if (bad.length) { const canon = mol.get_smiles(); mol.delete(); return { valid: false, reason: 'element', badElems: bad, canon }; }
    const x = featurize(S, mol);
    const sascore = saScore(S, mol);
    const canon = mol.get_smiles(); mol.delete();
    const out = await S.session.run({ x_std: new ort.Tensor('float32', x, [1, x.length]) });
    const mu_std = out.mu_std.data, lv_std = out.logvar_std.data, M = S.meta.n_members;
    const mus = [], va = [];
    for (let m = 0; m < M; m++) { const tm = S.meta.target_means[m], ts = S.meta.target_stds[m];
      mus.push(mu_std[m]*ts+tm); va.push(Math.exp(lv_std[m])*ts*ts); }
    const mu = mean(mus), ale = mean(va), epi = variance(mus);
    const known = S.gridBySmiles[smiles] || S.gridBySmiles[canon];
    return { valid: true, canon, mu, ale, epi, sigma: Math.sqrt(ale + epi), sascore,
      ood: epi > S.meta.epistemic_ood_threshold, known };
  }

  function svg2d(S, smiles) {
    const mol = S.RDKit.get_mol(smiles); if (!mol) return '';
    const s = mol.get_svg(260, 200); mol.delete(); return s;
  }

  return { load, predict, saScore, featurize, svg2d, elementsOf };
})();
