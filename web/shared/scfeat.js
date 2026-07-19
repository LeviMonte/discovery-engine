/* In-browser reconstruction of the UCI (Hamidieh 2018) 81-feature composition
 * featurization. Validated in Node to reproduce train.csv to < 1e-6, so the
 * superconductor ONNX model receives exactly the features it was trained on.
 * Element property values are recovered exactly from the dataset (element_props.json). */
window.SCFeat = (function () {
  const EPS = 1e-12;

  function parseFormula(s) {
    const comp = {};
    const re = /([A-Z][a-z]?)(\d*\.?\d*)/g;
    let m;
    while ((m = re.exec(s)) !== null) {
      if (!m[1]) continue;
      const amt = m[2] === '' ? 1 : parseFloat(m[2]);
      comp[m[1]] = (comp[m[1]] || 0) + amt;
    }
    return comp;
  }

  function stats(t, p) {
    const n = t.length;
    const sum = t.reduce((a, b) => a + b, 0);
    const mean = sum / n;
    const wmean = t.reduce((a, v, i) => a + p[i] * v, 0);
    const gmean = Math.exp(t.reduce((a, v) => a + Math.log(Math.max(v, EPS)), 0) / n);
    const wgmean = Math.exp(t.reduce((a, v, i) => a + p[i] * Math.log(Math.max(v, EPS)), 0));
    const f = t.map(v => v / sum);
    const entropy = -f.reduce((a, v) => a + v * Math.log(Math.max(v, EPS)), 0);
    let gsum = 0; const pw = t.map((v, i) => { const x = p[i] * v; gsum += x; return x; });
    const g = pw.map(v => v / gsum);
    const wentropy = -g.reduce((a, v) => a + v * Math.log(Math.max(v, EPS)), 0);
    const range = Math.max(...t) - Math.min(...t);
    const wrange = Math.max(...pw) - Math.min(...pw);
    const std = Math.sqrt(t.reduce((a, v) => a + (v - mean) ** 2, 0) / n);
    const wstd = Math.sqrt(t.reduce((a, v, i) => a + p[i] * (v - wmean) ** 2, 0));
    return { mean, wtd_mean: wmean, gmean, wtd_gmean: wgmean, entropy,
      wtd_entropy: wentropy, range, wtd_range: wrange, std, wtd_std: wstd };
  }

  /* comp: {element: amount}. EP: element_props.json. Returns {map, missing}. */
  function features(comp, EP) {
    const els = Object.keys(comp).filter(e => comp[e] > 0 && EP.elements[e]);
    const missing = Object.keys(comp).filter(e => comp[e] > 0 && !EP.elements[e]);
    const map = { number_of_elements: els.length };
    if (!els.length) return { map, missing };
    const amts = els.map(e => comp[e]);
    const psum = amts.reduce((a, b) => a + b, 0);
    const p = amts.map(a => a / psum);
    for (const prop of EP.props) {
      const t = els.map(e => EP.elements[e][prop]);
      const st = stats(t, p);
      for (const s of EP.stats) map[`${s}_${prop}`] = st[s];
    }
    return { map, missing };
  }

  function vector(map, featureNames) { return featureNames.map(n => map[n]); }

  return { parseFormula, features, vector };
})();
