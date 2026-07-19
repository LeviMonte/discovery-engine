/* Experience A — predict & explain. Both domains run live ONNX in the browser.
 * Superconductor: sliders -> 81-vector. Molecule: sketch/SMILES -> RDKit.js
 * featurization -> ONNX. Auto-saves finds; honest synthesis context; metal-gated. */
'use strict';
const C = window.CAVEATS;
const $ = (id) => document.getElementById(id);
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const variance = (a) => { const m = mean(a); return mean(a.map(v => (v - m) ** 2)); };

$('uncertainty-note').innerHTML = '<b>About the ± band.</b> ' + C.uncertainty;
$('sc-tcif').textContent = C.superconductor.tc_if;
$('sc-sigmarank').textContent = C.superconductor.sigma_ranks;
$('mol-sigmarank').textContent = C.molecular.sigma_ranks;
$('mol-footer').innerHTML = '<b>Read me.</b> ' + C.molecular.not_tg + ' ' + C.molecular.fingerprint;

function show(which) {
  $('panel-sc').style.display = which === 'sc' ? '' : 'none';
  $('panel-mol').style.display = which === 'mol' ? '' : 'none';
  $('tab-sc').classList.toggle('primary', which === 'sc');
  $('tab-mol').classList.toggle('primary', which === 'mol');
}
$('tab-sc').onclick = () => show('sc');
$('tab-mol').onclick = () => show('mol');

/* ---------- modals ---------- */
function openModal(id) { $(id).classList.add('show'); }
function closeModal(id) { $(id).classList.remove('show'); }
$('synth-close').onclick = () => closeModal('synth-modal');
$('sketch-close').onclick = () => closeModal('sketch-modal');
$('sketch-cancel').onclick = () => closeModal('sketch-modal');
[ 'synth-modal', 'sketch-modal' ].forEach(id =>
  $(id).addEventListener('click', e => { if (e.target.id === id) closeModal(id); }));

let lastSC = null, lastMol = null;
function openSynth(entry) {
  const s = window.SYNTH.get(entry);
  $('synth-title').textContent = s.title;
  $('synth-body').innerHTML =
    '<ol class="synth-steps">' + s.steps.map(x => `<li>${x}</li>`).join('') + '</ol>' +
    (s.danger ? `<div class="synth-danger">⚠️ ${s.danger}</div>` : '') +
    `<div class="note small"><b>Honesty:</b> ${s.caveat}</div>` +
    `<div class="synth-refs">References: ${s.refs.join('; ')}<br>` +
    `<a href="${s.search}" target="_blank" rel="noopener">↗ literature search</a></div>`;
  openModal('synth-modal');
}
$('sc-synth').onclick = () => lastSC && openSynth(lastSC);
$('mol-synth').onclick = () => lastMol && openSynth(lastMol);

function renderPrediction(p, mu, ale, epi, unit, digits) {
  const total = ale + epi, sigma = Math.sqrt(total);
  $(p + '-mu').textContent = mu.toFixed(digits);
  $(p + '-sigma').textContent = sigma.toFixed(digits) + (p === 'mol' ? unit : '');
  $(p + '-alea').textContent = ale.toFixed(3);
  $(p + '-epi').textContent = epi.toFixed(3);
  const aPct = total > 0 ? ale / total * 100 : 50;
  $(p + '-alea-bar').style.width = aPct.toFixed(1) + '%';
  $(p + '-epi-bar').style.width = (100 - aPct).toFixed(1) + '%';
  return sigma;
}

/* =========================================================
 *  SUPERCONDUCTOR (live ONNX)
 * ========================================================= */
let SC = null, scSession = null, state = null, SC_EP = null;
async function initSC() {
  try {
    ort.env.wasm.wasmPaths = new URL('../vendor/', location.href).href;
    ort.env.wasm.numThreads = 1; ort.env.wasm.simd = true;
    SC = await (await fetch('models/model_meta.json')).json();
    scSession = await ort.InferenceSession.create('models/' + SC.onnx, { executionProviders: ['wasm'] });
    SC_EP = await (await fetch('models/element_props.json')).json();
    try { await StabScreen.load('models/'); } catch (e) { console.warn('stability layer failed', e); }
    state = SC.feat_median.slice();
    $('sc-target').textContent = SC.target.name + ' — ' + SC.target.units;
    buildPresets(); buildSliders();
    $('sc-status').textContent = 'Live model ready — ' + SC.n_members + ' members in your browser.';
    await predictSC();
  } catch (e) { $('sc-status').innerHTML = '<span style="color:var(--danger)">model load failed: ' + e + '</span>'; console.error(e); }
}
function buildPresets() {
  const box = $('sc-presets'); box.innerHTML = '';
  Object.entries(SC.presets).forEach(([name, p]) => {
    const b = document.createElement('button'); b.className = 'btn ghost'; b.textContent = name;
    b.onclick = () => {
      state = p.values.slice(); syncSliders();
      const tc = p.true_tc != null ? p.true_tc.toFixed(1) + ' K' : 'n/a';
      const n = $('sc-preset-note'); n.textContent = `Loaded ${name} (${p.formula}). Measured Tc: ${tc}.`;
      n.classList.add('show'); lastSC = { domain: 'superconductor', id: p.formula, label: p.formula };
      predictSC();
    };
    box.appendChild(b);
  });
}
function buildSliders() {
  const box = $('sc-sliders'); box.innerHTML = '';
  SC.sliders.forEach((s) => {
    const row = document.createElement('div');
    row.innerHTML = `<div class="nm">${s.name} <span class="muted">(imp ${(s.importance*100).toFixed(1)}%)</span></div>
      <div class="slider-row"><input type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${state[s.index]}" data-idx="${s.index}"><span class="val" id="v-${s.index}"></span></div>`;
    const inp = row.querySelector('input');
    inp.oninput = () => { state[s.index] = parseFloat(inp.value); $('v-'+s.index).textContent = state[s.index].toFixed(2);
      $('sc-preset-note').classList.remove('show'); lastSC = { domain:'superconductor', id:'custom composition', label:'custom composition' }; predictSC(); };
    box.appendChild(row);
  });
  syncSliders();
}
function syncSliders() {
  SC.sliders.forEach((s) => { const inp = document.querySelector(`input[data-idx="${s.index}"]`);
    if (inp) inp.value = state[s.index]; const v = $('v-'+s.index); if (v) v.textContent = Number(state[s.index]).toFixed(2); });
}
async function predictSC() {
  const n = SC.n_features, x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = (state[i] - SC.feat_mean[i]) / SC.feat_std[i];
  const out = await scSession.run({ x_std: new ort.Tensor('float32', x, [1, n]) });
  const mu_std = out.mu_std.data, lv_std = out.logvar_std.data, M = SC.n_members;
  const mus = [], va = [];
  for (let m = 0; m < M; m++) { const tm = SC.target_means[m], ts = SC.target_stds[m];
    mus.push(mu_std[m]*ts+tm); va.push(Math.exp(lv_std[m])*ts*ts); }
  const mu = mean(mus), ale = mean(va), epi = variance(mus);
  const sigma = renderPrediction('sc', mu, ale, epi, ' K', 1);
  flagOOD_SC();
  if (!lastSC) lastSC = { domain:'superconductor', id:'custom composition', label:'custom composition' };
  autosave('sc', lastSC, 'Tc', mu, sigma, ale, epi, 'K');
}
function flagOOD_SC() {
  const bad = [];
  SC.sliders.forEach((s) => { const v = state[s.index]; if (v < SC.feat_min[s.index] || v > SC.feat_max[s.index]) bad.push(s.name); });
  const el = $('sc-ood');
  if (bad.length) { el.innerHTML = '⚠️ <b>Out of distribution.</b> ' + C.superconductor.ood + '<br><span class="small muted">Outside training range: ' + bad.join(', ') + '</span>'; el.classList.add('show'); }
  else el.classList.remove('show');
}

/* =========================================================
 *  MOLECULE (live ONNX via RDKit.js)
 * ========================================================= */
let MOL = null, molSession = null, RDKit = null, gridBySmiles = {};
let SAMODEL = null, MOL3D = {}, curStruct = null;
const DKEY = { MolWt:'amw', MolLogP:'CrippenClogP', TPSA:'tpsa', NumHDonors:'NumHBD',
  NumRotatableBonds:'NumRotatableBonds', RingCount:'NumRings', FractionCSP3:'FractionCSP3',
  NumAromaticRings:'NumAromaticRings', HeavyAtomCount:'NumHeavyAtoms' };

async function initMol() {
  try {
    MOL = await (await fetch('models/molecular_meta.json')).json();
    $('mol-target').textContent = MOL.target.name + ' — ' + MOL.target.units;
    RDKit = await initRDKitModule({ locateFile: () => new URL('../vendor/rdkit/RDKit_minimal.wasm', location.href).href });
    molSession = await ort.InferenceSession.create('models/' + MOL.onnx, { executionProviders: ['wasm'] });
    SAMODEL = await (await fetch('models/sascore_model.json')).json();
    try { MOL3D = await (await fetch('models/mol3d.json')).json(); } catch (e) { MOL3D = {}; }
    const grid = await (await fetch('models/grid_molecular.json')).json();
    grid.library.forEach(e => { gridBySmiles[e.smiles] = e; });
    const sel = $('mol-select'); sel.innerHTML = '<option value="">— examples —</option>';
    grid.library.forEach(e => { const o = document.createElement('option'); o.value = e.smiles;
      o.textContent = `${e.name} ${e.novel_scaffold ? '· novel' : ''} (log S ${e.mu})`; sel.appendChild(o); });
    sel.onchange = () => { if (sel.value) { $('mol-smiles').value = sel.value; predictMol(sel.value); } };
    $('mol-status').textContent = 'RDKit + live model ready. Draw, type a SMILES, or pick an example.';
    predictMol('CC(=O)Oc1ccccc1C(=O)O'); // aspirin
  } catch (e) { $('mol-status').innerHTML = '<span style="color:var(--danger)">load failed: ' + e + '</span>'; console.error(e); }
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

function featurizeMol(mol) {
  const nb = MOL.n_bits;
  const fp = mol.get_morgan_fp(JSON.stringify({ radius: MOL.radius, nBits: nb }));
  const d = JSON.parse(mol.get_descriptors());
  const x = new Float32Array(nb + MOL.descriptors.length);
  for (let i = 0; i < nb; i++) x[i] = fp[i] === '1' ? 1 : 0;
  MOL.descriptors.forEach((name, j) => {
    const raw = d[DKEY[name]];
    x[nb + j] = (raw - MOL.desc_mean[j]) / MOL.desc_std[j];
  });
  return x;
}

async function predictMol(smiles) {
  const gate = $('mol-gate'); gate.classList.remove('show');
  smiles = (smiles || '').trim(); if (!smiles) return;
  const mol = RDKit.get_mol(smiles);
  if (!mol || !mol.is_valid()) {
    gate.innerHTML = '❌ Could not parse that SMILES. Check the syntax or use the sketcher.';
    gate.classList.add('show'); if (mol) mol.delete(); return;
  }
  // metal / unsupported-element gate (honest redirect)
  const els = elementsOf(mol);
  const allowed = new Set(MOL.allowed_elements);
  const bad = [...els].filter(e => e && !allowed.has(e));
  if (bad.length) {
    gate.innerHTML = `⛔ <b>${bad.join(', ')} not supported.</b> ` +
      'SMILES + this solubility model only handle organic molecules — metals and ' +
      'inorganic/extended structures can’t be represented this way. For metallic/' +
      'inorganic compositions use the <a href="./">Superconductor tab</a> instead.';
    gate.classList.add('show'); mol.delete(); return;
  }
  const x = featurizeMol(mol);
  const sa = saScoreLocal(mol);
  const svg = mol.get_svg(148, 118);
  const canon = mol.get_smiles(); mol.delete();
  const known0 = gridBySmiles[smiles] || gridBySmiles[canon];
  renderStruct(canon, svg, known0);
  const saExact = (known0 && known0.sascore != null) ? known0.sascore : null;
  $('mol-sa').textContent = (saExact != null ? saExact : sa).toFixed(1);
  $('mol-sa-kind').textContent = saExact != null ? '(exact)' : `(est, R²=${SAMODEL.r2})`;
  const out = await molSession.run({ x_std: new ort.Tensor('float32', x, [1, x.length]) });
  const mu_std = out.mu_std.data, lv_std = out.logvar_std.data, M = MOL.n_members;
  const mus = [], va = [];
  for (let m = 0; m < M; m++) { const tm = MOL.target_means[m], ts = MOL.target_stds[m];
    mus.push(mu_std[m]*ts+tm); va.push(Math.exp(lv_std[m])*ts*ts); }
  const mu = mean(mus), ale = mean(va), epi = variance(mus);
  const known = gridBySmiles[smiles] || gridBySmiles[canon];
  $('mol-name').textContent = (known ? known.name + '  ·  ' : '') + canon;
  if (known) { $('mol-true-wrap').style.display = ''; $('mol-true').textContent = known.true + ' ' + MOL.target.units; }
  else { $('mol-true-wrap').style.display = 'none'; }
  const sigma = renderPrediction('mol', mu, ale, epi, '', 2);
  // OOD via epistemic threshold (novelty)
  const el = $('mol-ood');
  if (epi > MOL.epistemic_ood_threshold || (known && known.novel_scaffold)) {
    el.innerHTML = '⚠️ <b>Novel / uncertain.</b> ' +
      (known && known.novel_scaffold ? C.molecular.ood + ' ' : '') +
      `Epistemic σ² ${epi.toFixed(3)} vs training 90th-pct ${MOL.epistemic_ood_threshold.toFixed(3)} — the ensemble disagrees, so treat this as extrapolation.`;
    el.classList.add('show');
  } else el.classList.remove('show');
  lastMol = { domain: 'molecular', id: canon, label: (known ? known.name : canon), smiles: canon };
  autosave('mol', lastMol, 'log S', mu, sigma, ale, epi, 'log(mol/L)');
}

function saScoreLocal(mol) {
  const m = SAMODEL;
  const fp = mol.get_morgan_fp(JSON.stringify({ radius: 2, nBits: m.nbits }));
  let sum = 0, n = 0;
  for (let i = 0; i < fp.length; i++) if (fp[i] === '1') { sum += m.table[i]; n++; }
  const frag = n ? sum / n : m.default;
  const d = JSON.parse(mol.get_descriptors());
  const f = [frag, d.NumHeavyAtoms, d.NumRings, d.FractionCSP3, d.amw,
    d.NumAromaticRings, d.NumRotatableBonds, d.NumHBD, d.tpsa, d.CrippenClogP];
  let s = m.intercept;
  for (let i = 0; i < m.coef.length; i++) s += m.coef[i] * f[i];
  return Math.max(1, Math.min(10, s));
}

let viewer3d = null;
function renderStruct(canon, svg, known) {
  curStruct = { canon, svg, mb: (known && MOL3D[known.smiles]) || null };
  show2D();
  $('mol-3d').style.display = curStruct.mb ? '' : 'none';
}
function show2D() {
  const box = $('mol-struct'); box.classList.remove('d3'); viewer3d = null;
  box.innerHTML = curStruct ? curStruct.svg : '';
}
function show3D() {
  if (!curStruct || !curStruct.mb || !window.$3Dmol) return show2D();
  const box = $('mol-struct'); box.classList.add('d3'); box.innerHTML = '';
  try {
    viewer3d = $3Dmol.createViewer(box, { backgroundColor: '0x0b1120' });
    viewer3d.addModel(curStruct.mb, 'sdf');
    viewer3d.setStyle({}, { stick: { radius: 0.14 }, sphere: { scale: 0.24 } });
    viewer3d.zoomTo(); viewer3d.render();
  } catch (e) { show2D(); }
}
$('mol-3d').onclick = show3D;
$('mol-2d').onclick = show2D;

$('mol-go').onclick = () => predictMol($('mol-smiles').value);
$('mol-smiles').addEventListener('keydown', e => { if (e.key === 'Enter') predictMol($('mol-smiles').value); });

/* ---------- sketcher (JSME) ---------- */
let jsme = null;
$('mol-draw').onclick = () => {
  openModal('sketch-modal');
  setTimeout(() => {
    if (!jsme && window.__jsmeReady && window.JSApplet) {
      try { jsme = new JSApplet.JSME('jsme-container', '100%', '340px', { options: 'oldlook,star,newLook' }); }
      catch (e) { console.error(e); $('sketch-note').textContent = 'sketcher failed to init: ' + e; }
    }
  }, 60);
};
$('sketch-use').onclick = () => {
  if (!jsme) { closeModal('sketch-modal'); return; }
  const smi = jsme.smiles();
  closeModal('sketch-modal');
  if (smi) { $('mol-smiles').value = smi; predictMol(smi); }
  else $('mol-gate').innerHTML = 'Draw something first.', $('mol-gate').classList.add('show');
};

/* ---------- autosave to leaderboard ---------- */
function autosave(p, entry, metric, mu, sigma, ale, epi, units) {
  const rec = { domain: entry.domain, id: entry.id, label: entry.label,
    smiles: entry.smiles, metric, value: +mu.toFixed(p === 'sc' ? 1 : 2),
    sigma: +sigma.toFixed(2), aleatoric: +ale.toFixed(3), epistemic: +epi.toFixed(3), units };
  window.LB.add(rec);
  $(p + '-saved').textContent = '✓ saved to your leaderboard';
}

/* ---------- superconductor: predict from a chemical formula ---------- */
async function predictSCFormula() {
  const f = $('sc-formula').value.trim(); const note = $('sc-formula-note');
  if (!f) { note.classList.remove('show'); return; }
  const r = await SCModel.fromFormula(scSession, SC, SC_EP, f);
  if (!r.valid) {
    note.classList.add('show');
    note.innerHTML = r.reason === 'unsupported'
      ? `⛔ elements not in the model's table: <b>${r.missing.join(', ')}</b>. The UCI dataset covers 77 elements.`
      : 'Could not parse that formula — try e.g. YBa2Cu3O7.';
    return;
  }
  note.classList.remove('show');
  const sigma = renderPrediction('sc', r.mu, r.ale, r.epi, ' K', 1);
  const el = $('sc-ood');
  if (r.oob > 0) {
    el.innerHTML = `⚠️ <b>Out of distribution.</b> ${r.oob}/81 features are outside the ` +
      `training range — ${C.superconductor.ood}`;
    el.classList.add('show');
  } else el.classList.remove('show');
  $('sc-preset-note').classList.remove('show');
  lastSC = { domain: 'superconductor', id: f, label: f };
  autosave('sc', lastSC, 'Tc', r.mu, sigma, r.ale, r.epi, 'K');
  showStability(f);
}
async function showStability(formula) {
  const box = $('sc-stab'); if (!StabScreen.meta) return;
  box.style.display = ''; box.innerHTML = '<span class="spinner"></span> stability screen…';
  const s = await StabScreen.assess(SC_EP, formula);
  if (!s.valid) { box.style.display = 'none'; return; }
  const sm = s.smact.ok === true ? '✓ charge-balanced (SMACT)'
    : s.smact.ok === null ? 'SMACT: n/a'
    : '✗ not charge-balanced <span class="muted">(normal for metals/intermetallics)</span>';
  box.innerHTML = `<b>🔬 Verification layer.</b> Predicted formation energy ` +
    `<b>${s.eform >= 0 ? '+' : ''}${s.eform.toFixed(2)} ± ${s.sigma.toFixed(2)} eV/atom</b> ` +
    `(${s.favourable ? '<span style="color:var(--accent2)">favourable to form</span>' : '<span style="color:var(--warn)">not favourable (≥0)</span>'}, ` +
    `model R²=${s.r2}) · ${sm}. <span class="muted">A first-pass thermodynamic screen — ` +
    `a proxy for stability, not energy-above-hull. This is what turns a bare Tc guess ` +
    `into a candidate worth taking seriously.</span>`;
}
$('sc-formula-go').onclick = predictSCFormula;
$('sc-formula').addEventListener('keydown', e => { if (e.key === 'Enter') predictSCFormula(); });

/* ---------- molecule -> Tc bridge (heavily caveated) ---------- */
function compositionFromSmiles(smiles) {
  const m = RDKit.get_mol(smiles); if (!m) return null;
  try { m.add_hs_in_place(); } catch (e) {}
  const mb = m.get_molblock().split('\n'); const na = parseInt(mb[3].substr(0, 3));
  const comp = {};
  for (let i = 0; i < na; i++) { const el = mb[4 + i].substr(31, 3).trim(); if (el) comp[el] = (comp[el] || 0) + 1; }
  m.delete(); return comp;
}
$('mol-tc').onclick = async () => {
  const out = $('mol-tc-out');
  if (!lastMol || !lastMol.smiles || !SC_EP || !scSession) { return; }
  const comp = compositionFromSmiles(lastMol.smiles);
  const formula = SCModel.formulaStr(comp);
  const r = await SCModel.fromFormula(scSession, SC, SC_EP, formula);
  if (!r.valid) { out.innerHTML = `Can't bridge: ${r.reason === 'unsupported' ? 'contains ' + r.missing.join(', ') : r.reason}.`; out.classList.add('show'); return; }
  out.innerHTML = `⚡ <b>${r.mu.toFixed(1)} ± ${r.sigma.toFixed(1)} K</b> for composition ${formula}. ` +
    `<br><span class="small">Big caveat: this feeds the molecule's raw <b>element composition</b> ` +
    `into the <b>inorganic</b> superconductor model. ${r.oob}/81 features are out of range. ` +
    `Composition throws away the bonding/structure that governs real molecular superconductivity, ` +
    `and the model never saw organic molecules. This is a curiosity showing the honest bridge — ` +
    `not a claim this molecule superconducts.</span>`;
  out.classList.add('show');
};

/* ---------- boot ---------- */
show('sc');
initSC();
initMol();
