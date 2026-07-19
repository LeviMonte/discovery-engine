/* Generative candidate finder — a simple evolutionary search over molecules,
 * scored by the live ensemble. Honest framing: candidates are model hypotheses. */
'use strict';
const $ = (id) => document.getElementById(id);
let S = null;
$('honesty').innerHTML = '<b>Honesty.</b> ' + window.CAVEATS.project +
  ' Structures here are produced by mutating known molecules and scored by the model; ' +
  'predicted solubility and the SAscore estimate are model outputs, and none of these ' +
  'are validated compounds. Treat the list as leads to investigate, not discoveries.';

$('synth-close').onclick = () => $('synth-modal').classList.remove('show');
$('synth-modal').addEventListener('click', e => { if (e.target.id === 'synth-modal') $('synth-modal').classList.remove('show'); });
function openSynth(entry) {
  const s = window.SYNTH.get(entry);
  $('synth-title').textContent = s.title;
  $('synth-body').innerHTML = '<ol class="synth-steps">' + s.steps.map(x => `<li>${x}</li>`).join('') + '</ol>' +
    (s.danger ? `<div class="synth-danger">⚠️ ${s.danger}</div>` : '') +
    `<div class="note small"><b>Honesty:</b> ${s.caveat}</div>` +
    `<div class="synth-refs">References: ${s.refs.join('; ')}<br><a href="${s.search}" target="_blank" rel="noopener">↗ literature search</a></div>`;
  $('synth-modal').classList.add('show');
}

/* ---------- mutation operators (string-level, RDKit-validated) ---------- */
const ALIPH = ['C', 'N', 'O', 'S'], AROM = ['c', 'n', 'o', 's'], BRANCH = ['(C)', '(F)', '(Cl)', '(O)', '(N)'];
const rint = (n) => Math.floor(Math.random() * n);

function mutate(smi) {
  const out = new Set();
  const positions = (pred) => [...smi].map((c, i) => pred(c) ? i : -1).filter(i => i >= 0);
  for (let t = 0; t < 6; t++) {
    const kind = rint(3);
    if (kind === 0) {                       // swap aliphatic atom
      const p = positions(c => ALIPH.includes(c)); if (!p.length) continue;
      const i = p[rint(p.length)]; const rep = ALIPH[rint(ALIPH.length)];
      out.add(smi.slice(0, i) + rep + smi.slice(i + 1));
    } else if (kind === 1) {                // swap aromatic atom
      const p = positions(c => AROM.includes(c)); if (!p.length) continue;
      const i = p[rint(p.length)]; const rep = AROM[rint(AROM.length)];
      out.add(smi.slice(0, i) + rep + smi.slice(i + 1));
    } else {                                 // add a branch after an aliphatic C
      const p = positions(c => c === 'C'); if (!p.length) continue;
      const i = p[rint(p.length)]; const br = BRANCH[rint(BRANCH.length)];
      out.add(smi.slice(0, i + 1) + br + smi.slice(i + 1));
    }
  }
  return [...out];
}

function validCanon(smi) {
  const m = S.RDKit.get_mol(smi);
  if (!m || !m.is_valid()) { if (m) m.delete(); return null; }
  const els = window.MolStack.elementsOf(m);
  const allowed = new Set(S.meta.allowed_elements);
  if ([...els].some(e => e && !allowed.has(e))) { m.delete(); return null; }
  const c = m.get_smiles(); m.delete(); return c;
}

/* ---------- GA ---------- */
let scored = {};   // canon -> {result, fitness}
function fitness(r) {
  const obj = $('obj').value; const sp = $('synthpref').value === '1';
  let f = obj === 'max' ? r.mu : obj === 'min' ? -r.mu : r.mu + 1.5 * r.sigma;
  if (sp && r.sascore > 4) f -= (r.sascore - 4) * 0.3;
  return f;
}
async function scoreSmiles(canon) {
  if (scored[canon]) return scored[canon];
  const r = await window.MolStack.predict(S, canon);
  if (!r.valid) { scored[canon] = null; return null; }
  const rec = { result: r, fitness: fitness(r) };
  scored[canon] = rec; return rec;
}

async function run() {
  $('run').disabled = true; scored = {};
  const obj = $('obj').value;
  const seedSort = (a, b) => obj === 'min' ? a.true - b.true : b.true - a.true;
  let pop = S.grid.library.slice().sort(seedSort).slice(0, 24).map(e => e.smiles);
  pop = [...new Set(pop.map(validCanon).filter(Boolean))];
  const GENS = 8;
  for (let g = 0; g < GENS; g++) {
    for (const c of pop) await scoreSmiles(c);
    // rank current pop
    pop.sort((a, b) => (scored[b]?.fitness ?? -1e9) - (scored[a]?.fitness ?? -1e9));
    const parents = pop.slice(0, 12);
    const children = new Set();
    for (const p of parents) for (const m of mutate(p)) { const c = validCanon(m); if (c) children.add(c); }
    // new population: parents + children + a few fresh seeds for diversity
    const fresh = S.grid.library.slice().sort(() => Math.random() - 0.5).slice(0, 4).map(e => validCanon(e.smiles)).filter(Boolean);
    pop = [...new Set([...parents, ...children, ...fresh])].slice(0, 34);
    $('status').textContent = `generation ${g + 1}/${GENS} — ${Object.values(scored).filter(Boolean).length} unique candidates scored…`;
    await new Promise(r => setTimeout(r, 0));
  }
  for (const c of pop) await scoreSmiles(c);
  showResults();
  $('run').disabled = false;
}

function showResults() {
  const obj = $('obj').value;
  const all = Object.entries(scored).filter(([, v]) => v).map(([canon, v]) => ({ canon, ...v }));
  all.sort((a, b) => b.fitness - a.fitness);
  const top = all.slice(0, 15);
  $('status').textContent = `done — ${all.length} unique candidates scored; showing top 15 by objective.`;
  const box = $('results'); box.innerHTML = '';
  top.forEach((c, i) => {
    const r = c.result;
    const novel = !r.known;
    const el = document.createElement('div'); el.className = 'cand';
    el.innerHTML =
      `<div class="struct">${window.MolStack.svg2d(S, c.canon)}</div>` +
      `<div><div class="mono small">#${i + 1} · ${c.canon}</div>` +
      `<div style="margin:4px 0"><span class="big" style="font-size:24px">${r.mu.toFixed(2)}</span> ` +
      `<span class="pm">± ${r.sigma.toFixed(2)} log S</span></div>` +
      `<div class="small muted">SAscore (est) <b>${r.sascore.toFixed(1)}</b>/10 · ` +
      `${novel ? '<span style="color:var(--accent2)">novel</span>' : 'in library (true ' + r.known.true + ')'}` +
      `${r.ood ? ' · <span style="color:var(--warn)">uncertain/OOD</span>' : ''}</div>` +
      `<div class="btnrow" style="margin-top:6px"></div></div>`;
    const row = el.querySelector('.btnrow');
    const save = document.createElement('button'); save.className = 'btn ghost small'; save.textContent = '★ save';
    save.onclick = () => { window.LB.add({ domain: 'molecular', id: c.canon, label: r.known ? r.known.name : c.canon, smiles: c.canon, metric: 'log S', value: +r.mu.toFixed(2), sigma: +r.sigma.toFixed(2), aleatoric: +r.ale.toFixed(3), epistemic: +r.epi.toFixed(3), units: 'log(mol/L)' }); save.textContent = '✓ saved'; };
    const syn = document.createElement('button'); syn.className = 'btn ghost small'; syn.textContent = '🧪 synthesis';
    syn.onclick = () => openSynth({ domain: 'molecular', smiles: c.canon, label: r.known ? r.known.name : c.canon });
    row.append(save, syn);
    box.appendChild(el);
  });
}

/* ======================= SUPERCONDUCTOR MODE ======================= */
let SCM = null, SC_SEEDS = [], scScored = {};
async function ensureSC() {
  if (SCM) return SCM;
  SCM = await window.SCModel.load('../predict/models/');
  try { await window.StabScreen.load('../predict/models/'); } catch (e) { console.warn('stability off', e); }
  try {
    const cs = await (await fetch('../shared/community_seed.json')).json();
    SC_SEEDS = cs.entries.filter(e => e.domain === 'superconductor').map(e => e.label);
  } catch (e) { SC_SEEDS = ['Y1Ba2Cu3O7', 'Mg1B2', 'La1.85Sr0.15Cu1O4']; }
  return SCM;
}
const rndf = (lo, hi) => +(lo + Math.random() * (hi - lo)).toFixed(2);
function mutateFormula(comp) {
  const out = []; const tableEls = Object.keys(SCM.EP.elements);
  for (let k = 0; k < 5; k++) {
    const c = Object.assign({}, comp); const els = Object.keys(c); const op = rint(4);
    if (op === 0 && els.length) { const e = els[rint(els.length)]; c[e] = Math.max(0.05, +(c[e] * rndf(0.6, 1.5)).toFixed(2)); }
    else if (op === 1 && els.length > 1) { delete c[els[rint(els.length)]]; }
    else if (op === 2) { const e = tableEls[rint(tableEls.length)]; c[e] = (c[e] || 0) + rndf(0.1, 1.5); }
    else if (els.length) { const e = els[rint(els.length)]; const ne = tableEls[rint(tableEls.length)]; const a = c[e]; delete c[e]; c[ne] = (c[ne] || 0) + a; }
    out.push(window.SCModel.formulaStr(c));
  }
  return out;
}
function scFitness(r) { return $('obj').value === 'ucb' ? r.mu + 1.5 * r.sigma : r.mu; }
async function scoreFormula(f) {
  if (f in scScored) return scScored[f];
  const r = await window.SCModel.fromFormula(SCM.session, SCM.meta, SCM.EP, f);
  scScored[f] = r.valid ? { f, result: r, fitness: scFitness(r) } : null;
  return scScored[f];
}
async function runSC() {
  $('run').disabled = true; scScored = {};
  await ensureSC();
  let pop = [...new Set(SC_SEEDS)];
  const GENS = 7;
  for (let g = 0; g < GENS; g++) {
    for (const f of pop) await scoreFormula(f);
    pop.sort((a, b) => (scScored[b]?.fitness ?? -1e9) - (scScored[a]?.fitness ?? -1e9));
    const parents = pop.slice(0, 10);
    const kids = new Set();
    for (const p of parents) for (const m of mutateFormula(window.SCFeat.parseFormula(p))) kids.add(m);
    pop = [...new Set([...parents, ...kids, ...SC_SEEDS.slice(0, 3)])].slice(0, 30);
    $('status').textContent = `generation ${g + 1}/${GENS} — ${Object.values(scScored).filter(Boolean).length} formulas evaluated…`;
    await new Promise(r => setTimeout(r, 0));
  }
  for (const f of pop) await scoreFormula(f);
  showResultsSC();
  $('run').disabled = false;
}
async function showResultsSC() {
  const all = Object.values(scScored).filter(Boolean);
  all.sort((a, b) => b.fitness - a.fitness);
  let top = all.slice(0, 20);
  // VERIFICATION LAYER: predict formation energy + SMACT for each, then surface
  // thermodynamically favourable candidates first (stable high-Tc is the goal).
  $('status').textContent = 'running verification layer (formation energy + SMACT)…';
  for (const c of top) {
    try { c.stab = await window.StabScreen.assess(SCM.EP, c.f); } catch (e) { c.stab = null; }
  }
  top.sort((a, b) => {
    const fa = a.stab && a.stab.favourable ? 1 : 0, fb = b.stab && b.stab.favourable ? 1 : 0;
    if (fa !== fb) return fb - fa;               // favourable first
    return b.fitness - a.fitness;                // then by Tc
  });
  top = top.slice(0, 15);
  const nfav = top.filter(c => c.stab && c.stab.favourable).length;
  $('status').textContent = `done — top 15 shown; ${nfav} pass the formation-energy screen ` +
    `(favourable to form). Stable + high-Tc candidates are surfaced first.`;
  const box = $('results'); box.innerHTML = '';
  top.forEach((c, i) => {
    const r = c.result; const seed = SC_SEEDS.includes(c.f); const s = c.stab;
    let stabHtml = '';
    if (s && s.valid) {
      const sm = s.smact.ok === true ? '✓ SMACT' : s.smact.ok === null ? '' : '✗ SMACT (metal?)';
      stabHtml = `<div class="small muted">🔬 E<sub>form</sub> ${s.eform >= 0 ? '+' : ''}${s.eform.toFixed(2)} eV/atom ` +
        `${s.favourable ? '<span style="color:var(--accent2)">✓ favourable</span>' : '<span style="color:var(--warn)">✗ ≥0</span>'} ${sm}</div>`;
    }
    const el = document.createElement('div'); el.className = 'cand';
    el.innerHTML =
      `<div class="struct" style="background:#12203a;color:#dfe8ff;font-family:var(--mono);font-size:13px;padding:6px;text-align:center">${c.f}</div>` +
      `<div><div class="mono small">#${i + 1}</div>` +
      `<div style="margin:4px 0"><span class="big" style="font-size:24px">${r.mu.toFixed(1)}</span> <span class="pm">± ${r.sigma.toFixed(1)} K</span></div>` +
      `<div class="small muted">${seed ? 'known reference' : '<span style="color:var(--accent2)">novel</span>'}` +
      `${r.oob > 0 ? ` · <span style="color:var(--warn)">${r.oob}/81 OOD</span>` : ''}</div>` +
      stabHtml + `<div class="btnrow" style="margin-top:6px"></div></div>`;
    const row = el.querySelector('.btnrow');
    const save = document.createElement('button'); save.className = 'btn ghost small'; save.textContent = '★ save';
    save.onclick = () => { window.LB.add({ domain: 'superconductor', id: c.f, label: c.f, metric: 'Tc', value: +r.mu.toFixed(1), sigma: +r.sigma.toFixed(1), aleatoric: +r.ale.toFixed(2), epistemic: +r.epi.toFixed(2), units: 'K' }); save.textContent = '✓ saved'; };
    const syn = document.createElement('button'); syn.className = 'btn ghost small'; syn.textContent = '🧪 synthesis';
    syn.onclick = () => openSynth({ domain: 'superconductor', label: c.f, id: c.f });
    row.append(save, syn);
    box.appendChild(el);
  });
}

/* ======================= domain switch + boot ======================= */
let domain = 'molecular';
function setDomain(d) {
  domain = d;
  $('dom-mol').classList.toggle('primary', d === 'molecular');
  $('dom-sc').classList.toggle('primary', d === 'superconductor');
  $('synthpref-wrap').style.display = d === 'molecular' ? '' : 'none';
  const obj = $('obj');
  obj.innerHTML = d === 'molecular'
    ? '<option value="max">Maximize solubility (high log S)</option><option value="min">Minimize solubility (low log S)</option><option value="ucb">Explore (mean + uncertainty, UCB)</option>'
    : '<option value="maxtc">Maximize predicted Tc</option><option value="ucb">Explore (Tc + uncertainty, UCB)</option>';
  $('results').innerHTML = '';
  $('status').textContent = d === 'molecular' ? 'Pick an objective and run.' : 'Evolves real chemical formulas; each is featurized live (81 UCI descriptors) and scored. Novel formulas are hypotheses — Tc is "if it superconducts", and OOD counts warn when you leave the training space.';
}
$('dom-mol').onclick = () => setDomain('molecular');
$('dom-sc').onclick = () => setDomain('superconductor');
$('run').onclick = () => (domain === 'molecular' ? run() : runSC());

(async function () {
  setDomain('molecular');
  try {
    S = await window.MolStack.load('../predict/models/');
    $('status').textContent = 'Model ready. Pick an objective and run the search.';
  } catch (e) { $('status').innerHTML = '<span style="color:var(--danger)">load failed: ' + e + '</span>'; console.error(e); }
})();
