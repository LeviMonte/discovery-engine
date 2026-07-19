/* Retrosynthesis client — pluggable backend, no fabricated routes. */
'use strict';
const $ = (id) => document.getElementById(id);
const KEY = 'discovery_engine_retro_endpoint';
$('honesty').innerHTML =
  '<b>Honesty.</b> Predicted routes are <i>suggestions</i> from a model trained on ' +
  'past reactions — real success rates are well below 100%, and feasibility, yield, ' +
  'and safety must be confirmed against primary literature by a chemist. This tool ' +
  'never claims a route is guaranteed, and it does not work for inorganic ' +
  'superconductors (retrosynthesis is an organic-chemistry method).';

$('endpoint').value = localStorage.getItem(KEY) || '';
const q = new URLSearchParams(location.search).get('smiles');
if (q) $('smiles').value = q;

$('go').onclick = async () => {
  const smiles = $('smiles').value.trim();
  const url = $('endpoint').value.trim();
  localStorage.setItem(KEY, url);
  const gate = $('gate'), routes = $('routes');
  routes.innerHTML = ''; gate.classList.remove('show');
  if (!smiles) { gate.textContent = 'Enter a target SMILES.'; gate.classList.add('show'); return; }
  if (!url) {
    gate.innerHTML = 'ℹ️ <b>No backend connected.</b> Retrosynthesis needs a Python ' +
      'service (see below) — this static site won’t fabricate a route. Connect a ' +
      'backend URL above, or use <a href="https://rxn.res.ibm.com" target="_blank" rel="noopener">IBM RXN</a> / ' +
      '<a href="https://askcos.mit.edu" target="_blank" rel="noopener">ASKCOS</a> directly.';
    gate.classList.add('show'); return;
  }
  gate.innerHTML = '<span class="spinner"></span> querying your backend…'; gate.classList.add('show');
  try {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ smiles }) });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    gate.classList.remove('show');
    const list = data.routes || [];
    if (!list.length) { routes.innerHTML = '<p class="muted small">Backend returned no routes.</p>'; return; }
    routes.innerHTML = '<h2>Suggested routes <span class="tag pre">from your backend</span></h2>' +
      list.slice(0, 5).map((rt, i) =>
        `<div class="card" style="background:#0d1322;margin:8px 0">
          <b>Route ${i + 1}</b> ${rt.score != null ? `<span class="muted small">score ${(+rt.score).toFixed(2)}</span>` : ''}
          <ol class="synth-steps">${(rt.steps || []).map(s => `<li class="mono small">${typeof s === 'string' ? s : JSON.stringify(s)}</li>`).join('')}</ol>
        </div>`).join('') +
      '<p class="note small">These come from your connected model and are unverified suggestions.</p>';
  } catch (e) {
    gate.innerHTML = '❌ Backend call failed: ' + e.message + '. Check the URL/CORS on your service.';
    gate.classList.add('show');
  }
};
