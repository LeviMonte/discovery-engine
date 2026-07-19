/* Leaderboard tab: real benchmark references + your auto-saved local finds. */
'use strict';
const $ = (id) => document.getElementById(id);
let SEED = { entries: [] }, domain = 'superconductor';

$('framing').innerHTML =
  'Your finds auto-save to <b>this browser only</b> (no server, so nothing is shared ' +
  'across people — we won’t pretend otherwise). The pre-filled entries are <b>real ' +
  'benchmark references</b> (measured values from the public datasets), there so your ' +
  'predictions have an honest reference to beat. “Submit” opens a GitHub issue a ' +
  'maintainer can review.';

function openSynth(entry) {
  const s = window.SYNTH.get(entry);
  $('synth-title').textContent = s.title;
  $('synth-body').innerHTML =
    '<ol class="synth-steps">' + s.steps.map(x => `<li>${x}</li>`).join('') + '</ol>' +
    (s.danger ? `<div class="synth-danger">⚠️ ${s.danger}</div>` : '') +
    `<div class="note small"><b>Honesty:</b> ${s.caveat}</div>` +
    `<div class="synth-refs">References: ${s.refs.join('; ')}<br>` +
    `<a href="${s.search}" target="_blank" rel="noopener">↗ literature search</a></div>`;
  $('synth-modal').classList.add('show');
}
$('synth-close').onclick = () => $('synth-modal').classList.remove('show');
$('synth-modal').addEventListener('click', e => { if (e.target.id === 'synth-modal') $('synth-modal').classList.remove('show'); });

function setDomain(d) {
  domain = d;
  $('dom-sc').classList.toggle('primary', d === 'superconductor');
  $('dom-mol').classList.toggle('primary', d === 'molecular');
  $('metric-h').textContent = d === 'superconductor' ? 'Tc (K)' : 'log S (mol/L)';
  render();
}

function render() {
  const rows = window.LB.ranked(domain, SEED.entries);
  const tb = $('board').querySelector('tbody'); tb.innerHTML = '';
  if (!rows.length) { tb.innerHTML = '<tr><td colspan="6" class="muted">No entries yet — go predict something.</td></tr>'; }
  rows.forEach((r, i) => {
    const tr = document.createElement('tr');
    const badge = r.kind === 'your find'
      ? '<span class="tag live">your find</span>'
      : '<span class="tag">benchmark ref</span>';
    tr.innerHTML =
      `<td class="mono">${i + 1}</td>` +
      `<td>${escapeHtml(r.label)}${r.smiles ? `<br><span class="small muted mono">${escapeHtml(r.smiles)}</span>` : ''}</td>` +
      `<td class="mono">${r.value}</td>` +
      `<td class="mono">${r.sigma != null ? '±' + r.sigma : '—'}</td>` +
      `<td>${badge}${r.kind !== 'your find' ? `<br><span class="small muted">${r.source || 'measured'}</span>` : ''}</td>` +
      `<td><button class="btn ghost small" data-i="${i}">🧪 + submit</button></td>`;
    const btn = tr.querySelector('button');
    btn.onclick = () => {
      openSynth(r);
      if (r.kind === 'your find') {
        const a = document.createElement('a'); a.href = window.LB.submitUrl(r); a.target = '_blank';
        a.rel = 'noopener'; a.className = 'btn primary'; a.textContent = 'Submit this find to the community board →';
        a.style.display = 'inline-block'; a.style.marginTop = '12px';
        $('synth-body').appendChild(a);
      }
    };
    tb.appendChild(tr);
  });
  const mine = window.LB.all().filter(r => r.domain === domain).length;
  $('note').textContent = `${rows.length} shown · ${mine} of them are your local finds · ${SEED.entries.filter(e => e.domain === domain).length} benchmark references. Click a row for synthesis context (and, for your own finds, a submit link).`;
}

function escapeHtml(s) { return String(s).replace(/[&<>"]/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;' }[c])); }

$('dom-sc').onclick = () => setDomain('superconductor');
$('dom-mol').onclick = () => setDomain('molecular');
$('clear').onclick = () => { if (confirm('Clear your locally-saved finds?')) { window.LB.clear(); render(); } };

(async function () {
  try { SEED = await (await fetch('../shared/community_seed.json')).json(); } catch (e) { console.error(e); }
  setDomain('superconductor');
})();
