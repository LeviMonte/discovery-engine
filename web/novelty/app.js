/* Confidence & Novelty tab — real calibration diagnostics from discovery_runs.json. */
'use strict';
const $ = (id) => document.getElementById(id);
let DATA = null, domain = 'superconductor', chart = null;
$('honesty').innerHTML = '<b>Honesty.</b> ' + window.CAVEATS.uncertainty;
$('ood-note').textContent = window.CAVEATS.molecular.sigma_ranks;

function setDomain(d) {
  domain = d;
  $('dom-sc').classList.toggle('primary', d === 'superconductor');
  $('dom-mol').classList.toggle('primary', d === 'molecular');
  render();
}

function render() {
  const c = DATA.domains[domain].calibration;
  const units = DATA.domains[domain].units;
  const ps = c.reliability.nominal;
  const single = c.reliability.empirical_single, ens = c.reliability.empirical_ensemble;
  const cfg = {
    type: 'line',
    data: {
      labels: ps.map(x => x.toFixed(2)),
      datasets: [
        { label: 'perfect', data: ps, borderColor: '#8aa0b6', borderDash: [6, 5], pointRadius: 0, borderWidth: 1.5 },
        { label: `single (ECE ${c.single.ece.toFixed(3)})`, data: single, borderColor: '#ffa24b', borderWidth: 2.4, pointRadius: 2 },
        { label: `ensemble (ECE ${c.ensemble.ece.toFixed(3)})`, data: ens, borderColor: '#4f9dff', borderWidth: 2.4, pointRadius: 2 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'nominal coverage' }, grid: { color: '#243050' }, ticks: { color: '#9aa6c2', maxTicksLimit: 6 } },
        y: { title: { display: true, text: 'empirical coverage' }, grid: { color: '#243050' }, ticks: { color: '#9aa6c2' }, min: 0, max: 1 },
      },
      plugins: { legend: { labels: { color: '#e8ecf6' } } },
    },
  };
  if (chart) chart.destroy();
  chart = new Chart($('rel').getContext('2d'), cfg);
  $('rel-cap').textContent =
    `Held-out reliability over ${c.n_seeds} seeds (n_test ${c.n_test}). On the diagonal = ` +
    `perfectly calibrated. Below = overconfident, above = under-confident.`;

  $('k-es').textContent = c.single.ece.toFixed(3);
  $('k-ee').textContent = `${c.ensemble.ece.toFixed(3)} → ${c.ensemble_recal_ece.toFixed(3)}`;
  $('k-sp').textContent = c.ensemble.spearman.toFixed(2) +
    (Math.abs(c.ensemble.spearman) > 0.3 ? '  (σ ranks error)' : '  (σ does NOT rank error)');
  $('k-ae').textContent = `${c.mean_aleatoric.toExponential(2)}  vs  ${c.mean_epistemic.toExponential(2)}`;

  const better = c.ensemble.ece <= c.single.ece;
  $('finding').innerHTML =
    `<b>Honest finding (${domain}).</b> The deep ensemble’s raw calibration is ` +
    `${better ? 'better than' : 'actually slightly worse than'} the fair single net ` +
    `(${c.single.ece.toFixed(3)} vs ${c.ensemble.ece.toFixed(3)} ECE) — ` +
    `${better ? '' : 'it over-disperses; '}post-hoc recalibration brings it to ` +
    `${c.ensemble_recal_ece.toFixed(3)}. ` +
    (Math.abs(c.ensemble.spearman) > 0.3
      ? 'Here the uncertainty also usefully ranks which predictions are wrong.'
      : 'Here the uncertainty does NOT rank which predictions are wrong — a real limitation, shown not hidden.');
}

$('dom-sc').onclick = () => setDomain('superconductor');
$('dom-mol').onclick = () => setDomain('molecular');

(async function () {
  DATA = await (await fetch('../replay/data/discovery_runs.json')).json();
  setDomain('superconductor');
})();
