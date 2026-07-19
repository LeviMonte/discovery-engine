/* Experience B — discovery replay. Animates the saved Phase-5 runs. */
'use strict';
const C = window.CAVEATS;
const $ = (id) => document.getElementById(id);
$('framing').textContent = C.replay;
$('metrics-note').textContent =
  'Enrichment, AUDC and acquisitions-to-K are final-budget values read directly ' +
  'from results/metrics.json; "hits so far" tracks the animation frame.';

const COLORS = { random:'#8aa0b6', greedy:'#4f9dff', ucb:'#ff5d6c',
  thompson:'#38d39f', ei:'#b48cff', max_var:'#ffa24b' };
const ORDER = ['random','greedy','ucb','thompson','ei','max_var'];

let DATA = null, domain = 'superconductor', selected = new Set(ORDER),
    frame = 0, maxFrame = 0, playing = false, timer = null, chart = null;

async function boot() {
  DATA = (await (await fetch('data/discovery_runs.json')).json()).domains;
  buildChecks();
  $('dom-sc').onclick = () => setDomain('superconductor');
  $('dom-mol').onclick = () => setDomain('molecular');
  $('play').onclick = togglePlay;
  $('reset').onclick = () => { setFrame(0); };
  $('scrub').oninput = () => { pause(); setFrame(+$('scrub').value); };
  setDomain('superconductor');
}

function buildChecks() {
  const box = $('strat-checks'); box.innerHTML = '';
  ORDER.forEach((s) => {
    const l = document.createElement('label'); l.className = 'chk';
    l.innerHTML = `<input type="checkbox" ${selected.has(s) ? 'checked' : ''}>` +
      `<span class="dot" style="background:${COLORS[s]}"></span>${s}`;
    l.querySelector('input').onchange = (e) => {
      if (e.target.checked) selected.add(s); else selected.delete(s);
      rebuild();
    };
    box.appendChild(l);
  });
}

function setDomain(d) {
  domain = d;
  $('dom-sc').classList.toggle('primary', d === 'superconductor');
  $('dom-mol').classList.toggle('primary', d === 'molecular');
  const dd = DATA[domain];
  $('metrics-domain').textContent = `${dd.dataset.split('(')[0].trim()} · pool ${dd.pool_size} · top-${dd.n_topk} · ${dd.n_seeds} seeds`;
  $('halfK').textContent = dd.acq_to_half_target;
  const anyS = dd.strategies.greedy;
  maxFrame = anyS.hits_mean.length - 1;
  $('scrub').max = maxFrame;
  $('chart-cap').textContent =
    `x-axis: number of materials measured. y-axis: how many of the top-${dd.n_topk} ` +
    `${dd.units === 'K' ? 'highest-Tc' : 'best'} materials have been found. Shaded = 95% CI over ${dd.n_seeds} seeds.`;
  rebuild(); setFrame(maxFrame);
}

function buildDatasets(upTo) {
  const dd = DATA[domain], ds = [];
  ORDER.forEach((s) => {
    if (!selected.has(s)) return;
    const st = dd.strategies[s];
    const x = st.n_labeled, mean = st.hits_mean, ci = st.hits_ci95;
    const pts = (arr, f) => arr.slice(0, upTo + 1).map((v, i) => ({ x: x[i], y: f(v, i) }));
    const lower = pts(mean, (v, i) => Math.max(0, v - ci[i]));
    const upper = pts(mean, (v, i) => v + ci[i]);
    const line = pts(mean, (v) => v);
    ds.push({ label: s + '_lo', data: lower, borderWidth: 0, pointRadius: 0,
      fill: false, tension: .2, borderColor: 'transparent' });
    ds.push({ label: s + '_hi', data: upper, borderWidth: 0, pointRadius: 0,
      fill: '-1', backgroundColor: hexA(COLORS[s], .13), tension: .2,
      borderColor: 'transparent' });
    ds.push({ label: s, data: line, borderColor: COLORS[s], borderWidth: 2.4,
      pointRadius: 0, fill: false, tension: .2 });
  });
  return ds;
}

function rebuild() {
  const dd = DATA[domain];
  const cfg = {
    type: 'line',
    data: { datasets: buildDatasets(frame) },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: 'nearest', intersect: false },
      scales: {
        x: { type: 'linear', title: { display: true, text: '# materials measured' },
          grid: { color: '#243050' }, ticks: { color: '#9aa6c2' } },
        y: { beginAtZero: true, suggestedMax: dd.n_topk,
          title: { display: true, text: `top-${dd.n_topk} found` },
          grid: { color: '#243050' }, ticks: { color: '#9aa6c2' } },
      },
      plugins: {
        legend: { labels: { filter: (i) => !i.text.endsWith('_lo') && !i.text.endsWith('_hi'),
          color: '#e8ecf6' } },
        tooltip: { filter: (i) => !i.dataset.label.endsWith('_lo') && !i.dataset.label.endsWith('_hi') },
      },
    },
  };
  if (chart) chart.destroy();
  const ctx = $('chart').getContext('2d');
  chart = new Chart(ctx, cfg);
  buildMetricsTable();
  updateFrameLabel();
}

function setFrame(f) {
  frame = Math.max(0, Math.min(maxFrame, f));
  $('scrub').value = frame;
  chart.data.datasets = buildDatasets(frame);
  chart.update('none');
  updateHits(); updateFrameLabel();
}

function updateFrameLabel() {
  const dd = DATA[domain];
  const nl = dd.strategies.greedy.n_labeled[frame];
  $('frame-label').textContent = `${Math.round(nl)} measured (${frame}/${maxFrame})`;
}

function buildMetricsTable() {
  const dd = DATA[domain], tb = $('metrics').querySelector('tbody');
  tb.innerHTML = '';
  ORDER.forEach((s) => {
    const st = dd.strategies[s];
    const tr = document.createElement('tr'); tr.id = 'row-' + s;
    tr.className = selected.has(s) ? '' : 'off';
    const a2k = st.acq_to_half_hits != null ? Math.round(st.acq_to_half_hits) : '—';
    tr.innerHTML =
      `<td><span class="dot" style="background:${COLORS[s]}"></span>${s}</td>` +
      `<td class="mono" id="hits-${s}">—</td>` +
      `<td class="mono">${st.enrichment.toFixed(2)}</td>` +
      `<td class="mono">${st.audc.toFixed(3)}</td>` +
      `<td class="mono">${a2k}</td>`;
    tb.appendChild(tr);
  });
  updateHits();
}

function updateHits() {
  const dd = DATA[domain];
  ORDER.forEach((s) => {
    const cell = $('hits-' + s); if (!cell) return;
    cell.textContent = Math.round(dd.strategies[s].hits_mean[frame]);
    $('row-' + s).className = selected.has(s) ? '' : 'off';
  });
}

function togglePlay() { playing ? pause() : play(); }
function play() {
  if (frame >= maxFrame) frame = 0;
  playing = true; $('play').textContent = '❚❚ Pause';
  timer = setInterval(() => {
    if (frame >= maxFrame) { pause(); return; }
    setFrame(frame + 1);
  }, 550);
}
function pause() { playing = false; $('play').textContent = '▶ Play';
  if (timer) clearInterval(timer); timer = null; }

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;
}

boot();
