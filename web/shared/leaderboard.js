/* Leaderboard store. Two layers:
 *  - LOCAL: your own finds, auto-saved to this browser (localStorage). Not shared
 *    across people — GitHub Pages is static, there is no server to sync to, and we
 *    will not pretend otherwise.
 *  - COMMUNITY: real benchmark reference entries seeded from the public datasets
 *    (community_seed.json), plus a "submit your find" that opens a GitHub issue so
 *    a human can merge genuine submissions into the repo.
 */
window.LB = (function () {
  const KEY = 'discovery_engine_finds_v1';
  const REPO = 'LeviMonte/discovery-engine';

  function all() {
    try { return JSON.parse(localStorage.getItem(KEY) || '[]'); }
    catch (e) { return []; }
  }
  function save(list) { localStorage.setItem(KEY, JSON.stringify(list)); }

  function add(rec) {
    rec.ts = Date.now();
    rec.kind = 'your find';
    const list = all();
    // de-dup by domain+id, keep the most extreme value
    const i = list.findIndex(r => r.domain === rec.domain && r.id === rec.id);
    if (i >= 0) { list[i] = rec; } else { list.push(rec); }
    save(list);
    return rec;
  }
  function clear() { localStorage.removeItem(KEY); }

  // rank: superconductors by Tc desc; molecules by predicted log S desc
  function ranked(domain, extra) {
    const mine = all().filter(r => r.domain === domain);
    const seed = (extra || []).filter(r => r.domain === domain);
    const merged = seed.concat(mine);
    merged.sort((a, b) => (b.value ?? -1e9) - (a.value ?? -1e9));
    return merged;
  }

  function submitUrl(rec) {
    const title = encodeURIComponent(
      `[discovery] ${rec.domain}: ${rec.label} = ${rec.value} ${rec.units}`);
    const body = encodeURIComponent(
      `**Domain:** ${rec.domain}\n` +
      `**Candidate:** ${rec.label}\n` +
      (rec.smiles ? `**SMILES:** ${rec.smiles}\n` : '') +
      `**Predicted ${rec.metric}:** ${rec.value} ± ${rec.sigma ?? '?'} ${rec.units}\n` +
      `**Aleatoric / epistemic:** ${rec.aleatoric ?? '?'} / ${rec.epistemic ?? '?'}\n\n` +
      `> Auto-generated from the interactive demo. This is a model prediction on a ` +
      `retrospective benchmark, not a validated discovery. A maintainer will review ` +
      `before it is added to the community board.`);
    return `https://github.com/${REPO}/issues/new?title=${title}&body=${body}&labels=discovery`;
  }

  return { all, add, clear, ranked, submitUrl, REPO };
})();
