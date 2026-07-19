/* SMACT-style plausibility screen (Davies et al. 2019): is there a choice of
 * common oxidation states that makes the composition charge-neutral, with anions
 * more electronegative than cations? A real first-pass validity filter for
 * ionic/oxide-type compounds. NOTE: metals/intermetallics (e.g. MgB2) legitimately
 * fail this — charge neutrality is an ionic concept — so a "fail" is not a claim the
 * compound is invalid, only that it isn't a classic charge-balanced ionic solid. */
window.SMACT = (function () {
  let TAB = null;
  async function load(base) {
    if (!TAB) TAB = await (await fetch((base || 'models/') + 'oxidation_en.json')).json();
    return TAB;
  }
  function check(comp) {
    const els = Object.keys(comp).filter(e => comp[e] > 0 && TAB[e]);
    const missing = Object.keys(comp).filter(e => comp[e] > 0 && !TAB[e]);
    if (missing.length) return { ok: false, reason: 'unsupported: ' + missing.join(',') };
    if (els.length < 2) return { ok: false, reason: 'single element' };
    const amts = els.map(e => comp[e]);
    const oxs = els.map(e => TAB[e].ox);
    let combos = [[]];
    for (const list of oxs) {
      const next = [];
      for (const c of combos) for (const o of list) next.push(c.concat(o));
      combos = next;
      if (combos.length > 300000) return { ok: null, reason: 'too many states to check' };
    }
    for (const combo of combos) {
      let charge = 0;
      for (let i = 0; i < amts.length; i++) charge += amts[i] * combo[i];
      if (Math.abs(charge) < 1e-6) {
        const cat = [], an = [];
        for (let i = 0; i < combo.length; i++) {
          const en = TAB[els[i]].en; if (en == null) continue;
          if (combo[i] > 0) cat.push(en); else if (combo[i] < 0) an.push(en);
        }
        if (!an.length || !cat.length || Math.min(...an) > Math.max(...cat))
          return { ok: true };
      }
    }
    return { ok: false, reason: 'no charge-neutral assignment (typical for metals/intermetallics)' };
  }
  return { load, check };
})();
