/* Honest synthesis CONTEXT — not fabricated recipes.
 *
 * For well-known real materials we describe their genuine, literature-established
 * synthesis *class* with citations (this is textbook science, not invention). For
 * anything the model merely predicted, we say plainly that no validated route
 * exists and point to a literature search. Every entry is capped with a caveat and
 * a safety note. This provider never claims a specific procedure will yield a
 * specific (let alone superconducting) product. */
window.SYNTH = (function () {
  const LIT = (q) => 'https://scholar.google.com/scholar?q=' + encodeURIComponent(q);
  const PUBCHEM = (smi) =>
    'https://pubchem.ncbi.nlm.nih.gov/#query=' + encodeURIComponent(smi);

  // classify a superconductor by formula string
  function scClass(formula) {
    const f = formula || '';
    const has = (...els) => els.every(e => new RegExp(e + '(?![a-z])').test(f));
    if (has('Cu') && (has('Y', 'Ba') || has('Bi', 'Sr') || has('Tl') || has('Hg')))
      return 'cuprate';
    if (has('Mg', 'B')) return 'mgb2';
    if (has('Fe') && (has('As') || has('Se'))) return 'ironpnictide';
    if (has('H') && (has('S') || has('La')) && f.length < 8) return 'hydride';
    return 'generic-oxide';
  }

  const SC = {
    cuprate: {
      title: 'High-Tc cuprate — solid-state ceramic route (established)',
      steps: [
        'Weigh stoichiometric oxides/carbonates (e.g. Y2O3, BaCO3, CuO) to the target cation ratio.',
        'Grind/mix thoroughly; calcine in air ~900–950 °C to react and drive off CO2.',
        'Regrind, pelletize, and sinter ~920–950 °C for extended time.',
        'Anneal in flowing O2 ~400–550 °C to set the oxygen stoichiometry (critical for Tc).',
      ],
      refs: ['Wu et al., Phys. Rev. Lett. 58, 908 (1987) — YBa2Cu3O7-δ',
             'Cava, J. Am. Ceram. Soc. 83, 5 (2000) — oxide superconductor synthesis review'],
    },
    mgb2: {
      title: 'MgB2 — direct elemental reaction (established)',
      steps: [
        'Mix Mg and B powders in 1:2 molar ratio.',
        'Seal in an inert (Ta/steel) tube or under Ar to prevent Mg oxidation.',
        'React ~650–950 °C; Mg vapor reacts with the B framework.',
      ],
      refs: ['Nagamatsu et al., Nature 410, 63 (2001) — discovery of superconductivity in MgB2'],
    },
    ironpnictide: {
      title: 'Iron-based superconductor — solid-state / flux growth (established)',
      steps: [
        'Combine elements/precursors (e.g. Fe, As, and dopant sources) in an evacuated sealed ampoule.',
        'React at high temperature; for single crystals use a flux (e.g. FeAs, Sn).',
        'Slow-cool; extract crystals; anneal as needed for the target doping.',
      ],
      refs: ['Kamihara et al., J. Am. Chem. Soc. 130, 3296 (2008) — LaFeAsO1-xFx'],
    },
    hydride: {
      title: 'Superhydride — EXTREME high-pressure synthesis (specialist only)',
      steps: [
        'Load precursor (e.g. H2S or La + H2 source) into a diamond anvil cell.',
        'Compress to ~150–200 GPa; the superconducting phase forms only under pressure.',
        'Cool while maintaining pressure to observe superconductivity.',
      ],
      refs: ['Drozdov et al., Nature 525, 73 (2015) — 203 K in the H–S system under pressure'],
      danger: 'Requires a diamond-anvil-cell lab and pressures of ~1.5 million atm. Not reproducible outside a specialist high-pressure facility.',
    },
    'generic-oxide': {
      title: 'Inorganic composition — general solid-state route (context only)',
      steps: [
        'Weigh precursor oxides/carbonates to the target composition.',
        'Mix, calcine, regrind, and sinter at high temperature in a controlled atmosphere.',
        'Anneal to set stoichiometry; characterize by XRD and transport measurements.',
      ],
      refs: ['West, "Solid State Chemistry and its Applications" (2014) — ceramic synthesis'],
    },
  };

  function forSuperconductor(formula) {
    const cls = SC[scClass(formula)];
    return {
      title: cls.title,
      steps: cls.steps,
      refs: cls.refs,
      danger: cls.danger || null,
      search: LIT((formula || 'superconductor') + ' synthesis'),
      caveat:
        'This is the GENERAL established route for this material class, provided as ' +
        'educational context — NOT a validated recipe for this exact stoichiometry, ' +
        'and the model cannot tell you whether this composition superconducts at all. ' +
        'Real synthesis, phase purity, and Tc must be confirmed experimentally against ' +
        'primary literature.',
    };
  }

  function forMolecule(smiles, name) {
    return {
      title: 'Organic molecule — synthesis is compound-specific',
      steps: [
        'There is no single generic route: organic synthesis depends entirely on the ' +
        'target’s functional groups and available building blocks.',
        'Check whether the compound is commercially available or has a reported ' +
        'preparation before attempting a de-novo synthesis.',
      ],
      refs: ['Consult primary literature (Reaxys / SciFinder) for reported routes.'],
      danger: null,
      search: PUBCHEM(smiles || name || ''),
      caveat:
        'The model predicts a property (solubility), not how to make the molecule. No ' +
        'synthesis route is claimed. Use the literature link to find validated ' +
        'preparations; treat any planning as expert-supervised only.',
    };
  }

  function get(entry) {
    if (entry.domain === 'superconductor') return forSuperconductor(entry.label || entry.id);
    return forMolecule(entry.smiles || entry.id, entry.label);
  }
  return { get };
})();
