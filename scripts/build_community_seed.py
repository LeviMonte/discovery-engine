#!/usr/bin/env python3
"""Phase 9 -- seed the community leaderboard with REAL benchmark reference entries.

These are not fabricated 'discoveries': they are the genuine highest-Tc known
superconductors and extreme-solubility molecules from the public datasets, shown
so the leaderboard isn't empty and so user finds have an honest reference to beat.
Every entry is labelled 'benchmark reference' with its true measured value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "web" / "shared" / "community_seed.json"


def main():
    entries = []

    # --- superconductors: highest true Tc, de-duplicated by formula ---
    um = pd.read_csv(ROOT / "data" / "superconductor" / "unique_m.csv")
    um = um.sort_values("critical_temp", ascending=False)
    seen = set()
    for _, r in um.iterrows():
        f = str(r["material"])
        if f in seen:
            continue
        seen.add(f)
        entries.append({
            "domain": "superconductor", "id": f, "label": f,
            "metric": "Tc", "value": round(float(r["critical_temp"]), 1), "units": "K",
            "kind": "benchmark reference (measured)", "source": "UCI unique_m.csv",
        })
        if sum(e["domain"] == "superconductor" for e in entries) >= 15:
            break

    # --- molecules: most / least soluble real ESOL molecules ---
    grid = json.loads((ROOT / "web" / "predict" / "models" / "grid_molecular.json").read_text())
    lib = sorted(grid["library"], key=lambda e: e["true"], reverse=True)
    for e in lib[:6] + lib[-6:]:
        entries.append({
            "domain": "molecular", "id": e["smiles"], "label": e["name"],
            "smiles": e["smiles"], "metric": "log S", "value": e["true"],
            "units": "log(mol/L)", "kind": "benchmark reference (measured)",
            "source": "MoleculeNet ESOL",
        })

    OUT.write_text(json.dumps({"generated": "from public benchmark data",
                               "entries": entries}, indent=2))
    print(f"wrote {OUT} with {len(entries)} benchmark reference entries "
          f"({sum(e['domain']=='superconductor' for e in entries)} SC, "
          f"{sum(e['domain']=='molecular' for e in entries)} mol)")


if __name__ == "__main__":
    main()
