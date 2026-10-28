"""F0 database subset (JARVIS-BCS supercon_3d) and its hull reference phases (JARVIS dft_3d).

    python db.py <store dir> <subset.json> <refs.json>

Subset (FUNNEL-LOG pre-registration): press = 0 GPa, <= 12 atoms, no f-electron elements, random 200
(seed 0). Positive: Tc >= 5 K; negative: Tc < 0.5 K.
References: for every chemical (sub)system of the subset, the JARVIS dft_3d entry with the lowest
ehull per reduced composition, kept if ehull <= 0.05 eV/atom and <= 24 atoms; the elements always.
The subset entries themselves are also hull members (relaxed in the T1 subset run).
"""
import itertools
import json
import random
import sys
from pathlib import Path

from jarvis.core.atoms import Atoms as JAtoms
from jarvis.db.figshare import data
from pymatgen.core import Composition

F_ELEMENTS = set("Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())


def to_pmg(a):
    return JAtoms.from_dict(a).pymatgen_converter()


def main(store, out_subset, out_refs):
    sc = data("supercon_3d", store_dir=store)
    ok = []
    for e in sc:
        s = to_pmg(e["atoms"])
        els = {x.symbol for x in s.composition.elements}
        if str(e.get("press", "")).strip() not in ("0 GPa", "0", "0.0 GPa") or len(s) > 12 or els & F_ELEMENTS:
            continue
        ok.append((e, s))
    random.Random(0).shuffle(ok)
    sub = ok[:200]
    subset = {}
    for e, s in sub:
        tc = float(e["Tc"])
        label = "pos" if tc >= 5 else ("neg" if tc < 0.5 else "mid")
        subset[e["jid"]] = dict(structure=s.as_dict(), role="db", Tc=tc, lamb=e["lamb"], wlog=e["wlog"],
                                stability=e["stability"], label=label)
    n = {k: sum(1 for v in subset.values() if v["label"] == k) for k in ("pos", "neg", "mid")}
    print(f"eligible {len(ok)}; subset {len(subset)}; {n}")
    Path(out_subset).write_text(json.dumps(subset))

    systems = set()
    for v in subset.values():
        els = sorted({sp["element"] for site in v["structure"]["sites"] for sp in site["species"]})
        for r in range(1, len(els) + 1):
            systems.update(tuple(c) for c in itertools.combinations(els, r))
    d3 = data("dft_3d", store_dir=store)
    best = {}
    for e in d3:
        try:
            comp = Composition(e["formula"]) if e.get("formula") else None
        except Exception:
            continue
        if comp is None:
            continue
        key = tuple(sorted(x.symbol for x in comp.elements))
        if key not in systems:
            continue
        eh = e.get("ehull")
        if not isinstance(eh, (int, float)):
            continue
        rf = comp.reduced_formula
        if rf not in best or eh < best[rf][0]:
            best[rf] = (eh, e)
    refs = {}
    for rf, (eh, e) in best.items():
        s = to_pmg(e["atoms"])
        if len(Composition(rf).elements) == 1 or (eh <= 0.05 and len(s) <= 24):
            refs[e["jid"]] = dict(structure=s.as_dict(), role="ref", formula=rf, ehull_jarvis=eh)
    missing = {el for sy in systems if len(sy) == 1 for el in sy} - {
        Composition(v["formula"]).elements[0].symbol for v in refs.values() if len(Composition(v["formula"]).elements) == 1}
    print(f"systems {len(systems)}; refs {len(refs)}; missing elements {sorted(missing)}")
    for k in list(refs):
        if k in subset:
            del refs[k]
    Path(out_refs).write_text(json.dumps(refs))


if __name__ == "__main__":
    main(*sys.argv[1:4])
