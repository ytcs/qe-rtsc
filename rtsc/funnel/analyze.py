"""F0 analysis: access gate (T1a MLIP and T2 DFT), MLIP hull (T1a), T1b, T2 metallicity/hull.

    python analyze.py access <t1 ms.jsonl> <t1 mace.jsonl> <t1b ms.jsonl> <t2.jsonl> <out.json>
    python analyze.py hull   <all.json> <t1 ms.jsonl> <t1 mace.jsonl> <out.json>

Conventions: FUNNEL-LOG F0 pre-registration (hull channel; PASS <= 0.55 in both functionals, REJECT
> 0.8 in both, else HOLD; T1a line 1.1 eV/H2 in either model; ZPE from MatterSim phonons + H2 0.270 eV).
"""
import json
import sys
from pathlib import Path

from pymatgen.core import Composition

ZPE_H2 = 0.270
RX = {
    "MgH2": [({"Mg": 1}, 1)],
    "Mg2FeH6": [({"MgH2": 2, "Fe": 1}, 1), ({"Mg": 2, "Fe": 1}, 3)],
    "Mg2IrH6": [({"MgH2": 2, "Ir": 1}, 1), ({"Mg2IrH5": 1}, 0.5), ({"Mg": 2, "Ir": 1}, 3)],
    "Li2AgH6": [({"LiH": 2, "Ag": 1}, 2), ({"LiAg": 1, "LiH": 1}, 2.5), ({"LiAg": 1, "Li": 1}, 3)],
    "Li2AuH6": [({"LiH": 2, "Au": 1}, 2), ({"LiAu": 1, "LiH": 1}, 2.5), ({"LiAu": 0.5, "Li3Au": 0.5}, 3)],
}
FORMULA_ATOMS = {k: Composition(k).num_atoms for k in
                 ["MgH2", "Mg", "Mg2FeH6", "Fe", "Mg2IrH6", "Ir", "Mg2IrH5", "Li2AgH6", "LiH", "Ag", "LiAg", "Li",
                  "Li2AuH6", "Au", "LiAu", "Li3Au", "H2"]}


def load(p, key="e_atom", by=("id",)):
    out = {}
    if not p or not Path(p).exists():
        return out
    for ln in Path(p).read_text().splitlines():
        r = json.loads(ln)
        if key in r:
            out[tuple(r[b] for b in by) if len(by) > 1 else r[by[0]]] = r
    return out


def delta(e_atom, zpe_atom, P):
    """[(channel, dE, dE_zpe)] per H2, and the hull channel (lowest total product energy)."""
    out = []
    for prod, n in RX[P]:
        try:
            EP = e_atom[P] * FORMULA_ATOMS[P]
            Epr = sum(c * e_atom[p] * FORMULA_ATOMS[p] for p, c in prod.items()) + n * e_atom["H2"] * 2
        except KeyError:
            continue
        d = (EP - Epr) / n
        dz = None
        if zpe_atom is not None and all(x in zpe_atom for x in [P, *prod]):
            ZP = zpe_atom[P] * FORMULA_ATOMS[P]
            Zpr = sum(c * zpe_atom[p] * FORMULA_ATOMS[p] for p, c in prod.items()) + n * ZPE_H2
            dz = d + (ZP - Zpr) / n
        out.append(dict(channel=" + ".join(f"{c:g} {p}" for p, c in prod.items()) + f" + {n:g} H2",
                        n=n, dE=d, dE_zpe=dz, total=d * n))
    hull = max(out, key=lambda r: r["total"]) if out else None
    return out, hull


def gate(vals, lo=0.55, hi=0.8):
    if any(v is None for v in vals):
        return "n/a"
    if all(v <= lo for v in vals):
        return "PASS"
    if all(v > hi for v in vals):
        return "REJECT"
    return "HOLD"


def cmd_access(ms, mace, t1b, t2, out):
    e = {"mattersim": {k: v["e_atom"] for k, v in load(ms).items()},
         "mace": {k: v["e_atom"] for k, v in load(mace).items()}}
    zpe = {}
    for k, v in load(t1b, key="zpe_eV").items():
        zpe[k] = v["zpe_eV"] / v["nat"]
    t2r = load(t2, by=("id", "xc"))
    for xc in ("pbe", "pbesol"):
        e[f"dft-{xc}"] = {k: r["e_atom"] for (k, x), r in t2r.items() if x == xc}
    # elements and non-hydride refs have one functional only at T2 when no H: fill PBEsol from... none needed:
    # every phase in RX contains H or is an element/intermetallic; T2 runs PBEsol only for hydrides, so
    # non-H phases are run in both by listing them explicitly (see run script).
    res = {}
    for P in RX:
        row = {}
        for src, ea in e.items():
            ch, hull = delta(ea, zpe, P)
            row[src] = dict(channels=ch, hull=hull)
        def v(src, z=True):
            h = row.get(src, {}).get("hull")
            return None if not h else (h["dE_zpe"] if z else h["dE"])
        row["T1a_line_1.1"] = ("pass" if any(x is not None and x <= 1.1 for x in (v("mattersim"), v("mace")))
                               else "reject")
        row["T1a_gate"] = gate([v("mattersim"), v("mace")])
        row["T2_gate"] = gate([v("dft-pbe"), v("dft-pbesol")])
        row["T2_gate_noZPE"] = gate([v("dft-pbe", False), v("dft-pbesol", False)])
        res[P] = row
        print(f"{P:8s} " + "  ".join(f"{s}: {v(s, False) if v(s, False) is None else round(v(s, False), 3)}/"
                                     f"{v(s) if v(s) is None else round(v(s), 3)}" for s in e)
              + f"  | T1a-line {row['T1a_line_1.1']}  T1a-gate {row['T1a_gate']}  T2 {row['T2_gate']}")
    Path(out).write_text(json.dumps(res, indent=1))


def cmd_hull(allj, ms, mace, out):
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    data = json.loads(Path(allj).read_text())
    res = {}
    for model, p in (("mattersim", ms), ("mace", mace)):
        rec = load(p)
        entries = {}
        for k, r in rec.items():
            if k == "H2" or data.get(k, {}).get("role") not in ("db", "ref"):
                continue
            n = r["nat"]
            from pymatgen.core import Structure
            comp = Structure.from_dict(r["structure"]).composition
            entries[k] = PDEntry(comp, r["e_atom"] * n, name=k)
        for k, v in data.items():
            if v.get("role") != "db" or k not in entries:
                continue
            els = {x.symbol for x in entries[k].composition.elements}
            pool = [en for kk, en in entries.items() if {x.symbol for x in en.composition.elements} <= els]
            try:
                pd = PhaseDiagram(pool)
                eh = pd.get_e_above_hull(entries[k])
                others = PhaseDiagram([en for en in pool if en.name != k])
                dec = others.get_decomposition(entries[k].composition)
                neigh = {en.name: float(a) for en, a in dec.items()}
                ed = (entries[k].energy_per_atom - sum(a * en.energy_per_atom for en, a in dec.items()))
            except Exception as ex:
                eh, neigh, ed = None, {}, None
            res.setdefault(k, dict(label=v["label"], Tc=v["Tc"], stability=v["stability"]))
            res[k][model] = dict(ehull=eh, decomp_energy=ed, neighbours=neigh)
    for k, r in res.items():
        hs = [r.get(m, {}).get("ehull") for m in ("mattersim", "mace")]
        r["pass_T1a"] = any(h is not None and h <= 0.10 for h in hs)
    Path(out).write_text(json.dumps(res, indent=1))
    lab = {}
    for r in res.values():
        lab.setdefault(r["label"], [0, 0])
        lab[r["label"]][0] += 1
        lab[r["label"]][1] += r["pass_T1a"]
    print("T1a pass by label (n, pass):", lab)


if __name__ == "__main__":
    {"access": cmd_access, "hull": cmd_hull}[sys.argv[1]](*sys.argv[2:])
