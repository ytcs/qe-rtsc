"""F0 analysis: access gate (T1a MLIP and T2 DFT), MLIP hull (T1a), T1b, T2 metallicity/hull.

    python analyze.py access <t1 ms.jsonl> <t1 mace.jsonl> <t1b ms.jsonl> <t2.jsonl> <out.json>
    python analyze.py hull   <all.json> <t1 ms.jsonl> <t1 mace.jsonl> <out.json>

F0b (FUNNEL.md section 7, pre-registered before any F0b run):
    python analyze.py hull1  <t1 ms.jsonl> <t1 mace.jsonl> <id> <ref ids,comma> <out.json>
    python analyze.py t1b2   <t1b ms.jsonl> <t1b mace.jsonl> <out.json>
    python analyze.py access2 <t1 ms> <t1 mace> <t1b ms> <t1b mace> <t2.jsonl> <dft-zpe.jsonl> <P> <out.json>
t1b2 is the T1b two-model rule: reject only if BOTH MatterSim and MACE-MPA-0 end below -30 cm^-1 after soft-mode
following (each model on its own relaxed geometry); one unstable model -> pass with flag t1b_split.
access2 is the ZPE-source rule: a phase is 'stable' if both MLIPs give omega_min >= -30 cm^-1 on that phase's
own (parent) cell; if every phase of every channel of P is stable, all solids use the MLIP Gamma-ZPE
(MatterSim, the F0 source model); otherwise each unstable phase uses the DFT Gamma-ZPE (zpe.py) and the stable
ones keep the MatterSim Gamma-ZPE. H2: 0.270 eV. The F0-convention values (MatterSim 8^3-mesh ZPE) are reported
alongside for comparison only.

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


def cmd_t2db(hull, t1b, t2, out):
    """T2 on database entries: metal (N(E_F) >= 0.05 /eV/atom), non-magnetic (|M| <= 0.1 muB/cell), DFT
    decomposition energy vs the MatterSim-hull neighbours <= 0.10 eV/atom."""
    h = json.loads(Path(hull).read_text())
    b = load(t1b, key="pass_T1b")
    d = {k: r for (k, x), r in load(t2, by=("id", "xc")).items() if x == "pbe"}
    res = {}
    for k, v in h.items():
        if k not in d:
            continue
        r = d[k]
        neigh = v["mattersim"]["neighbours"]
        ed = None
        if all(n in d for n in neigh) and neigh:
            ed = r["e_atom"] - sum(a * d[n]["e_atom"] for n, a in neigh.items())
        metal = r.get("N_EF", 0) >= 0.05
        nonmag = abs(r.get("mag", 0.0)) <= 0.1
        res[k] = dict(label=v["label"], Tc=v["Tc"], N_EF=r.get("N_EF"), mag=r.get("mag"), dft_decomp=ed,
                      metal=metal, nonmag=nonmag, pass_T1b=b.get(k, {}).get("pass_T1b"),
                      hull_evaluated=ed is not None,
                      pass_T2=bool(metal and nonmag and (ed is None or ed <= 0.10)), wall_s=r["wall_s"])
    Path(out).write_text(json.dumps(res, indent=1))
    for lab in ("pos", "neg", "mid"):
        rs = [x for x in res.values() if x["label"] == lab]
        print(lab, len(rs), "pass", sum(x["pass_T2"] for x in rs), "metal", sum(x["metal"] for x in rs),
              "hull<=0.1", sum(1 for x in rs if x["dft_decomp"] is not None and x["dft_decomp"] <= 0.10))


def cmd_hull1(ms, mace, pid, refs, out):
    """T1a E_hull of one phase against named reference phases (+ itself), per MLIP; pass if <= 0.10 in either."""
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Structure
    res = dict(id=pid)
    for model, p in (("mattersim", ms), ("mace", mace)):
        rec = load(p)
        ents = []
        for k in [pid] + refs.split(","):
            r = rec[k]
            comp = Structure.from_dict(r["structure"]).composition
            ents.append(PDEntry(comp, r["e_atom"] * r["nat"], name=k))
        pd = PhaseDiagram(ents)
        me = [e for e in ents if e.name == pid][0]
        dec = pd.get_decomposition(me.composition)
        res[model] = dict(ehull=float(pd.get_e_above_hull(me)), decomposition={e.name: float(a) for e, a in dec.items()})
    res["pass_T1a_hull"] = any(res[m]["ehull"] <= 0.10 for m in ("mattersim", "mace"))
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def t1b_rule(a, b):
    """Two-model T1b rule (FUNNEL.md section 7). a, b: T1b records (MatterSim, MACE)."""
    ua = not a.get("pass_T1b", False)
    ub = not b.get("pass_T1b", False)
    return dict(ms_wmin=a.get("wmin"), mace_wmin=b.get("wmin"), ms_status=a.get("status"), mace_status=b.get("status"),
                ms_unstable=ua, mace_unstable=ub, pass_T1b=not (ua and ub), t1b_split=ua != ub)


def cmd_t1b2(ms, mace, out):
    a, b = load(ms, key="pass_T1b"), load(mace, key="pass_T1b")
    res = {k: t1b_rule(a[k], b[k]) for k in a if k in b}
    Path(out).write_text(json.dumps(res, indent=1))
    for k, r in res.items():
        print(f"{k:10s} MS {r['ms_wmin']:9.1f}  MACE {r['mace_wmin']:9.1f}  pass {r['pass_T1b']}  split {r['t1b_split']}")


def cmd_access2(ms, mace, t1bms, t1bmace, t2, dftzpe, P, out):
    e = {"mattersim": {k: v["e_atom"] for k, v in load(ms).items()},
         "mace": {k: v["e_atom"] for k, v in load(mace).items()}}
    t2r = load(t2, by=("id", "xc"))
    for xc in ("pbe", "pbesol"):
        e[f"dft-{xc}"] = {k: r["e_atom"] for (k, x), r in t2r.items() if x == xc}
    bm, bc = load(t1bms, key="zpe_eV"), load(t1bmace, key="zpe_eV")
    dz = {k: v for k, v in load(dftzpe, key="zpe_eV").items() if v.get("status") == "ok"}
    phases = sorted({P} | {p for prod, _ in RX[P] for p in prod})
    stab = {p: dict(ms_wmin0=bm[p]["wmin0"], mace_wmin0=bc[p]["wmin0"],
                    stable=bm[p]["wmin0"] >= -30.0 and bc[p]["wmin0"] >= -30.0) for p in phases}
    all_stable = all(s["stable"] for s in stab.values())
    zpe, src = {}, {}
    for p in phases:
        if all_stable or stab[p]["stable"]:
            zpe[p], src[p] = bm[p]["zpe_gamma_eV"] / bm[p]["nat"], "MLIP Gamma (MatterSim)"
        else:
            zpe[p], src[p] = dz[p]["zpe_eV"] / dz[p]["nat"], "DFT Gamma (PBE, ph.x)"
    zf0 = {p: bm[p]["zpe_eV"] / bm[p]["nat"] for p in phases}
    row = dict(phases={p: dict(stab[p], zpe_source=src[p], zpe_cell_eV=zpe[p] * FORMULA_ATOMS[p],
                               zpe_F0_mesh_cell_eV=zf0[p] * FORMULA_ATOMS[p]) for p in phases}, all_stable=all_stable)

    for s, ea in e.items():
        _, h = delta(ea, zpe, P)
        _, h0 = delta(ea, zf0, P)
        row[s] = dict(hull_channel=h and h["channel"], dE=h and h["dE"], dE_zpe=h and h["dE_zpe"],
                      dE_zpe_F0mesh=h0 and h0["dE_zpe"])
    v = lambda s: row[s]["dE_zpe"]
    row["T1a_line_1.1"] = "pass" if any(v(s) is not None and v(s) <= 1.1 for s in ("mattersim", "mace")) else "reject"
    row["T2_gate"] = gate([v("dft-pbe"), v("dft-pbesol")])
    row["T2_gate_noZPE"] = gate([row["dft-pbe"]["dE"], row["dft-pbesol"]["dE"]])
    row["T2_gate_F0mesh"] = gate([row["dft-pbe"]["dE_zpe_F0mesh"], row["dft-pbesol"]["dE_zpe_F0mesh"]])
    Path(out).write_text(json.dumps(row, indent=1))
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    {"access": cmd_access, "hull": cmd_hull, "t2db": cmd_t2db, "hull1": cmd_hull1, "t1b2": cmd_t1b2,
     "access2": cmd_access2}[sys.argv[1]](*sys.argv[2:])
