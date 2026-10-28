"""F0 control structures (FUNNEL.md section 2.1) as pymatgen Structures, written to controls.json.

    python structs.py <q2q3 dir with *_pbe_vc-relax_e90.in> <out.json>

Access controls and their reference phases start from the local Q2/Q3 DFT inputs where they exist;
the rest are built here from literature lattice constants (the T1 MLIP relax and the T3a DFT relax move
them to their own minima). Physics controls carry their target temperature and expected outcome.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure


def from_qe_in(path):
    t = Path(path).read_text()
    cell = re.search(r"CELL_PARAMETERS angstrom\n((?:.*\n){3})", t).group(1)
    lat = [[float(x) for x in ln.split()] for ln in cell.strip().splitlines()]
    nat = int(re.search(r"nat=(\d+)", t).group(1))
    pos = t.split("ATOMIC_POSITIONS crystal\n")[1].strip().splitlines()[:nat]
    sp = [ln.split()[0] for ln in pos]
    fr = [[float(x) for x in ln.split()[1:4]] for ln in pos]
    return Structure(lat, sp, fr)


def k2ptcl6(m, a, x):
    """Fm-3m Mg2MH6 (K2PtCl6 type) primitive cell, 9 atoms."""
    lat = Lattice([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    cart = [("Ir" if m == "Ir" else m, (0, 0, 0)), ("Mg", (a / 4,) * 3), ("Mg", (-a / 4,) * 3)]
    for s in (x, -x):
        for i in range(3):
            v = [0.0, 0.0, 0.0]
            v[i] = s * a
            cart.append(("H", tuple(v)))
    return Structure(lat, [c[0] for c in cart], [c[1] for c in cart], coords_are_cartesian=True)


def cac6():
    """R-3m CaC6 (Calandra-Mauri 2005): a_hex 4.333, c_hex 13.572; C layers at z = 1/6, 1/2, 5/6."""
    a, c = 4.333, 13.572
    lat = Lattice.hexagonal(a, c)
    layer = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1), (2, 2)]   # (i, j)/3 with i + j != 0 mod 3
    sp, fr = ["Ca"], [(0, 0, 0)]
    sp += ["C"] * 6
    fr += [(i / 3, j / 3, 1 / 6) for i, j in layer]
    conv = Structure.from_spacegroup(166, lat, sp[:1], fr[:1])  # Ca at 3a
    for s, f in zip(sp[1:], fr[1:]):
        for t in ((0, 0, 0), (2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)):
            conv.append(s, [(f[k] + t[k]) % 1.0 for k in range(3)])
    conv.merge_sites(tol=0.01, mode="delete")
    return conv.get_primitive_structure()


def mgb3c3():
    """hP7 Mg2/3B2C2 = MgB3C3 (Tomassetti et al. 2024): flat B-C honeycomb in a sqrt3 x sqrt3 cell,
    B and C alternating, one Mg per three hexagons. B-C 1.60 A, interlayer 3.6 A (starting guess)."""
    a, c = 3 * 1.60, 3.6
    lat = Lattice.hexagonal(a, c)
    sp, fr = ["Mg"], [(0, 0, 0.5)]
    for i, j in [(1, 0), (0, 1), (2, 2)]:      # i + j = 1 mod 3
        sp.append("B"); fr.append((i / 3, j / 3, 0))
    for i, j in [(2, 0), (0, 2), (1, 1)]:      # i + j = 2 mod 3
        sp.append("C"); fr.append((i / 3, j / 3, 0))
    return Structure(lat, sp, fr)


def main(q2q3, out):
    S = {}
    for f in sorted(Path(q2q3).glob("*_pbe_*relax_e90.in")):
        name = f.name.split("_pbe_")[0]
        S[name] = from_qe_in(f)
    S["Mg2FeH6"] = k2ptcl6("Fe", 6.443, 0.2402)   # Fe-H 1.548 A
    S["Fe"] = Structure(Lattice.cubic(2.83), ["Fe", "Fe"], [(0, 0, 0), (0.5, 0.5, 0.5)])
    S["Ir"] = Structure(Lattice([[0, 1.92, 1.92], [1.92, 0, 1.92], [1.92, 1.92, 0]]), ["Ir"], [(0, 0, 0)])
    vac = S["Mg2IrH6"].copy()
    vac.remove_sites([max(i for i, s in enumerate(vac) if s.specie.symbol == "H")])
    S["Mg2IrH5"] = vac
    # physics controls (T3a)
    fcc = lambda el, a: Structure(Lattice([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]), [el], [(0, 0, 0)])
    bcc = lambda el, a: Structure(Lattice([[-a / 2, a / 2, a / 2], [a / 2, -a / 2, a / 2], [a / 2, a / 2, -a / 2]]), [el], [(0, 0, 0)])
    S["Al"] = fcc("Al", 4.04)
    S["Cu"] = fcc("Cu", 3.615)
    S["Pb"] = fcc("Pb", 4.95)
    S["Nb"] = bcc("Nb", 3.30)
    a, c = 3.0834, 3.5213
    S["MgB2"] = Structure(Lattice.hexagonal(a, c), ["Mg", "B", "B"], [(0, 0, 0), (1 / 3, 2 / 3, 0.5), (2 / 3, 1 / 3, 0.5)])
    S["CaC6"] = cac6()
    S["MgB3C3"] = mgb3c3()
    a = 3.09   # Im-3m H3S near 200 GPa (Duan 2014; Errea 2015)
    S["H3S"] = Structure(Lattice([[-a / 2, a / 2, a / 2], [a / 2, -a / 2, a / 2], [a / 2, a / 2, -a / 2]]),
                         ["S", "H", "H", "H"], [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)])
    meta = {
        "Al": dict(role="physics", Ttarget=1.2, expect="weak positive; fails 40 K; Tc^AD > 0"),
        "Nb": dict(role="physics", Ttarget=9.0, expect="must pass T3a at 9 K"),
        "Pb": dict(role="physics", Ttarget=9.0, expect="must pass T3a at 9 K (no SOC, flagged)"),
        "Cu": dict(role="physics", Ttarget=1.0, expect="must fail: Tc^AD < 0.3 K"),
        "MgB2": dict(role="physics", Ttarget=39.0, expect="must pass T3a at 39 K"),
        "CaC6": dict(role="physics", Ttarget=11.5, expect="positive"),
        "H3S": dict(role="physics", Ttarget=77.0, expect="must pass T3a at 77 K", press_kbar=2000.0),
        "Mg2IrH6": dict(role="physics+access", Ttarget=77.0, expect="must pass T3a at 77 K; must FAIL access"),
        "MgB3C3": dict(role="physics", Ttarget=39.0, expect="known false positive: expected to pass e-ph tiers"),
        "MgH2": dict(role="access", expect="must PASS access"),
        "Mg2FeH6": dict(role="access", expect="must PASS access"),
        "Li2AgH6": dict(role="access", expect="must FAIL access"),
        "Li2AuH6": dict(role="access", expect="marginal (0.77)"),
    }
    out_d = {k: dict(structure=v.as_dict(), **meta.get(k, dict(role="reference"))) for k, v in S.items()}
    Path(out).write_text(json.dumps(out_d))
    for k, v in S.items():
        print(f"{k:9s} {v.composition.reduced_formula:9s} nat={len(v):2d} vol/at={v.volume / len(v):6.2f}  "
              f"dmin={np.min(v.distance_matrix + 99 * np.eye(len(v))):.3f}  {meta.get(k, {}).get('role', 'reference')}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
