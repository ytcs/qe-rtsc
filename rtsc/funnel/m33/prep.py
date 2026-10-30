"""M3-3 Phase 0 inputs (M3-3-LOG pre-registration): B1 and B2 frozen cells/ecut/sigma (bench/mkbench.py), Si control.

    prep.py <system B1|B2|Si> <outdir> <pseudo dir>

Writes scf.in (production k), [scf_ph.in (smoke k, B2 only)], pp.in, ph.in (single q, irrep 1, fildvscf), nscf_f.in (full N_f^3,
no symmetry), nscf_c.in (full N_c^3, Wannier bands). prefix 'x'.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import mkbench  # noqa: E402

SI = dict(prefix="x", cell="ibrav = 2\n    celldm(1) = 10.2612", nat=2, ntyp=1,
          species=[("Si", 28.086, "Si_ONCV_PBE-1.2.upf")], units="alat",
          positions=[("Si", 0, 0, 0), ("Si", 0.25, 0.25, 0.25)], ecutwfc=36, degauss=0.02, nbnd=12)
# N_f (production DFPT k), N_c (realisable coarse), q (B1, Si: explicit cart 2pi/alat; B2: q5 of the ldisp 4^3 list),
# nbnd for the fine (std) and coarse (Wannier) nscf
CFG = dict(B1=dict(nf=18, nc=6, kscf=18, q="0.0 0.0 0.145933920287798", nbf=16, nbc=30),
           B2=dict(nf=12, nc=4, kscf=12, kph=6, nq=4, iq=5, nbf=40, nbc=72),
           Si=dict(nf=12, nc=4, kscf=12, q="-0.25 -0.25 0.25", nbf=12, nbc=30))


def kfull(n):
    pts = list(itertools.product(range(n), repeat=3))
    return f"K_POINTS crystal\n{len(pts)}\n" + "".join(f" {i / n:.10f} {j / n:.10f} {k / n:.10f} 1.0\n" for i, j, k in pts)


def main(name, out, pseudo):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    s = dict(mkbench.SYSTEMS[name], prefix="x") if name != "Si" else SI
    c = CFG[name]
    mkbench.PSEUDO = pseudo
    pw = lambda calc, kb, extra="": mkbench.pw(s, calc, kb, extra).replace("@PSEUDO@", pseudo)  # noqa: E731
    k = c["kscf"]
    (out / "scf.in").write_text(pw("scf", f"K_POINTS automatic\n{k} {k} {k} 0 0 0\n"))
    if "kph" in c:
        kp = c["kph"]
        (out / "scf_ph.in").write_text(pw("scf", f"K_POINTS automatic\n{kp} {kp} {kp} 0 0 0\n").replace(
            "outdir = './'", "outdir = './phrun'"))
    (out / "nscf_f.in").write_text(pw("nscf", kfull(c["nf"]), f"\n    nbnd = {c['nbf']}\n    nosym = .true.\n    noinv = .true.")
                                   .replace("outdir = './'", "outdir = './nf'"))
    (out / "nscf_c.in").write_text(pw("nscf", kfull(c["nc"]), f"\n    nbnd = {c['nbc']}\n    nosym = .true.\n    noinv = .true.")
                                   .replace("outdir = './'", "outdir = './nc'"))
    (out / "pp.in").write_text("&inputpp\n  prefix = 'x'\n  outdir = './'\n  filplot = 'veff.dat'\n  plot_num = 1\n/\n")
    masses = "".join(f"  amass({i + 1}) = {m}\n" for i, (_, m, _) in enumerate(s["species"]))
    phdir = "./phrun" if "kph" in c else "./"
    if "q" in c:
        ph = (f"single q\n&inputph\n  prefix = 'x'\n  fildyn = 'x.dyn'\n{masses}  outdir = '{phdir}'\n  fildvscf = 'dvscf'\n"
              f"  tr2_ph = 1.0d-14\n  start_irr = 1\n  last_irr = 1\n/\n{c['q']}\n")
    else:
        ph = (f"q{c['iq']} of {c['nq']}^3\n&inputph\n  prefix = 'x'\n  fildyn = 'x.dyn'\n{masses}  outdir = '{phdir}'\n"
              f"  ldisp = .true.\n  nq1 = {c['nq']}, nq2 = {c['nq']}, nq3 = {c['nq']}\n  start_q = {c['iq']}\n"
              f"  last_q = {c['iq']}\n  fildvscf = 'dvscf'\n  tr2_ph = 1.0d-14\n  start_irr = 1\n  last_irr = 1\n/\n")
    (out / "ph.in").write_text(ph)
    (out / "cfg.txt").write_text(repr(c) + "\n")
    print(name, c)


if __name__ == "__main__":
    main(*sys.argv[1:])
