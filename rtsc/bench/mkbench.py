"""Generate frozen benchmark inputs (MILESTONE-1.md).

    python mkbench.py B1 frozen   -> B1/frozen/
    python mkbench.py B1 smoke    -> B1/smoke/   (reduced grids, local profiling only)

The frozen parameter sets must not change after a result has been recorded.
"""
import itertools
import pathlib
import sys

PSEUDO = "@PSEUDO@"  # filled in by run.sh from $PSEUDO_DIR

SYSTEMS = {
    "B1": dict(
        prefix="MgB2",
        cell="ibrav = 4\n    celldm(1) = 5.8260252227888\n    celldm(3) = 1.1420694129095",
        nat=3, ntyp=2,
        species=[("Mg", 24.305, "Mg_ONCV_PBE-1.2.upf"), ("B", 10.811, "B_ONCV_PBE-1.2.upf")],
        positions=[("Mg", 0, 0, 0), ("B", 1/3, 2/3, 0.5), ("B", 2/3, 1/3, 0.5)],
        ecutwfc=60, degauss=0.02, nbnd=16,
        # SG15 Mg carries 2s2p semicore (bands 1-4, at -65/-32 eV): keep them out of
        # the Wannier subspace. Frozen window top = E_F(scf, 9.56 eV) + 1.4 eV.
        wannier=dict(nbndsub=5, dis_froz_max=11.0, exclude="1-4",
                     proj=["B:pz", "f=0.5,1.0,0.5:s", "f=0.0,0.5,0.5:s", "f=0.5,0.5,0.5:s"]),
        grids=dict(
            frozen=dict(kscf=18, kq=6, kf=48, qf=24, temps=(10, 30), nstemp=5),
            smoke=dict(kscf=12, kq=3, kf=12, qf=6, temps=(20, 50), nstemp=2),
        ),
    ),
    "B2": dict(
        prefix="Mg2IrH6",
        # Fm-3m, a = 6.6513 A from the SG15/PBE vc-relax (B2/Mg2IrH6_relaxed.cif); Ir-H = 1.7204 A.
        cell="ibrav = 2\n    celldm(1) = 12.569178",
        nat=9, ntyp=3,
        species=[("Mg", 24.305, "Mg_ONCV_PBE-1.2.upf"), ("Ir", 192.217, "Ir_ONCV_PBE-1.2.upf"),
                 ("H", 1.008, "H_ONCV_PBE-1.2.upf")],
        units="alat",
        positions=[("Ir", 0, 0, 0), ("Mg", 0.25, 0.25, 0.25), ("Mg", -0.25, -0.25, -0.25),
                   ("H", 0.258656, 0, 0), ("H", -0.258656, 0, 0), ("H", 0, 0.258656, 0),
                   ("H", 0, -0.258656, 0), ("H", 0, 0, 0.258656), ("H", 0, 0, -0.258656)],
        ecutwfc=80, degauss=0.02, nbnd=40,
        # Bands 1-12: Mg 2s2p + Ir 5s5p semicore. Bands 13-21: H-s bonding + Ir t2g (filled);
        # 22-23: Ir-eg*/H-s antibonding, crossing E_F (10.95 eV). Band 24 starts at E_F+1.17 eV,
        # so the frozen window stops at E_F+1.1 eV. H phonons reach ~250 meV: wscut 2 eV.
        wannier=dict(nbndsub=11, dis_froz_max=12.05, exclude="1-12", proj=["Ir:d", "H:s"]),
        wscut=2.0,
        grids=dict(
            frozen=dict(kscf=12, kq=4, kf=32, qf=16, temps=(10, 25), nstemp=4),
            smoke=dict(kscf=6, kq=2, kf=8, qf=4, temps=(10, 20), nstemp=2),
        ),
    ),
}


# Every temperature must sit below the T_c that the given grid produces: above T_c the
# anisotropic solver's Broyden update goes singular and EPW stops (B2 smoke at 80 K with
# McMillan T_c 28 K). The frozen sets therefore stay at 10-30 K and the accuracy gate
# compares gaps there (MILESTONE-1.md, 2026-10-27 amendment).


def species_block(s):
    return "ATOMIC_SPECIES\n" + "".join(f" {el} {m} {pp}\n" for el, m, pp in s["species"])


def positions_block(s):
    return f"ATOMIC_POSITIONS {s.get('units', 'crystal')}\n" + "".join(
        f" {el} {x:.10f} {y:.10f} {z:.10f}\n" for el, x, y, z in s["positions"])


def pw(s, calc, kblock, extra=""):
    return f"""&control
    calculation = '{calc}'
    prefix = '{s["prefix"]}'
    pseudo_dir = '{PSEUDO}'
    outdir = './'
/
&system
    {s["cell"]}
    nat = {s["nat"]}
    ntyp = {s["ntyp"]}
    ecutwfc = {s["ecutwfc"]}
    occupations = 'smearing'
    smearing = 'mp'
    degauss = {s["degauss"]}{extra}
/
&electrons
    conv_thr = 1.0d-10
    mixing_beta = 0.7
/
{species_block(s)}{positions_block(s)}{kblock}"""


def kfull(n):
    pts = list(itertools.product(range(n), repeat=3))
    w = 1.0 / len(pts)
    return f"K_POINTS crystal\n{len(pts)}\n" + "".join(
        f" {i/n:.10f} {j/n:.10f} {k/n:.10f} {w:.6e}\n" for i, j, k in pts)


def ph(s, g):
    masses = "".join(f"  amass({i+1}) = {m}\n" for i, (_, m, _) in enumerate(s["species"]))
    return f"""&inputph
  prefix = '{s["prefix"]}'
  fildyn = '{s["prefix"]}.dyn'
{masses}  outdir = './'
  ldisp = .true.
  fildvscf = 'dvscf'
  nq1 = {g["kq"]}, nq2 = {g["kq"]}, nq3 = {g["kq"]}
  tr2_ph = 1.0d-14
  reduce_io = .true.
/
"""


def epw(s, g):
    w = s["wannier"]
    masses = "".join(f"  amass({i+1}) = {m}\n" for i, (_, m, _) in enumerate(s["species"]))
    proj = "".join(f"  proj({i+1}) = '{p}'\n" for i, p in enumerate(w["proj"]))
    n, kf, qf = g["kq"], g["kf"], g["qf"]
    return f"""&inputepw
  prefix = '{s["prefix"]}'
{masses}  outdir = './'
  ep_coupling = .true.
  elph = .true.
  epbwrite = .true.
  epwwrite = .true.
  etf_mem = 1
  nbndsub = {w["nbndsub"]}
  bands_skipped = 'exclude_bands = {w["exclude"]}'
  wannierize = .true.
  num_iter = 500
  dis_froz_max = {w["dis_froz_max"]}
{proj}  eps_acoustic = 2.0
  ephwrite = .true.
  fsthick = 0.4
  degaussw = 0.10
  degaussq = 0.5
  nqstep = 500
  eliashberg = .true.
  laniso = .true.
  limag = .true.
  lpade = .true.
  conv_thr_iaxis = 1.0d-4
  wscut = {s.get('wscut', 1.0)}
  temps = {g["temps"][0]} {g["temps"][1]}
  nstemp = {g["nstemp"]}
  nsiter = 500
  muc = 0.16
  dvscf_dir = './save'
  nk1 = {n}, nk2 = {n}, nk3 = {n}
  nq1 = {n}, nq2 = {n}, nq3 = {n}
  mp_mesh_k = .true.
  nkf1 = {kf}, nkf2 = {kf}, nkf3 = {kf}
  nqf1 = {qf}, nqf2 = {qf}, nqf3 = {qf}
/
"""


def main(sysid, variant):
    s = SYSTEMS[sysid]
    g = s["grids"][variant]
    out = pathlib.Path(__file__).parent / sysid / variant
    out.mkdir(parents=True, exist_ok=True)
    k = g["kscf"]
    (out / "scf.in").write_text(pw(s, "scf", f"K_POINTS automatic\n{k} {k} {k} 0 0 0\n"))
    (out / "ph.in").write_text(ph(s, g))
    (out / "nscf.in").write_text(pw(s, "nscf", kfull(g["kq"]), f"\n    nbnd = {s['nbnd']}"))
    (out / "epw.in").write_text(epw(s, g))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
