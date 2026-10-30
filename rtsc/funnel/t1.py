"""T1 (FUNNEL.md): MLIP relax (T1a) and MLIP phonons with soft-mode following (T1b).

    python t1.py relax  <model> <structs.json> <out.jsonl>        # model: mattersim | mace
    python t1.py phonon <model> <relaxed.jsonl> <ids|all> <out.jsonl>

relax: FIRE + FrechetCellFilter (fmax 0.02 eV/A, <= 400 steps); molecules (id 'H2') relax positions only
in their box. Each record: id, energy/atom, relaxed structure, steps, wall seconds. Restart-safe: ids
already in out.jsonl are skipped.
phonon: phonopy finite displacements (0.01 A) in a supercell with every perpendicular height >= 10 A.
omega_min over the supercell's commensurate q (the 3 acoustic modes at Gamma excluded). If below -30 cm^-1,
the parent is displaced along that mode (0.15 A max), the supercell re-relaxed and reduced to its
primitive cell, and phonons recomputed: at most 2 rounds, the child replaces the parent. ZPE (T = 0
phonopy free energy, 8^3 mesh) in eV per parent formula cell; zpe_gamma_eV: Gamma-point ZPE of the parent cell
(sum hbar omega / 2 over the 3N - 3 non-acoustic Gamma modes, imaginary ones dropped and counted), the like-for-like
partner of the DFT Gamma-ZPE (zpe.py) under the F0b ZPE-source rule (FUNNEL.md section 7).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

THZ_TO_CM = 33.35641
CM_TO_EV = 1.239841984e-4
KJMOL_TO_EV = 0.0103642688


def calculator(model):
    if model == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device="cuda")
    if model == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model="medium-mpa-0", device="cuda", default_dtype="float64")
    raise SystemExit(f"unknown model {model}")


def relax(atoms, calc, molecule=False, fmax=0.02, steps=400):
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    atoms = atoms.copy()
    atoms.calc = calc
    obj = atoms if molecule else FrechetCellFilter(atoms)
    opt = FIRE(obj, logfile=None)
    ok = opt.run(fmax=fmax, steps=steps)
    return atoms, opt.nsteps, bool(ok)


def done_ids(out):
    if not Path(out).exists():
        return set()
    return {json.loads(ln)["id"] for ln in Path(out).read_text().splitlines() if ln.strip()}


def cmd_relax(model, src, out):
    calc = calculator(model)
    data = json.loads(Path(src).read_text())
    seen = done_ids(out)
    with open(out, "a") as fo:
        for k, d in data.items():
            if k in seen:
                continue
            s = Structure.from_dict(d["structure"])
            at = AseAtomsAdaptor.get_atoms(s)
            t0 = time.time()
            rec = dict(id=k, model=model, nat=len(s), formula=s.composition.reduced_formula)
            try:
                mol = k == "H2"
                if mol:
                    at.set_cell([12, 12, 12]); at.center(); at.pbc = True
                r, n, ok = relax(at, calc, molecule=mol)
                rec.update(e_atom=float(r.get_potential_energy()) / len(r), steps=n, converged=ok,
                           structure=AseAtomsAdaptor.get_structure(r).as_dict())
            except Exception as ex:  # a failure is a result
                rec.update(status=f"fail:{type(ex).__name__}:{ex}"[:200])
            rec["wall_s"] = round(time.time() - t0, 3)
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(k, rec.get("e_atom"), rec.get("steps"), rec["wall_s"], flush=True)


def supercell_dim(s, hmin=10.0):
    m = s.lattice.matrix
    v = abs(np.linalg.det(m))
    h = [v / np.linalg.norm(np.cross(m[(i + 1) % 3], m[(i + 2) % 3])) for i in range(3)]
    return [int(np.ceil(hmin / x)) for x in h]


def phonons(s, calc):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    u = PhonopyAtoms(symbols=[x.specie.symbol for x in s], cell=s.lattice.matrix, scaled_positions=s.frac_coords)
    dim = supercell_dim(s)
    ph = Phonopy(u, supercell_matrix=np.diag(dim), primitive_matrix="P")
    ph.generate_displacements(distance=0.01)
    forces = []
    for sc in ph.supercells_with_displacements:
        from ase import Atoms
        a = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        a.calc = calc
        forces.append(a.get_forces())
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    # commensurate q of the supercell
    qs = [np.array([i / dim[0], j / dim[1], k / dim[2]]) for i in range(dim[0]) for j in range(dim[1]) for k in range(dim[2])]
    wmin, qmin, imin = 1e9, None, None
    for q in qs:
        f = ph.get_frequencies(q) * THZ_TO_CM
        if np.allclose(q, 0):
            f[np.argsort(f)[:3]] = np.inf   # acoustic modes at Gamma
        j = int(np.argmin(f))
        if f[j] < wmin:
            wmin, qmin, imin = float(f[j]), q, j
    ph.run_mesh([8, 8, 8])
    ph.run_thermal_properties(t_min=0, t_max=0, t_step=1)
    zpe = float(ph.get_thermal_properties_dict()["free_energy"][0]) * KJMOL_TO_EV
    fg = np.sort(ph.get_frequencies([0, 0, 0]) * THZ_TO_CM)
    opt = fg[np.argsort(np.abs(fg))[3:]] if len(fg) > 3 else np.array([])
    zg = dict(zpe=float(0.5 * opt[opt > 0].sum() * CM_TO_EV), n_imag=int((opt < 0).sum()))
    return ph, dim, wmin, qmin, imin, zpe, len(ph.supercells_with_displacements), zg


def follow(s, ph, dim, q, band, calc, amp=0.15):
    """Displace the supercell along mode (q, band), relax, reduce to the primitive cell."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    _, evec = ph.get_frequencies_with_eigenvectors(q)
    e = evec[:, band].reshape(-1, 3)
    sc = s.copy(); sc.make_supercell(dim)
    nprim = len(s)
    disp = []
    masses = np.array([x.specie.atomic_mass for x in s])
    for i, site in enumerate(sc):
        p = i // int(np.prod(dim))      # pymatgen orders supercell images per parent site
        R = sc.lattice.get_cartesian_coords(site.frac_coords) - s.lattice.get_cartesian_coords(s[p].frac_coords)
        R = s.lattice.get_fractional_coords(R)
        d = np.real(e[p] * np.exp(2j * np.pi * np.dot(q, R))) / np.sqrt(masses[p])
        disp.append(d)
    disp = np.array(disp)
    disp *= amp / max(np.linalg.norm(disp, axis=1).max(), 1e-12)
    sc = Structure(sc.lattice, sc.species, sc.cart_coords + disp, coords_are_cartesian=True, to_unit_cell=True)
    at, n, ok = relax(AseAtomsAdaptor.get_atoms(sc), calc)
    child = AseAtomsAdaptor.get_structure(at)
    try:
        child = SpacegroupAnalyzer(child, symprec=0.05).find_primitive() or child
    except Exception:
        pass
    return child, float(at.get_potential_energy()) / len(at)


def cmd_phonon(model, src, ids, out):
    calc = calculator(model)
    recs = {}
    for ln in Path(src).read_text().splitlines():
        r = json.loads(ln)
        if "structure" in r:
            recs[r["id"]] = r
    want = list(recs) if ids == "all" else ids.split(",")
    seen = done_ids(out)
    with open(out, "a") as fo:
        for k in want:
            if k in seen or k not in recs or k == "H2":
                continue
            t0 = time.time()
            s = Structure.from_dict(recs[k]["structure"])
            rec = dict(id=k, model=model, nat=len(s), rounds=[])
            try:
                ph, dim, wmin, q, j, zpe, ndisp, zg = phonons(s, calc)
                rec.update(dim=dim, ndisp=ndisp, wmin0=wmin, zpe_eV=zpe * 1.0, zpe_gamma_eV=zg["zpe"],
                           gamma_n_imag=zg["n_imag"])
                rec["rounds"].append(dict(wmin=wmin, q=list(map(float, q)), nat=len(s)))
                cur = s
                for rnd in range(2):
                    if wmin >= -30.0:
                        break
                    child, e_child = follow(cur, ph, dim, q, j, calc)
                    if len(child) > 4 * len(s):
                        rec["status"] = "fail:soft-large-child"; break
                    cur = child
                    ph, dim, wmin, q, j, zpe_c, nd, _ = phonons(cur, calc)
                    rec["rounds"].append(dict(wmin=wmin, q=list(map(float, q)), nat=len(cur), e_atom=e_child))
                rec["wmin"] = wmin
                rec["pass_T1b"] = bool(wmin >= -30.0)
                if len(rec["rounds"]) > 1:
                    rec["child"] = cur.as_dict()
            except Exception as ex:
                rec.update(status=f"fail:{type(ex).__name__}:{ex}"[:200], pass_T1b=False)
            rec["wall_s"] = round(time.time() - t0, 3)
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(k, rec.get("wmin"), rec.get("zpe_eV"), rec["wall_s"], flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    if sys.argv[1] == "relax":
        cmd_relax(*sys.argv[2:5])
    elif sys.argv[1] == "phonon":
        cmd_phonon(*sys.argv[2:6])
