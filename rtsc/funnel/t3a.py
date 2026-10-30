"""T3a (FUNNEL.md): DFT vc-relax -> scf -> ph.x (engine, MP) -> nscf -> EPW alpha^2F -> lambda, omega_log, Tc^AD.

    python3 t3a.py <jobs.json> <id> <workroot> [NT omp threads, default 8] [NE epw ranks, default 8]

jobs.json: {id: {structure: pymatgen dict, Ttarget: K, press_kbar?: float, g4_refine?: bool}}.
Writes <workroot>/<id>/result.json (status ok | fail:<stage>), timings.tsv; restart-safe per stage marker.
Environment: BIN (QE bin dir), PSEUDO_DIR. RTSC_STH=1, RTSC_STRICT=1, RTSC_STH_MP=1 are set for ph.x.
EPW computes alpha^2F only (eliashberg, no liso/laniso).
Generator G1G2 (FUNNEL.md section 7): band exclusion by qe.n_semicore and Wannier coarse k = 2 q (qe.t3a_grids).
Generator G4 (proposed in WANNIER-LOG.md, study W1; not yet registered): (a) canonical cell (qe.canonical: fractional
coordinates wrapped into [0,1), sites sorted by Z); (b) EPW use_ws = .true.; (c) per-setup gate = G3 spread rule + on-grid
band check; (d) two setups, generic atomic projections and SCDM erfc (mu E_F, sigma 2 eV) WITH the frozen window; (e) if
they disagree (S_W > 0.15) or only one passes, Wannier k is refined to 3 q and both rerun; (f) still S_W > 0.15 ->
flag_wannier (E1 s x 2; pass rule uses the larger Tc); no setup passes -> fail:wannier (FUNNEL.md section 8 routing).
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import qe  # noqa: E402

BIN = os.environ.get("BIN", "/root/rtsc/qe-rtsc/build-gpu/bin")
PSEUDO = os.environ.get("PSEUDO_DIR", "/root/rtsc/qe-rtsc/rtsc/pseudo/sg15")


def run(cmd, out, env_extra=None, timeout=None):
    env = dict(os.environ, **(env_extra or {}))
    t0 = time.time()
    with open(out, "w") as fo:
        p = subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT, env=env, timeout=timeout, shell=isinstance(cmd, str))
    return p.returncode, time.time() - t0


def main(jobs, jid, root, nt="8", ne="8"):
    job = json.loads(Path(jobs).read_text())[jid]
    W = Path(root) / jid
    W.mkdir(parents=True, exist_ok=True)
    os.chdir(W)
    st0 = qe.canonical(job["st"] if "st" in job else qe.from_pmg(job["structure"]))   # G4(a): wrapped into [0,1), sites sorted by Z
    res = dict(id=jid, Ttarget=job.get("Ttarget"), stages={}, nat=len(st0["species"]), generator="G1G2")
    tsv = open("timings.tsv", "a")

    def stage(name, cmd, out, env_extra=None, timeout=None):
        rc, dt = run(cmd, out, env_extra, timeout)
        res["stages"][name] = round(dt, 2)
        tsv.write(f"{name}\t{dt:.2f}\t{rc}\n"); tsv.flush()
        return rc

    def finish(status):
        res["status"] = status
        res["total_s"] = round(sum(res["stages"].values()), 2)
        Path("result.json").write_text(json.dumps(res, indent=1))
        print(jid, status, res.get("metrics"), res["total_s"], flush=True)
        return res

    mpi = ["mpirun", "--bind-to", "none", "-np", "1", "-x", "OMP_NUM_THREADS", "-x", "RTSC_STH", "-x", "RTSC_STRICT",
           "-x", "RTSC_STH_MP"]
    omp = {"OMP_NUM_THREADS": nt, "RTSC_STH": "0", "RTSC_STRICT": "0", "RTSC_STH_MP": "0"}
    ecut = job.get("ecut") or qe.ecut_rule(st0["species"])
    nq_over = job.get("nq")          # study override (task 4: q 3^3); the generator rule is qe.t3a_grids
    g = qe.t3a_grids(st0, nq=nq_over)
    res["grids"] = dict(g, ecut=ecut)
    prefix = "x"
    done = {}
    if Path("timings.tsv").exists():
        for ln in Path("timings.tsv").read_text().splitlines():
            n, dt, rc = ln.split("\t")
            if rc == "0":
                done[n] = float(dt)
    if all(k in done for k in ("relax", "scf", "ph", "nscf")) and "JOB DONE" in Path("nscf.out").read_text():
        # resume at EPW (driver crash after nscf): stage times from the first run
        res["stages"].update({k: done[k] for k in ("relax", "scf", "ph", "nscf")})
        res["resumed_at"] = "epw"
        xn = qe.read_xml(f"{prefix}.save/data-file-schema.xml")
        st = xn["structure"]
        res["relaxed"] = st
        g = qe.t3a_grids(st, nq=nq_over)
        res["grids"] = dict(g, ecut=ecut)
        import re
        ef = float(re.findall(r"the Fermi energy is\s+(-?[\d.]+) ev", Path("scf.out").read_text())[-1])
        xs = dict(xn, ef=ef)
        res["ef_scf"] = ef
        freqs = [float(c) for _, c in re.findall(r"freq \(\s*\d+\) =\s*(-?[\d.]+) \[THz\] =\s*(-?[\d.]+) \[cm-1\]",
                                                   Path("ph.out").read_text())]
        res["omega_min_cm"] = min(freqs) if freqs else None
        return epw_stage(res, st, g, xs, xn, prefix, ne, stage, finish, job)
    if job.get("no_relax"):          # study override: the structure is already the T3a-relaxed cell
        return after_relax(res, st0, ecut, nq_over, prefix, ne, stage, finish, job, mpi, omp)
    # 1. relax (fixed-symmetry vc-relax at the target pressure)
    Path("relax.in").write_text(qe.pw_input(st0, "vc-relax", prefix, g["k"], ecut, pseudo=PSEUDO,
                                            press=job.get("press_kbar"), conv="1.0d-8"))
    rc = stage("relax", mpi + [f"{BIN}/pw.x", "-in", "relax.in"], "relax.out", omp)
    try:
        x = qe.read_xml(f"{prefix}.save/data-file-schema.xml")
    except Exception:
        return finish("fail:relax")
    if rc != 0 and "bfgs converged" not in Path("relax.out").read_text():
        res["relax_note"] = "not converged; continuing from last geometry"
    st = qe.canonical(x["structure"])   # G4(a): the relax does not wrap; re-canonicalise before scf/ph/nscf
    return after_relax(res, st, ecut, nq_over, prefix, ne, stage, finish, job, mpi, omp)


def after_relax(res, st, ecut, nq_over, prefix, ne, stage, finish, job, mpi, omp):
    res["relaxed"] = st
    shutil.rmtree(f"{prefix}.save", ignore_errors=True)
    for f in Path(".").glob(f"{prefix}.wfc*"):
        f.unlink()
    g = qe.t3a_grids(st, nq=nq_over)      # grids from the relaxed cell
    res["grids"] = dict(g, ecut=ecut)
    # 2. scf
    Path("scf.in").write_text(qe.pw_input(st, "scf", prefix, g["k"], ecut, pseudo=PSEUDO))
    env_sth = dict(omp, RTSC_STH="1", RTSC_STRICT="1", RTSC_STH_MP="1")
    if stage("scf", mpi + [f"{BIN}/pw.x", "-in", "scf.in"], "scf.out", env_sth) != 0:
        return finish("fail:scf")
    xs = qe.read_xml(f"{prefix}.save/data-file-schema.xml")
    res["ef_scf"] = xs["ef"]
    res["N_EF"] = qe.dos_ef(xs, len(st["species"]))
    # 3. ph
    Path("ph.in").write_text(qe.ph_input(prefix, st["species"], g["nq"]))
    if stage("ph", mpi + [f"{BIN}/ph.x", "-in", "ph.in"], "ph.out", env_sth) != 0:
        tail = Path("ph.out").read_text()[-600:]
        res["ph_tail"] = tail
        return finish("fail:ph")
    phout = Path("ph.out").read_text()
    import re
    freqs = [float(c) for _, c in re.findall(r"freq \(\s*\d+\) =\s*(-?[\d.]+) \[THz\] =\s*(-?[\d.]+) \[cm-1\]", phout)]
    res["omega_min_cm"] = min(freqs) if freqs else None
    # 4. collect (as run-fork.sh)
    nqirr = int(open(f"{prefix}.dyn0").read().split("\n")[1].split()[0])
    os.makedirs("save", exist_ok=True)
    for i in range(1, nqirr + 1):
        shutil.copy(f"{prefix}.dyn{i}", f"save/{prefix}.dyn_q{i}")
        if i == 1:
            shutil.copy(f"_ph0/{prefix}.dvscf1", f"save/{prefix}.dvscf_q1")
            if Path(f"save/{prefix}.phsave").exists():
                shutil.rmtree(f"save/{prefix}.phsave")
            shutil.copytree(f"_ph0/{prefix}.phsave", f"save/{prefix}.phsave")
        else:
            shutil.copy(f"_ph0/{prefix}.q_{i}/{prefix}.dvscf1", f"save/{prefix}.dvscf_q{i}")
    # 5. nscf on the EPW coarse grid
    try:
        nsemi, sinfo = qe.n_semicore(xs, st["species"], PSEUDO)
    except ValueError as ex:
        res["semicore_error"] = str(ex)
        return finish("fail:semicore")
    nw = qe.n_wann(st["species"])
    nbnd = nsemi + nw + max(8, nw // 2)
    res["wannier"] = dict(nsemi=nsemi, nw=nw, nbnd=nbnd, **sinfo)
    Path("nscf.in").write_text(qe.pw_input(st, "nscf", prefix, None, ecut, pseudo=PSEUDO, nbnd=nbnd,
                                           kblock=qe.kfull(g["nk"])))
    if stage("nscf", mpi + [f"{BIN}/pw.x", "-in", "nscf.in"], "nscf.out", omp) != 0:
        return finish("fail:nscf")
    xn = qe.read_xml(f"{prefix}.save/data-file-schema.xml")
    return epw_stage(res, st, g, xs, xn, prefix, ne, stage, finish, job)


def run_setups(st, g, nk, xs, xn, nsemi, nw, froz, prefix, ne, stage, tag=""):
    """G4(d): setup A = generic atomic projections, setup B = SCDM (erfc, mu = E_F, sigma = 2 eV) WITH the same frozen
    window, both with use_ws on the canonical cell; B reuses A's .epb files. Gate per setup (G4(c)): G3 spread rule and
    the on-grid band check (Wannier vs nscf eigenvalues at the coarse k, E_F +- 0.3 eV, <= 50 meV). Runs in the cwd."""
    epw_cmd = (f"nvidia-cuda-mps-control -d 2>/dev/null; OMP_NUM_THREADS=1 mpirun --bind-to none -np {ne} "
               f"-x OMP_NUM_THREADS {BIN}/epw.x -npool {ne} -in epw.in")
    setups = {}
    for attempt, scdm in (("epw_atomic", False), ("epw_scdmf", True)):
        if froz is None and not scdm:
            continue
        # EPW resumes from restart.fmt ("All q-points are done, no need to restart") and would silently reuse the
        # previous setup's interpolated e-ph data (found in W1: the second setup returned a stale lambda). Clear it.
        for pat in (f"{prefix}.a2f*", "restart.fmt", "selecq.fmt", f"{prefix}.ephmat", f"{prefix}.epmatwp", "epwdata.fmt",
                    "wigner.fmt", f"{prefix}.ukk", f"{prefix}.chk", f"{prefix}.wout"):
            for f in Path(".").glob(pat):
                shutil.rmtree(f) if f.is_dir() else f.unlink()
        epb = scdm and any(Path(".").glob(f"{prefix}.epb*"))
        Path("epw.in").write_text(qe.epw_input(prefix, st["species"], nk, g["nq"], g["nkf"], g["nqf"], nsemi, nw,
                                               froz, xs["ef"], scdm=scdm, epbread=epb))
        shutil.copy("epw.in", f"{attempt}.in")
        stage(attempt + tag, epw_cmd, f"{attempt}.out", timeout=3 * 3600)
        if not Path(f"{prefix}.a2f_proj").exists():
            setups[attempt] = dict(status="fail:epw")
            continue
        s = dict(status="ok", metrics=qe.a2f_metrics(f"{prefix}.a2f_proj", muc=0.10))
        wout = Path(f"{prefix}.wout")
        sp = [float(l.split()[-1]) for l in wout.read_text().splitlines() if "WF centre and spread" in l][-nw:] \
            if wout.exists() else []
        s["max_spread_A2"] = max(sp) if sp else None
        s["median_spread_A2"] = float(np.median(sp)) if sp else None
        s["g3_ok"] = bool(sp) and max(sp) <= 10.0 and max(sp) <= 5 * float(np.median(sp))
        try:
            s["band_err_meV"] = qe.band_check(".", xn, nk, nsemi, xs["ef"])
        except Exception as ex:  # noqa: BLE001
            s["band_err_meV"], s["band_err_note"] = None, str(ex)[:80]
        s["band_ok"] = s["band_err_meV"] is not None and s["band_err_meV"] <= 50.0
        s["gate_ok"] = s["g3_ok"] and s["band_ok"]
        shutil.copy(f"{prefix}.a2f_proj", f"{attempt}.a2f_proj")
        setups[attempt] = s
    good = [k for k in ("epw_atomic", "epw_scdmf") if setups.get(k, {}).get("gate_ok")]
    sw = None
    if len(good) == 2:
        la, lb = (setups[k]["metrics"]["lam"] for k in good)
        sw = abs(math.log(la / lb)) if la > 0 and lb > 0 else None
    return dict(nk=nk, setups=setups, good=good, S_W=sw, agree=(sw is not None and sw <= 0.15))


def epw_stage(res, st, g, xs, xn, prefix, ne, stage, finish, job):
    """G4 (WANNIER-LOG.md W1). (d) two setups at the G2 Wannier grid nk = 2q; (e) if they disagree (S_W > 0.15) or
    only one passes its gate, refine the Wannier grid to nk = 3q (new nscf in ./k3, same scf density and DFPT) and
    rerun both setups there; the refined result replaces the 2q one. (f) still S_W > 0.15 -> flag_wannier (E1: s x 2;
    the pass rule uses the larger Tc, never rejects on that basis); no setup passing its gate -> fail:wannier."""
    try:
        nsemi, sinfo = qe.n_semicore(xs if xs.get("eig") else xn, st["species"], PSEUDO)
    except ValueError as ex:
        res["semicore_error"] = str(ex)
        return finish("fail:semicore")
    nw = qe.n_wann(st["species"])
    res.setdefault("wannier", {}).update(nsemi=nsemi, nw=nw, **sinfo)
    xn = dict(xn, ef=xs["ef"])
    froz = qe.frozen_window(xn, nsemi, nw)
    res["wannier"]["froz"] = froz
    T = job.get("Ttarget") or 1.0
    res["line_K"] = 0.3 * T
    if froz is None:
        res["wannier"]["note"] = "no frozen window with <= nWF bands"
    r = run_setups(st, g, g["nk"], xs, xn, nsemi, nw, froz, prefix, ne, stage)
    res["wannier"]["k2q"] = r
    if not r["agree"] and job.get("g4_refine", True):
        nk3 = 3 * g["nq"]
        k3 = Path("k3")
        (k3 / f"{prefix}.save").mkdir(parents=True, exist_ok=True)
        for f in Path(f"{prefix}.save").iterdir():
            if not f.name.startswith("wfc"):
                shutil.copy(f, k3 / f"{prefix}.save" / f.name)
        if not (k3 / "save").exists():
            (k3 / "save").symlink_to(Path("save").resolve())
        cwd = os.getcwd()
        os.chdir(k3)
        nb = len(xn["eig"][0])
        Path("nscf.in").write_text(qe.pw_input(st, "nscf", prefix, None, res["grids"]["ecut"], pseudo=PSEUDO,
                                               nbnd=nb, kblock=qe.kfull(nk3)))
        mpi = ["mpirun", "--bind-to", "none", "-np", "1", "-x", "OMP_NUM_THREADS"]
        rc = stage("nscf_k3", mpi + [f"{BIN}/pw.x", "-in", "nscf.in"], "nscf.out",
                   {"OMP_NUM_THREADS": "8", "RTSC_STH": "0", "RTSC_STRICT": "0", "RTSC_STH_MP": "0"})
        if rc == 0:
            xn3 = dict(qe.read_xml(f"{prefix}.save/data-file-schema.xml"), ef=xs["ef"])
            froz3 = qe.frozen_window(xn3, nsemi, nw)
            r3 = run_setups(st, g, nk3, xs, xn3, nsemi, nw, froz3, prefix, ne, stage, tag="_k3")
            r3["froz"] = froz3
            res["wannier"]["k3q"] = r3
            if r3["good"]:
                r = r3
        os.chdir(cwd)
    res["wannier"]["final_nk"] = r["nk"]
    if not r["good"]:
        return finish("fail:wannier")
    prim = r["setups"][r["good"][0]]
    res["wannier"]["method"] = r["good"][0]
    res["metrics"] = prim["metrics"]
    res["S_W"] = r["S_W"]
    res["flag_wannier"] = not r["agree"]
    tcs = [r["setups"][k]["metrics"]["tc_ad"] for k in r["good"]]
    tc_used = max(tcs) if res["flag_wannier"] else prim["metrics"]["tc_ad"]
    res["pass_T3a"] = bool(tc_used >= 0.3 * T)
    return finish("ok")


if __name__ == "__main__":
    main(*sys.argv[1:])
