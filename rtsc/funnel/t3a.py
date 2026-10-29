"""T3a (FUNNEL.md): DFT vc-relax -> scf -> ph.x (engine, MP) -> nscf -> EPW alpha^2F -> lambda, omega_log, Tc^AD.

    python3 t3a.py <jobs.json> <id> <workroot> [NT omp threads, default 8] [NE epw ranks, default 8]

jobs.json: {id: {structure: pymatgen dict, Ttarget: K, press_kbar?: float}}.
Writes <workroot>/<id>/result.json (status ok | fail:<stage>), timings.tsv; restart-safe per stage marker.
Environment: BIN (QE bin dir), PSEUDO_DIR. RTSC_STH=1, RTSC_STRICT=1, RTSC_STH_MP=1 are set for ph.x.
EPW computes alpha^2F only (eliashberg, no liso/laniso); one retry with SCDM projections on a Wannier failure.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

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
    st0 = qe.from_pmg(job["structure"])
    res = dict(id=jid, Ttarget=job.get("Ttarget"), stages={}, nat=len(st0["species"]))
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
    ecut = qe.ecut_rule(st0["species"])
    g = qe.t3a_grids(st0)
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
        g = qe.t3a_grids(st)
        res["grids"] = dict(g, ecut=ecut)
        import re
        ef = float(re.findall(r"the Fermi energy is\s+(-?[\d.]+) ev", Path("scf.out").read_text())[-1])
        xs = dict(xn, ef=ef)
        res["ef_scf"] = ef
        freqs = [float(c) for _, c in re.findall(r"freq \(\s*\d+\) =\s*(-?[\d.]+) \[THz\] =\s*(-?[\d.]+) \[cm-1\]",
                                                   Path("ph.out").read_text())]
        res["omega_min_cm"] = min(freqs) if freqs else None
        return epw_stage(res, st, g, xs, xn, prefix, ne, stage, finish, job)
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
    st = x["structure"]
    res["relaxed"] = st
    shutil.rmtree(f"{prefix}.save", ignore_errors=True)
    for f in Path(".").glob(f"{prefix}.wfc*"):
        f.unlink()
    g = qe.t3a_grids(st)      # grids from the relaxed cell
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
    nsemi = qe.n_semicore(xs)
    nw = qe.n_wann(st["species"])
    nbnd = nsemi + nw + max(8, nw // 2)
    res["wannier"] = dict(nsemi=nsemi, nw=nw, nbnd=nbnd)
    Path("nscf.in").write_text(qe.pw_input(st, "nscf", prefix, None, ecut, pseudo=PSEUDO, nbnd=nbnd,
                                           kblock=qe.kfull(g["nk"])))
    if stage("nscf", mpi + [f"{BIN}/pw.x", "-in", "nscf.in"], "nscf.out", omp) != 0:
        return finish("fail:nscf")
    xn = qe.read_xml(f"{prefix}.save/data-file-schema.xml")
    return epw_stage(res, st, g, xs, xn, prefix, ne, stage, finish, job)


def epw_stage(res, st, g, xs, xn, prefix, ne, stage, finish, job):
    nsemi = qe.n_semicore(xs if xs.get("eig") else xn)
    nw = qe.n_wann(st["species"])
    res.setdefault("wannier", {}).update(nsemi=nsemi, nw=nw)
    xn = dict(xn, ef=xs["ef"])
    froz = qe.frozen_window(xn, nsemi, nw)
    res["wannier"]["froz"] = froz
    # 6. EPW, atomic projections then SCDM
    epw_cmd = (f"nvidia-cuda-mps-control -d 2>/dev/null; OMP_NUM_THREADS=1 mpirun --bind-to none -np {ne} "
               f"-x OMP_NUM_THREADS {BIN}/epw.x -npool {ne} -in epw.in")
    for attempt, scdm in (("epw_atomic", False), ("epw_scdm", True)):
        if not scdm and froz is None:
            res["wannier"]["note"] = "no frozen window with <= nWF bands"
            continue
        for f in Path(".").glob(f"{prefix}.a2f*"):
            f.unlink()
        Path("epw.in").write_text(qe.epw_input(prefix, st["species"], g["nk"], g["nq"], g["nkf"], g["nqf"], nsemi, nw,
                                               froz or 0.0, xs["ef"], scdm=scdm))
        shutil.copy("epw.in", f"{attempt}.in")
        stage(attempt, epw_cmd, f"{attempt}.out", timeout=3 * 3600)
        if Path(f"{prefix}.a2f_proj").exists():
            res["wannier"]["method"] = "scdm" if scdm else "atomic"
            wout = Path(f"{prefix}.wout")
            if wout.exists():
                sp = [float(l.split()[-1]) for l in wout.read_text().splitlines() if "WF centre and spread" in l]
                res["wannier"]["max_spread_A2"] = max(sp[-nw:]) if sp else None
            m = qe.a2f_metrics(f"{prefix}.a2f_proj", muc=0.10)
            res["metrics"] = m
            T = job.get("Ttarget") or 1.0
            res["line_K"] = 0.3 * T
            res["pass_T3a"] = bool(m["tc_ad"] >= 0.3 * T)
            # keep only small outputs
            return finish("ok")
    return finish("fail:wannier")


if __name__ == "__main__":
    main(*sys.argv[1:])
