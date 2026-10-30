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
G4 is registered in FUNNEL.md section 8 (2026-10-30). F0c fixes: no frozen window -> neither setup runs (SCDM without a
window is retired); a 3q result replaces the 2q one only if a 3q setup passes its gate.
G5 (section 8): ecut = qe.ecut_rule (the active set's normal hints; env PSET = sg15 | spms).
S1 (section 8, s1.py): any DFPT mode < -30 cm^-1 (Gamma acoustic excluded) -> no pass; EPW skipped; one re-relax along the
most negative mode and a rerun (child <id>__s1, cap S1_CAP s); still unstable -> status unstable_dfpt, disposition sscha
(H participation > 0.5) or reject.
Workflow-failure routing (section 8): a fail:* keeps its status and gets `route`; one automatic T3b-lite attempt
(child <id>__t3bl: q + 1, scf k = smallest multiple of that q >= 4/3 x the T3a k, Wannier k = 2 q, T3a pass rule); then the
hand queue (<workroot>/hand_queue.jsonl). fail:timeout -> hand queue.
Study overrides in a job (not generator rules): st (plain dict), ecut, nq, kscf, no_relax.
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


S1_CAP = int(os.environ.get("S1_CAP", "2700"))
T3BL_CAP = int(os.environ.get("T3BL_CAP", "3600"))


def child(jobs, jid, root, cid, cjob, nt, ne, cap):
    """Run a child T3a job (S1 rerun or T3b-lite) in <root>/<cid> under a wall cap; returns its result dict."""
    allj = json.loads(Path(jobs).read_text())
    allj[cid] = cjob
    cj = Path(root).resolve() / f"{cid}.jobs.json"
    cj.write_text(json.dumps(allj))
    t0 = time.time()
    pr = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(cj), cid, str(Path(root).resolve()), nt, ne],
                          stdout=open(Path(root) / f"{cid}.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        pr.wait(timeout=cap)
    except subprocess.TimeoutExpired:
        import signal
        os.killpg(pr.pid, signal.SIGTERM)     # the child's own session: its mpirun/pw.x/ph.x/epw.x go too
        time.sleep(5)
        try:
            os.killpg(pr.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return dict(status="fail:timeout", wall_s=round(time.time() - t0, 1))
    rp = Path(root) / cid / "result.json"
    r = json.loads(rp.read_text()) if rp.exists() else dict(status="fail:child")
    r["wall_s"] = round(time.time() - t0, 1)
    return r


def main(jobs, jid, root, nt="8", ne="8"):
    root = str(Path(root).resolve())
    jobs = str(Path(jobs).resolve())          # run_chain chdirs into the job directory
    job = json.loads(Path(jobs).read_text())[jid]
    W = Path(root) / jid
    if os.environ.get("T3A_POST_ONLY") and (W / "result.json").exists():
        res = json.loads((W / "result.json").read_text())   # resume S1 / routing on a finished chain
    else:
        res = run_chain(jobs, jid, root, nt, ne)
    if res["status"] == "s1_rerelax":                    # S1: one re-relax along the unstable mode, then a rerun
        import s1
        stc, ncell = s1.displaced(W, res["relaxed"], res["s1"]["worst"])
        cj = {k: v for k, v in job.items() if k not in ("structure", "st", "no_relax")}
        cj.update(st=stc, s1_rerun=True, s1_supercell=ncell)
        cr = child(jobs, jid, root, f"{jid}__s1", cj, nt, ne, S1_CAP)
        res["s1"]["rerun"] = json.loads(json.dumps({k: cr.get(k) for k in (
            "status", "metrics", "pass_T3a", "s1", "omega_min_cm", "wall_s", "total_s", "flag_wannier", "S_W")}))
        res["s1"]["supercell"] = ncell
        res["stages"]["s1_child"] = cr.get("wall_s", 0.0)
        if cr["status"] == "ok":                         # stable after the re-relax: the child's verdict stands
            res.update(status="ok", metrics=cr["metrics"], pass_T3a=cr["pass_T3a"], s1_rerelaxed=True,
                       flag_wannier=cr.get("flag_wannier"), S_W=cr.get("S_W"))
        else:
            wst = (cr.get("s1") or res["s1"])["worst"]
            res.update(status="unstable_dfpt", pass_T3a=False,
                       disposition="sscha" if wst["H_participation"] > 0.5 else "reject")
            if cr["status"] != "unstable_dfpt":
                res["s1"]["note"] = f"rerun {'cost-skipped' if cr['status'] == 'fail:timeout' else 'failed'} ({cr['status']})"
    elif res["status"].startswith("fail:") and not job.get("t3b_lite") and not job.get("s1_rerun"):
        route = ["G4 alternative path (setup B, 3q refinement)"] if res["status"] in ("fail:wannier",) else []
        if res["status"] != "fail:timeout":
            g = res.get("grids") or {}
            nq = (g.get("nq") or 2) + 1
            k = g.get("k") or [nq] * 3
            kscf = [nq * math.ceil(4 * x / (3 * nq)) for x in k]
            cj = {kk: v for kk, v in job.items() if kk not in ("nq", "kscf")}
            cj.update(nq=nq, kscf=kscf, t3b_lite=True)
            cr = child(jobs, jid, root, f"{jid}__t3bl", cj, nt, ne, T3BL_CAP)
            route.append(f"T3b-lite: {cr['status']}")
            res["t3b_lite"] = {k: cr.get(k) for k in ("status", "metrics", "pass_T3a", "grids", "wall_s", "flag_wannier")}
            res["stages"]["t3b_lite_child"] = cr.get("wall_s", 0.0)
            if cr["status"] == "ok":
                res["pass_T3a"] = cr["pass_T3a"]
                res["resolved_by"] = "t3b_lite"
        if "resolved_by" not in res:
            route.append("hand queue")
            with open(Path(root) / "hand_queue.jsonl", "a") as fo:
                fo.write(json.dumps(dict(id=jid, status=res["status"], t3b_lite=res.get("t3b_lite", {}).get("status"))) + "\n")
        res["route"] = route
    res["total_s"] = round(sum(res["stages"].values()), 2)
    (W / "result.json").write_text(json.dumps(res, indent=1))
    print("FINAL", jid, res["status"], res.get("pass_T3a"), res.get("disposition"), res.get("metrics"), flush=True)
    return res


def run_chain(jobs, jid, root, nt="8", ne="8"):
    job = json.loads(Path(jobs).read_text())[jid]
    W = Path(root) / jid
    W.mkdir(parents=True, exist_ok=True)
    os.chdir(W)
    st0 = qe.canonical(job["st"] if "st" in job else qe.from_pmg(job["structure"]))   # G4(a): wrapped into [0,1), sites sorted by Z
    res = dict(id=jid, Ttarget=job.get("Ttarget"), stages={}, nat=len(st0["species"]), generator="G1-G5+S1",
               pset=qe.pset(), overrides={k: job[k] for k in ("ecut", "nq", "kscf", "no_relax", "t3b_lite", "s1_rerun")
                                          if k in job})
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
    g = grids(st0, job)
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
        g = grids(st, job)
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


def grids(st, job):
    g = qe.t3a_grids(st, nq=job.get("nq"))
    if job.get("kscf"):
        g["k"] = list(job["kscf"])
    return g


def after_relax(res, st, ecut, nq_over, prefix, ne, stage, finish, job, mpi, omp):
    res["relaxed"] = st
    shutil.rmtree(f"{prefix}.save", ignore_errors=True)
    for f in Path(".").glob(f"{prefix}.wfc*"):
        f.unlink()
    g = grids(st, job)      # grids from the relaxed cell
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
    import s1   # S1: DFPT stability (Gamma acoustic excluded); an unstable candidate does not pass and skips EPW
    try:
        res["s1"] = s1.analyze(".", prefix)
    except Exception as ex:  # noqa: BLE001
        res["s1_error"] = str(ex)[:200]
        return finish("fail:s1")
    if res["s1"]["unstable"]:
        res["pass_T3a"] = False
        return finish("unstable_dfpt" if job.get("s1_rerun") else "s1_rerelax")
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
        if froz is None:     # F0c fix: no frozen window -> no setup (SCDM without a window is the retired G3 leg)
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
