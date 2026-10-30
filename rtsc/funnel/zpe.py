"""DFT Gamma-ZPE for the access gate (FUNNEL.md section 7 ZPE-source rule; the LIGHT-DFT Q3 recipe on the GPU).

    python3 zpe.py <jobs.json> <ids,comma> <workroot> <out.jsonl> [NT]

jobs.json: {id: {structure: pymatgen dict}} (spglib-symmetrised by the caller). Per phase (<= 10 atoms):
PBE vc-relax (ecut 80 Ry, k 0.20 A^-1, MP 0.02 Ry) -> scf -> ph.x at q = 0 (tr2_ph 1e-14, FP64: RTSC_STH=1,
RTSC_STRICT=1, RTSC_STH_MP=0; if the engine declines, rerun with RTSC_STH=0, recorded as 'engine').
ZPE = sum hbar omega / 2 over the 3N - 3 modes left after dropping the three smallest |omega| (acoustic);
imaginary modes are dropped and counted. The PBE geometry's ZPE is used for both functionals (Q3).
Restart-safe on id.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qe  # noqa: E402

BIN = os.environ.get("BIN", "/root/rtsc/qe-rtsc/build-gpu/bin")
PSEUDO = os.environ.get("PSEUDO_DIR", "/root/rtsc/qe-rtsc/rtsc/pseudo/sg15")
CM_TO_EV = 1.239841984e-4


def main(jobs, ids, root, out, nt="8"):
    jobs = json.loads(Path(jobs).read_text())
    seen = set()
    if Path(out).exists():
        seen = {json.loads(l)["id"] for l in Path(out).read_text().splitlines() if l.strip()}
    mpi = ["mpirun", "--bind-to", "none", "-np", "1", "-x", "OMP_NUM_THREADS", "-x", "RTSC_STH", "-x", "RTSC_STRICT",
           "-x", "RTSC_STH_MP"]
    for k in [i for i in ids.split(",") if i]:
        if k in seen:
            continue
        st = qe.from_pmg(jobs[k]["structure"])
        W = Path(root) / k
        W.mkdir(parents=True, exist_ok=True)
        rec = dict(id=k, nat=len(st["species"]), stages={})
        t00 = time.time()

        def run(name, cmd, inp, sth="0"):
            env = dict(os.environ, OMP_NUM_THREADS=nt, RTSC_STH=sth, RTSC_STRICT=sth, RTSC_STH_MP="0")
            t0 = time.time()
            with open(W / f"{name}.out", "w") as f:
                rc = subprocess.run(mpi + cmd + ["-in", inp], cwd=W, stdout=f, stderr=subprocess.STDOUT, env=env,
                                    timeout=3600).returncode
            rec["stages"][name] = round(time.time() - t0, 2)
            return rc

        k3 = qe.kgrid(st["lattice"], 0.20)
        (W / "relax.in").write_text(qe.pw_input(st, "vc-relax", "z", k3, 80, pseudo=PSEUDO, conv="1.0d-8"))
        run("relax", [f"{BIN}/pw.x"], "relax.in")
        try:
            x = qe.read_xml(W / "z.save/data-file-schema.xml")
            st = x["structure"]
            shutil.rmtree(W / "z.save")
            k3 = qe.kgrid(st["lattice"], 0.20)
            (W / "scf.in").write_text(qe.pw_input(st, "scf", "z", k3, 80, pseudo=PSEUDO))
            if run("scf", [f"{BIN}/pw.x"], "scf.in", sth="1") != 0:
                raise RuntimeError("scf")
            sp = qe.species_order(st)
            masses = "".join(f"  amass({i + 1}) = {qe.MASS[e]}\n" for i, e in enumerate(sp))
            (W / "ph.in").write_text(f"&inputph\n  prefix = 'z'\n  fildyn = 'z.dyn'\n{masses}  outdir = './'\n"
                                     f"  tr2_ph = 1.0d-14\n/\n0.0 0.0 0.0\n")
            rec["engine"] = "sth-fp64"
            if run("ph", [f"{BIN}/ph.x"], "ph.in", sth="1") != 0:
                rec["engine"] = "vanilla"
                shutil.rmtree(W / "_ph0", ignore_errors=True)
                if run("ph", [f"{BIN}/ph.x"], "ph.in", sth="0") != 0:
                    raise RuntimeError("ph")
            f = [float(c) for c in re.findall(r"freq \(\s*\d+\) =\s*-?[\d.]+ \[THz\] =\s*(-?[\d.]+) \[cm-1\]",
                                               (W / "ph.out").read_text())]
            f = sorted(f, key=abs)[3:]
            rec.update(freqs_cm=sorted(f), n_imag=sum(1 for v in f if v < 0),
                       zpe_eV=0.5 * sum(v for v in f if v > 0) * CM_TO_EV, relaxed=st, energy_pbe=x["energy"])
            rec["status"] = "ok"
        except Exception as ex:
            rec["status"] = f"fail:{ex}"[:200]
        for d in ("z.save", "_ph0"):
            shutil.rmtree(W / d, ignore_errors=True)
        rec["wall_s"] = round(time.time() - t00, 2)
        with open(out, "a") as fo:
            fo.write(json.dumps(rec) + "\n")
        print(k, rec["status"], rec.get("zpe_eV"), rec.get("n_imag"), rec["wall_s"], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
