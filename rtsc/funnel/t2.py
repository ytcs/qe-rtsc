"""T2 (FUNNEL.md): pw.x SCF single point at the MLIP geometry, PBE (+ PBEsol for hydrides), N(E_F), magnetisation.

    python3 t2.py <t1 relaxed.jsonl> <ids,comma|all> <workroot> <out.jsonl> [NT]

ecutwfc: G5 over all elements of the phases in one call (one cutoff across a reaction; F0/F0b: 80 Ry), MP smearing 0.02 Ry, k spacing 0.20 A^-1,
spin-polarised if a Cr/Mn/Fe/Co/Ni is present. 'H2' (molecule in its 12 A box) uses one k-point.
N(E_F): Gaussian 0.1 eV on the SCF eigenvalues, states/eV/atom. Restart-safe on (id, xc).
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


def main(src, ids, root, out, nt="8"):
    recs = {}
    for ln in Path(src).read_text().splitlines():
        r = json.loads(ln)
        if "structure" in r:
            recs[r["id"]] = r
    want = list(recs) if ids == "all" else [i for i in ids.split(",") if i]
    seen = set()
    if Path(out).exists():
        seen = {(json.loads(l)["id"], json.loads(l)["xc"]) for l in Path(out).read_text().splitlines() if l.strip()}
    env = dict(os.environ, OMP_NUM_THREADS=nt, RTSC_STH="0")
    # G5 (FUNNEL section 8): one cutoff across the reaction = the G5 cutoff over every element of the phases run together
    ecut = qe.ecut_rule({e for k in want if k in recs for e in qe.from_pmg(recs[k]["structure"])["species"]})
    with open(out, "a") as fo:
        for k in want:
            if k not in recs:
                continue
            st = qe.from_pmg(recs[k]["structure"])
            hyd = "H" in st["species"]
            for xc in (("pbe", "pbesol") if (hyd or os.environ.get("T2_BOTH") == "1") else ("pbe",)):
                if (k, xc) in seen:
                    continue
                W = Path(root) / f"{k}_{xc}"
                W.mkdir(parents=True, exist_ok=True)
                mag = bool(set(st["species"]) & qe.MAGNETIC)
                kk = [1, 1, 1] if k == "H2" else qe.kgrid(st["lattice"], 0.20)
                txt = qe.pw_input(st, "scf", "t2", kk, ecut, pseudo=PSEUDO, nspin=2 if mag else 1,
                                  xc="pbesol" if xc == "pbesol" else None, conv="1.0d-8")
                (W / "scf.in").write_text(txt)
                t0 = time.time()
                with open(W / "scf.out", "w") as f:
                    rc = subprocess.run(["mpirun", "--bind-to", "none", "-np", "1", "-x", "OMP_NUM_THREADS",
                                         f"{BIN}/pw.x", "-in", "scf.in"], cwd=W, stdout=f, stderr=subprocess.STDOUT,
                                        env=env, timeout=3600).returncode
                dt = time.time() - t0
                rec = dict(id=k, xc=xc, nat=len(st["species"]), k=kk, ecut=ecut, spin=mag, wall_s=round(dt, 2), rc=rc)
                try:
                    x = qe.read_xml(W / "t2.save/data-file-schema.xml")
                    rec.update(energy=x["energy"], e_atom=x["energy"] / len(st["species"]), ef=x["ef"],
                               N_EF=qe.dos_ef(x, len(st["species"])), mag=x["mag"],
                               formula=recs[k].get("formula"))
                except Exception as ex:
                    rec["status"] = f"fail:scf:{ex}"[:200]
                shutil.rmtree(W / "t2.save", ignore_errors=True)
                for f in W.glob("t2.wfc*"):
                    f.unlink()
                fo.write(json.dumps(rec) + "\n"); fo.flush()
                print(k, xc, rec.get("e_atom"), rec.get("N_EF"), rec.get("mag"), rec["wall_s"], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
