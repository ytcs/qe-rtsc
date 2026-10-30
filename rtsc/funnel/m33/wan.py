"""M3-3 Phase 0 tool, part 3: the W manifold (wannier90 on the coarse full-grid nscf; all bands from the lowest).

    wan.py <nscf dir (prefix.save inside)> <prefix> <n coarse> <E_F eV> <E_froz - E_F eV> <out dir> [poor]

N_W = max over k of the number of bands below E_froz, + 15 % (at least + 4). Outer window: the smallest E_out with at least
N_W + 2 bands at every k. Initial projections: SCDM (pw2wannier90 scdm_proj, erfc, mu = E_froz, sigma = 1.5 eV).
'poor' = projection-only (dis_num_iter = 0, num_iter = 0), the return's control (c).
Env: W90 (wannier90.x), P2W (pw2wannier90.x), MPIRUN prefix for pw2wannier90 (optional).
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pwtool as P  # noqa: E402


def main(nscf, prefix, n, ef_ev, dfroz, out, poor=""):
    nscf, out = Path(nscf).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    n = int(n); ef = float(ef_ev); efroz = ef + float(dfroz)
    x = P.read_xml(nscf / f"{prefix}.save")
    eig = x["eig"] * P.RYEV                                  # (nk, nb) eV
    nk, nbnd = eig.shape
    nfz = int((eig < efroz).sum(1).max())
    nw = nfz + max(4, int(math.ceil(0.15 * nfz)))
    srt = np.sort(eig, axis=1)
    if srt.shape[1] < nw + 2:
        raise SystemExit(f"nbnd {nbnd} < N_W + 2 = {nw + 2}")
    eout = float(srt[:, nw + 1].max()) + 1e-3                # every k has >= N_W + 2 bands below E_out
    at = x["at"] * P.BOHR if hasattr(P, "BOHR") else x["at"] * 0.529177210903
    kc = [xk @ (x["at"] / x["alat"]).T for xk in x["xk"]]
    tau = x["tau"] @ np.linalg.inv(x["at"])
    seed = "w"
    win = [f"num_bands = {nbnd}", f"num_wann = {nw}", f"dis_win_max = {eout:.6f}", f"dis_froz_max = {efroz:.6f}",
           "auto_projections = .true.", "write_u_matrices = .true.", "num_print_cycles = 50",
           f"dis_num_iter = {0 if poor else 2000}", f"num_iter = {0 if poor else 1000}", "dis_conv_tol = 1e-10",
           "conv_tol = 1e-10", "conv_window = 5", f"mp_grid = {n} {n} {n}",
           "begin unit_cell_cart", "bohr"] + [" ".join(f"{v:.10f}" for v in a) for a in x["at"]] + \
          ["end unit_cell_cart", "begin atoms_frac"] + \
          [f"{s} " + " ".join(f"{v:.10f}" for v in t) for s, t in zip(x["species"], tau)] + \
          ["end atoms_frac", "begin kpoints"] + [" ".join(f"{v:.10f}" for v in k) for k in kc] + ["end kpoints"]
    (out / f"{seed}.win").write_text("\n".join(win) + "\n")
    w90 = os.environ.get("W90", "wannier90.x"); p2w = os.environ.get("P2W", "pw2wannier90.x")
    mpi = os.environ.get("MPIRUN", "").split()
    subprocess.run([w90, "-pp", seed], cwd=out, check=True)
    (out / "p2w.in").write_text(f"&inputpp\n  outdir = '{nscf}'\n  prefix = '{prefix}'\n  seedname = '{seed}'\n"
                                f"  write_amn = .true.\n  write_mmn = .true.\n  write_unk = .false.\n"
                                f"  scdm_proj = .true.\n  scdm_entanglement = 'erfc'\n  scdm_mu = {efroz:.6f}\n"
                                f"  scdm_sigma = 1.5\n/\n")
    with open(out / "p2w.out", "w") as fo:
        subprocess.run(mpi + [p2w, "-in", "p2w.in"], cwd=out, check=True, stdout=fo, stderr=subprocess.STDOUT)
    subprocess.run([w90, seed], cwd=out, check=True)
    wout = (out / f"{seed}.wout").read_text()
    sp = [float(l.split()[-1]) for l in wout.splitlines() if "WF centre and spread" in l][-nw:]
    info = dict(nw=nw, n_frozen_max=nfz, efroz_eV=efroz, eout_eV=eout, nbnd=nbnd, poor=bool(poor),
                max_spread=max(sp) if sp else None, omega_I=None)
    for l in wout.splitlines():
        if "Omega I" in l and "=" in l:
            info["omega_I"] = float(l.split("=")[-1])
    (out / "info.txt").write_text(repr(info) + "\n")
    print(info)


if __name__ == "__main__":
    main(*sys.argv[1:])
