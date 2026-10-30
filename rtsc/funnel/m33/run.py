"""M3-3 Phase 0 driver (M3-3-LOG pre-registration).

    run.py <sysdir> <stage: A | B> <N_c> <N_f> <q1 q2 q3 (crystal)> <W dirs, comma: name=dir> [out.json]

<sysdir> holds: scf/x.save (scf), veff.dat (pp.x plot_num 1), dvscf (ph.x fildvscf at q, pattern 1 used), nf/x.save (full N_f^3
nscf), nc/x.save (full N_c^3 nscf, Wannier bands), upf/ (UPF dir). Stage A: delta-eps at the Fermi surface on G_f \\ G_c from
H_W(k_c) = V^+ diag(eps_pw) V interpolated with I. Stage B: std(N_f), std(N_c), the exactness gate fact(N_c, N_c) (first W),
fact(N_c, N_f) for every W; r and r_stdc.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pwtool as P  # noqa: E402
import resp as R  # noqa: E402


def stage_a(S, sysd, nc, nf, wdirs, ef, sig):
    xf = P.read_xml(sysd / "nf" / "x.save")
    xc = P.read_xml(sysd / "nc" / "x.save")
    kcf = [k @ (S.at / S.alat).T for k in xf["xk"]]
    kcc = [k @ (S.at / S.alat).T for k in xc["xk"]]
    idx_c = {R.kkey(k, nc): i for i, k in enumerate(kcc)}
    I = R.interp_matrix(nc, nf)
    keys_c = [(a, b, c) for a in range(nc) for b in range(nc) for c in range(nc)]
    out = {}
    for name, d in wdirs.items():
        kd, Ud = R.read_umat(Path(d) / "w_u_dis.mat")
        ku, U = R.read_umat(Path(d) / "w_u.mat")
        Vk = {R.kkey(k, nc): Ud[i] @ U[i] for i, k in enumerate(ku)}
        Hc = []
        for key in keys_c:
            e = xc["eig"][idx_c[key]][:Vk[key].shape[0]]
            V = Vk[key]
            Hc.append(V.conj().T @ (e[:, None] * V))
        Hc = np.array(Hc)
        Hf = np.einsum("fc,cij->fij", I, Hc)
        dev, devmax = [], 0.0
        win = 3 * sig
        for i, k in enumerate(kcf):
            key = R.kkey(k, nf)
            if all((x * nc) % nf == 0 for x in key):          # on the coarse sublattice: exact by construction
                continue
            j = (key[0] * nf + key[1]) * nf + key[2]
            ew = np.linalg.eigvalsh((Hf[j] + Hf[j].conj().T) / 2)
            ed = xf["eig"][i]
            for x in ed[np.abs(ed - ef) <= win]:
                dev.append(np.min(np.abs(ew - x)))
            for x in ew[np.abs(ew - ef) <= win]:
                dev.append(np.min(np.abs(ed - x)))
        dev = np.array(dev) * P.RYEV * 1000
        rms = float(np.sqrt(np.mean(dev ** 2))) if len(dev) else None
        out[name] = dict(nw=int(Hc.shape[1]), n_levels=len(dev), rms_meV=rms, max_meV=float(dev.max()) if len(dev) else None,
                         rms_over_sigma=rms / (sig * P.RYEV * 1000) if rms is not None else None)
        print("A", name, out[name], flush=True)
    return out


def main(sysd, stage, nc, nf, q1, q2, q3, wspec, outp=None):
    t0 = time.time()
    sysd = Path(sysd).resolve()
    nc, nf = int(nc), int(nf)
    q = np.array([float(q1), float(q2), float(q3)])
    wdirs = dict(x.split("=") for x in wspec.split(","))
    S = P.System(sysd / "scf" / "x.save", sysd / "upf", sysd / "veff.dat")
    sig = S.x["degauss"]
    ef = S.ef
    res = dict(stage=stage, nc=nc, nf=nf, q=q.tolist(), sigma_Ry=sig, ef_eV=ef * P.RYEV, nr=S.nr)
    if stage == "A":
        res["A"] = stage_a(S, sysd, nc, nf, wdirs, ef, sig)
    else:
        dV = P.read_dvscf(sysd / "dvscf", S.nr, mode=0)
        stf = R.GridStates(S, sysd / "nf" / "x.save", nf, sysd / "cache_nf")
        stc = R.GridStates(S, sysd / "nc" / "x.save", nc, sysd / "cache_nc")
        d_f = R.std_response(stf, q, dV, ef, sig)
        np.save(sysd / "dn_std_f.npy", d_f)
        d_c = R.std_response(stc, q, dV, ef, sig)
        res["r_stdc"] = R.rel(d_c, d_f)
        print("r_stdc", res["r_stdc"], flush=True)
        res["B"] = {}
        first = True
        for name, d in wdirs.items():
            info_d = eval((Path(d) / "info.txt").read_text())
            W = R.WannierW(stc, d, "w")
            efroz = info_d["efroz_eV"] / P.RYEV
            try:
                if first:
                    fe, ie = R.fact_response(W, q, dV, ef, sig, efroz, nc, return_info=True)
                    res["exactness"] = dict(W=name, r=R.rel(fe, d_c), info=ie)
                    print("EXACTNESS", res["exactness"], flush=True)
                    first = False
                ff, inf = R.fact_response(W, q, dV, ef, sig, efroz, nf, return_info=True)
                res["B"][name] = dict(r=R.rel(ff, d_f), info=inf, w=info_d)
            except ValueError as ex:
                res["B"][name] = dict(error=str(ex), w=info_d)
            print("B", name, res["B"][name], flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    if outp:
        Path(outp).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])
