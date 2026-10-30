"""P1 NLCC gate (FUNNEL-LOG F0c pre-registration; PSEUDO-SURVEY section 7(2) sub-test).

    PSET=spms PSEUDO_DIR=<spms dir> python3 p1gate.py prep <structure json (plain dict)> <dir>
    python3 p1gate.py compare <engine run dir> <vanilla run dir> <out.json>

prep: scf.in (G5 SPMS cutoff, T3a scf k) and ph.in (T3a q grid) in <dir>. compare: frequencies of every dyn file present
(q1-q2), max |d omega| against max(0.5 %, 1 cm^-1), Gamma acoustic excluded (s1 projection rule).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import qe  # noqa: E402
import s1  # noqa: E402


def prep(stj, d):
    st = qe.canonical(json.loads(Path(stj).read_text()))
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    g = qe.t3a_grids(st)
    ecut = qe.ecut_rule(st["species"])
    import os
    (d / "scf.in").write_text(qe.pw_input(st, "scf", "x", g["k"], ecut, pseudo=os.environ["PSEUDO_DIR"]))
    (d / "ph.in").write_text(qe.ph_input("x", st["species"], g["nq"]))
    print("ecut", ecut, "k", g["k"], "nq", g["nq"])


def freqs(d):
    out = {}
    for f in sorted(Path(d).glob("x.dyn[0-9]*")):
        if f.name == "x.dyn0":
            continue
        dd = s1.read_dyn(f)
        w, e = s1.modes(dd)
        keep = np.ones(len(w), bool)
        if np.linalg.norm(dd["q"]) < 1e-8:
            sm = np.repeat(np.sqrt(dd["mass"]), 3)
            T = np.zeros((len(w), 3))
            for a in range(3):
                t = np.zeros(len(w)); t[a::3] = sm[a::3]; T[:, a] = t / np.linalg.norm(t)
            keep[np.argsort(-np.sum(np.abs(T.T @ e) ** 2, axis=0))[:3]] = False
        out[f.name] = (w, keep)
    return out


def compare(a, b, out):
    fa, fb = freqs(a), freqs(b)
    rows, ok = [], True
    for k in sorted(set(fa) & set(fb)):
        (wa, ka), (wb, _) = fa[k], fb[k]
        dw = np.abs(wa - wb)
        tol = np.maximum(0.005 * np.abs(wb), 1.0)
        worst = float(np.max(dw[ka] - tol[ka])) if ka.any() else -1.0
        ok &= worst <= 0
        rows.append(dict(dyn=k, max_dw_cm=float(dw[ka].max()) if ka.any() else None, worst_excess_cm=worst,
                         gamma_acoustic_dw=[float(x) for x in dw[~ka]]))
    r = dict(pass_gate=bool(ok and rows), n_dyn=len(rows), rows=rows)
    Path(out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r))


if __name__ == "__main__":
    {"prep": prep, "compare": compare}[sys.argv[1]](*sys.argv[2:])
