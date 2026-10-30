"""Column ratio R (M3-3-LOG): standard = nksq(q) x n_occ (ph.x at the production k; n_occ = mean count of bands with
eps < E_F + 5 sigma on the full N_f grid); factorised = N_c^3 (no coarse symmetry) x N_W.

    colratio.py <sysdir> [nksq override]
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pwtool as P  # noqa: E402


def main(d, nksq=None):
    d = Path(d)
    cfg = eval((d / "cfg.txt").read_text())
    xs = P.read_xml(d / "scf" / "x.save")
    xf = P.read_xml(d / "nf" / "x.save")
    sig = xs["degauss"]
    nocc = float(np.mean((xf["eig"] < xs["ef"] + 5 * sig).sum(1)))
    if nksq is None:
        # ph.x prints the k and k+q list (q != 0 in every Phase-0 case): nksq = half of it
        nksq = int(re.findall(r"number of k points=\s*(\d+)", (d / "ph.out").read_text())[-1]) // 2
    else:
        nksq = int(nksq)
    std = nksq * nocc
    out = dict(nksq=nksq, n_occ=nocc, std_columns=std)
    for w in ("W2", "W4", "W6", "Wpoor"):
        f = d / w / "info.txt"
        if f.exists():
            nw = eval(f.read_text())["nw"]
            out[w] = dict(nw=nw, fact_columns=cfg["nc"] ** 3 * nw, R=std / (cfg["nc"] ** 3 * nw))
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
