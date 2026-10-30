"""S1 (FUNNEL.md section 8): T3a dynamic-stability rule, from the ph.x dyn files of the T3a run.

A candidate with any DFPT mode < -30 cm^-1 (Gamma acoustic excluded) does not pass T3a; it gets one re-relax along the
unstable mode and a rerun; still unstable -> hold `unstable_dfpt`, disposition `sscha` if the mode is H-dominated
(H participation > 0.5), else `reject`.

Realisation (FUNNEL-LOG F0c pre-registration): frequencies and mass-weighted eigenvectors are recomputed from the
dynamical matrices in <prefix>.dyn<i> (first q of each star). Gamma acoustic = the 3 Gamma modes with the largest
projection on rigid translations. The re-relax displacement is Re(e^{i q.R} e_a)/sqrt(m_a) of the most negative mode,
in the diagonal supercell commensurate with q (the primitive cell at Gamma), scaled to a 0.10 A max displacement.
"""
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

AMU_RY = 911.44424310865645   # QE amu_ry
RY_TO_CM = 109737.31570111268
BOHR = 0.529177210903
LINE = -30.0


def read_dyn(path):
    """-> dict(alat bohr, at (3x3, alat units, rows), species list, mass list (amu), tau (nat x 3, alat), q (cart 2pi/alat),
    C (3nat x 3nat complex, Ry/bohr^2)) for the first q of the file."""
    lines = Path(path).read_text().splitlines()
    hdr = lines[2].split()
    ntyp, nat, ibrav = int(hdr[0]), int(hdr[1]), int(hdr[2])
    alat = float(hdr[3])
    i = 3
    at = np.eye(3)
    if ibrav == 0:
        if "Basis" in lines[i]:
            i += 1
        at = np.array([[float(x) for x in lines[i + j].split()] for j in range(3)])
        i += 3
    names, masses = {}, {}
    for _ in range(ntyp):
        p = lines[i].replace("'", " ").split()
        names[int(p[0])] = p[1]
        masses[int(p[0])] = float(p[2]) / AMU_RY
        i += 1
    sp, mass, tau = [], [], []
    for _ in range(nat):
        p = lines[i].split()
        sp.append(names[int(p[1])]); mass.append(masses[int(p[1])])
        tau.append([float(x) for x in p[2:5]])
        i += 1
    while "q = (" not in lines[i]:
        i += 1
    q = np.array([float(x) for x in lines[i].split("(")[1].split(")")[0].split()])
    i += 1
    C = np.zeros((3 * nat, 3 * nat), complex)
    nread = 0
    while nread < nat * nat:
        p = lines[i].split()
        if len(p) == 2 and all(x.isdigit() for x in p):
            a, b = int(p[0]) - 1, int(p[1]) - 1
            for r in range(3):
                v = [float(x) for x in lines[i + 1 + r].split()]
                C[3 * a + r, 3 * b:3 * b + 3] = [complex(v[2 * c], v[2 * c + 1]) for c in range(3)]
            nread += 1
            i += 4
        else:
            i += 1
    return dict(alat=alat, at=at, species=sp, mass=np.array(mass), tau=np.array(tau), q=q, C=C)


def modes(d):
    """Frequencies (cm^-1, negative = imaginary) and mass-weighted unit eigenvectors (columns)."""
    m = np.repeat(d["mass"], 3) * AMU_RY
    D = d["C"] / np.sqrt(np.outer(m, m))
    D = (D + D.conj().T) / 2
    w2, e = np.linalg.eigh(D)
    w = np.sign(w2) * np.sqrt(np.abs(w2)) * RY_TO_CM
    return w, e


def analyze(rundir, prefix="x", line=LINE):
    """S1 verdict for a finished ph.x run in rundir. Returns dict(unstable, omega_min_cm (Gamma acoustic excluded),
    worst=dict(file, q, mode, omega, H_participation), gamma_acoustic_cm, per_q=[...])."""
    rundir = Path(rundir)
    nq = int((rundir / f"{prefix}.dyn0").read_text().split("\n")[1].split()[0])
    per_q, worst, gac = [], None, None
    for i in range(1, nq + 1):
        d = read_dyn(rundir / f"{prefix}.dyn{i}")
        w, e = modes(d)
        keep = np.ones(len(w), bool)
        if np.linalg.norm(d["q"]) < 1e-8:
            sm = np.repeat(np.sqrt(d["mass"]), 3)
            T = np.zeros((len(w), 3))
            for a in range(3):
                t = np.zeros(len(w)); t[a::3] = sm[a::3]; T[:, a] = t / np.linalg.norm(t)
            proj = np.sum(np.abs(T.T @ e) ** 2, axis=0)
            ac = np.argsort(-proj)[:3]
            keep[ac] = False
            gac = sorted(float(x) for x in w[ac])
        isH = np.repeat(np.array([s == "H" for s in d["species"]]), 3)
        for j in np.where(keep)[0]:
            hp = float(np.sum(np.abs(e[isH, j]) ** 2))
            if worst is None or w[j] < worst["omega"]:
                worst = dict(file=i, q=d["q"].tolist(), mode=int(j), omega=float(w[j]), H_participation=hp)
        if keep.any():
            per_q.append(dict(file=i, q=d["q"].tolist(), omega_min=float(w[keep].min()),
                              n_below=int((w[keep] < line).sum())))
    return dict(unstable=bool(worst["omega"] < line), omega_min_cm=worst["omega"], worst=worst, gamma_acoustic_cm=gac,
                per_q=per_q, line_cm=line)


def displaced(rundir, st, worst, prefix="x", umax=0.10):
    """Child structure for the S1 re-relax: st (canonical plain dict, the scf cell; atom order = the dyn files') displaced
    along mode `worst` in the diagonal supercell commensurate with its q, max displacement umax A."""
    d = read_dyn(Path(rundir) / f"{prefix}.dyn{worst['file']}")
    w, e = modes(d)
    ev = e[:, worst["mode"]]
    ev = ev * np.exp(-1j * np.angle(ev[np.argmax(np.abs(ev))]))   # global phase: largest component real
    lat = np.array(st["lattice"], float)
    at_alat = lat / (d["alat"] * BOHR)
    qf = at_alat @ np.array(worst["q"])        # q . a_i (units of 2 pi): crystal coordinates of q
    n = [Fraction(float(x)).limit_denominator(12).denominator for x in qf]
    frac0 = np.array(st["frac"], float)
    sp, fr, u = [], [], []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                R = np.array([i, j, k], float)
                ph = np.exp(2j * math.pi * qf @ R)
                for a, s in enumerate(st["species"]):
                    sp.append(s)
                    fr.append((frac0[a] + R) / np.array(n, float))
                    u.append(np.real(ph * ev[3 * a:3 * a + 3]) / math.sqrt(d["mass"][a]))
    u = np.array(u)
    if np.abs(u).max() == 0:
        u = np.array([np.imag(x) for x in u])
    u *= umax / np.linalg.norm(u, axis=1).max()
    slat = lat * np.array(n, float)[:, None]
    fr = np.array(fr) + u @ np.linalg.inv(slat)
    return dict(lattice=slat.tolist(), species=sp, frac=fr.tolist()), n
