"""M3-3 Phase 0 tool, part 1: plane-wave Hamiltonian from a QE save (norm-conserving, no spin), wavefunction I/O.

numpy by default; set M33_XP=cupy to run the array work on a GPU (cupy mirrors the numpy API used here).
Units: Rydberg atomic units (QE). Everything is in periodic-part form u_k(G) on the dense FFT box (NC: one grid).
"""
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.io import FortranFile
from scipy.special import spherical_jn

if os.environ.get("M33_XP") == "cupy":
    import cupy as xp           # noqa: F401
else:
    xp = np
RYEV = 13.605693122994


def to_np(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


# ----------------------------------------------------------------------------------------------------------- QE files
def read_xml(save):
    r = ET.parse(Path(save) / "data-file-schema.xml").getroot()
    o = r.find("output")
    ast = o.find("atomic_structure")
    alat = float(ast.get("alat"))
    at = np.array([[float(x) for x in ast.find(f"cell/a{i}").text.split()] for i in (1, 2, 3)])   # bohr, rows
    sp, tau = [], []
    for a in ast.find("atomic_positions").findall("atom"):
        sp.append(a.get("name")); tau.append([float(x) for x in a.text.split()])
    bs = o.find("basis_set")
    fft = bs.find("fft_grid")
    nr = [int(fft.get(k)) for k in ("nr1", "nr2", "nr3")]
    ecutwfc = float(bs.find("ecutwfc").text) * 2      # Ha -> Ry
    ecutrho = float(bs.find("ecutrho").text) * 2
    b = o.find("band_structure")
    ef = float(b.find("fermi_energy").text) * 2       # Ha -> Ry
    ks = b.findall("ks_energies")
    xk = np.array([[float(x) for x in k.find("k_point").text.split()] for k in ks])   # cart, 2pi/alat
    eig = np.array([[float(x) * 2 for x in k.find("eigenvalues").text.split()] for k in ks])
    species = {s.get("name"): s.find("pseudo_file").text for s in o.find("atomic_species").findall("species")}
    sm = r.find("input/bands/smearing")
    return dict(alat=alat, at=at, species=sp, tau=np.array(tau), nr=nr, ecutwfc=ecutwfc, ecutrho=ecutrho, ef=ef,
                xk=xk, eig=eig, upf=species, degauss=float(sm.get("degauss")) * 2 if sm is not None else None)


def wfc_path(save, ik):
    """1-based k index -> wfc file (.dat or .hdf5)."""
    for ext in (".dat", ".hdf5"):
        q = Path(save) / f"wfc{ik}{ext}"
        if q.exists():
            return q
    raise FileNotFoundError(f"{save}/wfc{ik}.*")


def read_wfc(path):
    if str(path).endswith(".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            mill = np.array(f["MillerIndices"]).astype(int)
            e = np.array(f["evc"])
            ev = e[:, 0::2] + 1j * e[:, 1::2]
            return dict(xk=np.array(f.attrs["xk"]), mill=mill, evc=ev)
    f = FortranFile(path, "r")
    rec = f.read_record("<i4,(3,)<f8,<i4,<i4,<f8")
    ngw, igwx, npol, nbnd = f.read_ints("<i4")
    f.read_reals("<f8")
    mill = f.read_ints("<i4").reshape(igwx, 3)
    ev = np.array([f.read_reals("<c16") for _ in range(nbnd)])
    f.close()
    return dict(xk=np.array(rec[0][1]), mill=mill, evc=ev)        # evc: (nbnd, npw)


def read_upf(path):
    t = Path(path).read_text()

    def arr(tag):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", t, re.S)
        return np.array([float(x) for x in m.group(1).split()])
    r, rab = arr("PP_R"), arr("PP_RAB")
    betas = []
    for m in re.finditer(r"<PP_BETA\.(\d+)([^>]*)>(.*?)</PP_BETA\.\d+>", t, re.S):
        att = m.group(2)
        l = int(re.search(r'angular_momentum="\s*(\d+)', att).group(1))
        ic = int(re.search(r'cutoff_radius_index="\s*(\d+)', att).group(1))
        betas.append(dict(l=l, ic=ic, rb=np.array([float(x) for x in m.group(3).split()])))
    nb = len(betas)
    D = arr("PP_DIJ").reshape(nb, nb)
    zv = float(re.search(r'z_valence="\s*([-0-9.Ee+]+)', t).group(1))
    nlcc = re.search(r'core_correction="\s*([TtFf.a-z]+)', t).group(1).lower().startswith(("t", ".t"))
    return dict(r=r, rab=rab, betas=betas, D=D, zv=zv, nlcc=nlcc)


def read_filplot(path, nr):
    """pp.x filplot (plot_io format): header, then nr1x*nr2x*nr3x values (Fortran order)."""
    lines = Path(path).read_text().splitlines()
    h = lines[1].split()
    nrx = [int(x) for x in h[:3]]
    nat, ntyp = int(h[6]), int(h[7])
    ibrav = int(lines[2].split()[0])
    i = 3 + (3 if ibrav == 0 else 0) + 1 + ntyp + nat
    vals = np.array([float(x) for ln in lines[i:] for x in ln.split()])
    v = vals[:nrx[0] * nrx[1] * nrx[2]].reshape(nrx[2], nrx[1], nrx[0]).transpose(2, 1, 0)
    return v[:nr[0], :nr[1], :nr[2]]


def read_dvscf(path, nr, mode=0, nmodes=None):
    """ph.x fildvscf: direct-access records of nr1x*nr2x*nr3x complex (nspin 1), one per mode (pattern basis)."""
    n = nr[0] * nr[1] * nr[2]
    a = np.fromfile(path, dtype=np.complex128)
    return a[mode * n:(mode + 1) * n].reshape(nr[2], nr[1], nr[0]).transpose(2, 1, 0)


# ------------------------------------------------------------------------------------------------------- Hamiltonian
def real_ylm(l, v):
    """Real spherical harmonics (orthonormal), v: (n, 3) unit vectors; returns (2l+1, n)."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    c = 1 / math.sqrt(4 * math.pi)
    if l == 0:
        return np.array([np.full_like(x, c)])
    if l == 1:
        f = math.sqrt(3) * c
        return np.array([f * z, -f * x, -f * y])
    if l == 2:
        return np.array([math.sqrt(5) * c * 0.5 * (3 * z * z - 1), -math.sqrt(15) * c * x * z,
                         -math.sqrt(15) * c * y * z, math.sqrt(15) * c * 0.5 * (x * x - y * y),
                         math.sqrt(15) * c * x * y])
    if l == 3:
        return np.array([math.sqrt(7) * c * 0.5 * z * (5 * z * z - 3),
                         math.sqrt(21 / 2) * c * 0.5 * x * (5 * z * z - 1), math.sqrt(21 / 2) * c * 0.5 * y * (5 * z * z - 1),
                         math.sqrt(105) * c * 0.5 * z * (x * x - y * y), math.sqrt(105) * c * x * y * z,
                         math.sqrt(35 / 2) * c * 0.5 * x * (x * x - 3 * y * y),
                         math.sqrt(35 / 2) * c * 0.5 * y * (3 * x * x - y * y)])
    raise ValueError(l)


class System:
    """Cell, local potential and pseudopotentials from a QE run directory (prefix.save + pp.x filplot)."""

    def __init__(self, save, upfdir, vfile):
        self.x = read_xml(save)
        x = self.x
        self.alat, self.nr = x["alat"], x["nr"]
        self.at = x["at"]                                   # bohr
        self.omega = abs(np.linalg.det(self.at))
        self.bg = 2 * math.pi * np.linalg.inv(self.at).T    # bohr^-1 rows
        self.tpiba = 2 * math.pi / self.alat
        self.veff = read_filplot(vfile, self.nr)            # Ry
        self.vr = xp.asarray(self.veff)
        self.upf = {e: read_upf(Path(upfdir) / f) for e, f in x["upf"].items()}
        self.tau = x["tau"]                                 # bohr (cart)
        self.ef = x["ef"]

    # ---- G-vector bases
    def basis(self, kcrys, mill=None):
        """Basis at k (crystal coords): mill from a wfc file or generated (|k+G|^2 <= ecutwfc)."""
        if mill is None:
            n = [int(math.ceil(math.sqrt(self.x["ecutwfc"]) / np.linalg.norm(self.bg[i]) * 1.0)) + 2 for i in range(3)]
            g = np.stack(np.meshgrid(*[np.arange(-m, m + 1) for m in n], indexing="ij"), -1).reshape(-1, 3)
            kpg = (g + kcrys) @ self.bg
            mill = g[np.sum(kpg ** 2, 1) <= self.x["ecutwfc"]]
        return Basis(self, np.asarray(kcrys, float), np.asarray(mill))


class Basis:
    def __init__(self, sysm, kcrys, mill):
        self.s, self.k, self.mill = sysm, kcrys, mill
        kpg = (mill + kcrys) @ sysm.bg                     # bohr^-1
        self.kin = xp.asarray(np.sum(kpg ** 2, 1))          # Ry
        nr = sysm.nr
        self.idx = xp.asarray(np.ravel_multi_index(tuple((mill % np.array(nr)).T), nr))
        # nonlocal projectors
        q = np.linalg.norm(kpg, axis=1)
        qh = kpg / np.maximum(q, 1e-12)[:, None]
        cols, Ds = [], []
        pref = 4 * math.pi / math.sqrt(sysm.omega)
        for ia, e in enumerate(sysm.x["species"]):
            u = sysm.upf[e]
            if not u["betas"]:
                continue
            sk = np.exp(-1j * kpg @ sysm.tau[ia])
            blocks = []
            for b in u["betas"]:
                ic = min(len(u["r"]), b["ic"] + 1 + (b["ic"] % 2))   # odd point count covering the cutoff
                r, rab, rb = u["r"][:ic], u["rab"][:ic], b["rb"][:ic]
                f = simpson_vec(rb * r * spherical_jn(b["l"], np.outer(q, r)), rab)
                ylm = real_ylm(b["l"], qh)
                blocks.append([pref * f * ylm[m] * (-1j) ** b["l"] * sk for m in range(2 * b["l"] + 1)])
            ls = [b["l"] for b in u["betas"]]
            off = []
            for bl in blocks:
                off.append(len(cols))
                cols.extend(bl)
            Ds.append((off, ls, u["D"]))
        self.vkb = xp.asarray(np.array(cols).T) if cols else None   # (npw, nproj)
        nproj = len(cols)
        Dm = np.zeros((nproj, nproj))
        for off, ls, D in Ds:
            for i in range(len(ls)):
                for j in range(len(ls)):
                    if ls[i] == ls[j] and D[i, j] != 0:
                        for m in range(2 * ls[i] + 1):
                            Dm[off[i] + m, off[j] + m] = D[i, j]
        self.D = xp.asarray(Dm)
        self.npw = len(mill)

    def to_box(self, C, shift=None):
        """C: (nb, npw) coefficients -> (nb, nr1, nr2, nr3) real-space periodic part. shift: integer G0 such that the
        represented function is e^{-i G0.r} u (i.e. coefficients placed at mill - G0)."""
        nr = self.s.nr
        idx = self.idx if shift is None else xp.asarray(
            np.ravel_multi_index(tuple(((self.mill - np.asarray(shift)) % np.array(nr)).T), nr))
        nb = C.shape[0]
        box = xp.zeros((nb, nr[0] * nr[1] * nr[2]), complex)
        box[:, idx] = C
        return xp.fft.ifftn(box.reshape(nb, *nr), axes=(1, 2, 3)) * (nr[0] * nr[1] * nr[2])

    def from_box(self, F, shift=None):
        nr = self.s.nr
        idx = self.idx if shift is None else xp.asarray(
            np.ravel_multi_index(tuple(((self.mill - np.asarray(shift)) % np.array(nr)).T), nr))
        nb = F.shape[0]
        g = xp.fft.fftn(F, axes=(1, 2, 3)).reshape(nb, -1) / (nr[0] * nr[1] * nr[2])
        return g[:, idx]

    def apply(self, C):
        """H C for C (nb, npw)."""
        out = C * self.kin[None, :]
        out = out + self.from_box(self.to_box(C) * self.s.vr[None])
        if self.vkb is not None:
            p = C @ self.vkb.conj()            # (nb, nproj) = <beta|psi>
            out = out + (p @ self.D.T) @ self.vkb.T
        return out


def simpson_vec(F, rab):
    """QE's simpson along the last axis (1-based i = 2, 4, ..., mesh-1: f(i-1) + 4 f(i) + f(i+1), times rab, / 3)."""
    mesh = F.shape[-1]
    w = np.zeros(mesh)
    for i in range(1, mesh - 1, 2):
        w[i - 1] += 1; w[i] += 4; w[i + 1] += 1
    return (F * (rab * w)[None, :]).sum(-1) / 3.0


# --------------------------------------------------------------------------------------------------- eigen solver
def refine(B, C, eps_tol=1e-10, maxit=60, nbuf=4, seed=0):
    """Refine eigenpairs of B's H starting from C (nb, npw) (e.g. pw.x's): LOBPCG-style block iteration on nb + nbuf
    vectors (nbuf random buffer vectors keep the top requested band away from the block edge). Returns (eig Ry (nb,),
    C (nb, npw) orthonormal, iterations); converged when the max residual of the nb requested bands <= eps_tol Ry.
    The degenerate-cluster alignment to the input gauge is done by the caller (align)."""
    nb = C.shape[0]
    rng = np.random.default_rng(seed)
    Rn = xp.asarray(rng.standard_normal((nbuf, C.shape[1])) + 1j * rng.standard_normal((nbuf, C.shape[1])))
    X = xp.concatenate([C, Rn / (1.0 + B.kin[None, :])], 0)
    Pm = None
    for it in range(maxit):
        HX = B.apply(X)
        e, v = eigh_gen(X.conj() @ HX.T, X.conj() @ X.T)
        X = v.T @ X; HX = v.T @ HX
        R = HX - e[:, None] * X
        rn = to_np(xp.sqrt(xp.sum(xp.abs(R) ** 2, 1)))
        if rn[:nb].max() <= eps_tol:
            return to_np(e[:nb]), X[:nb], it
        W = R / (1.0 + B.kin[None, :] / (1.0 + xp.abs(e)[:, None] + 1.0))
        W = W - (W @ X.conj().T) @ X                 # orthogonalise to the block, then normalise each direction
        W = W / xp.maximum(xp.sqrt(xp.sum(xp.abs(W) ** 2, 1)), 1e-300)[:, None]
        if Pm is not None:
            Pm = Pm - (Pm @ X.conj().T) @ X
            Pm = Pm / xp.maximum(xp.sqrt(xp.sum(xp.abs(Pm) ** 2, 1)), 1e-300)[:, None]
        Z = xp.concatenate([X, W] + ([Pm] if Pm is not None else []), 0)
        HZ = B.apply(Z)
        S = Z.conj() @ Z.T
        # drop near-dependent directions
        sv, U = xp.linalg.eigh((S + S.conj().T) / 2)
        keep = sv > 1e-10 * float(to_np(sv).max())
        T = U[:, keep] / xp.sqrt(sv[keep])[None, :]
        e2, v2 = xp.linalg.eigh(T.conj().T @ ((Z.conj() @ HZ.T + (Z.conj() @ HZ.T).conj().T) / 2) @ T)
        V = T @ v2[:, :X.shape[0]]
        Xn = V.T @ Z
        Pm = Xn - (Xn @ X.conj().T) @ X          # the new directions beyond the old block (LOBPCG "P")
        X = Xn
    HX = B.apply(X)
    e, v = eigh_gen(X.conj() @ HX.T, X.conj() @ X.T)
    X = v.T @ X
    return to_np(e[:nb]), X[:nb], maxit


def eigh_gen(H, S):
    H = (H + H.conj().T) / 2
    S = (S + S.conj().T) / 2
    L = xp.linalg.cholesky(S)
    Li = xp.linalg.inv(L)
    e, v = xp.linalg.eigh(Li @ H @ Li.conj().T)
    return e, Li.conj().T @ v


def align(C_new, C_old, e, tol=1e-5):
    """Within clusters of degenerate eigenvalues (|de| < tol Ry), rotate C_new to the closest orthonormal basis to C_old
    (Lowdin): keeps the input gauge (wannier90's U applies) while the vectors stay exact eigenvectors of the refined H."""
    nb = len(e)
    out = C_new.copy()
    i = 0
    while i < nb:
        j = i + 1
        while j < nb and e[j] - e[j - 1] < tol:
            j += 1
        if j - i > 0:
            O = C_new[i:j].conj() @ C_old[i:j].T           # (d, d)
            u, s, vh = xp.linalg.svd(O)
            Rm = u @ vh                                     # closest unitary
            out[i:j] = Rm.T @ C_new[i:j]
            # sign/phase per vector for d = 1 is included (u vh is a phase)
        i = j
    return out
