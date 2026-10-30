"""M3-3 Phase 0 tool, part 2: standard metallic chi0 and the occupation-factorised coarse-k response (return M3/3 D1-D5 with
referee A's F2), on the same Hamiltonian (pwtool), for one q and one local perturbation dV (periodic part on the dense box).

Conventions (M3-3-LOG pre-registration): MP order-1 occupations at E_F (scf) with width sigma; the full N^3 Gamma-centred grid,
no symmetry; densities are periodic parts, normalised as 2 (spin) x [2 (TR) x first-order pairs] / N^3 (the Omega factor is
common to every density and omitted). Every quantity entering r is computed with the SAME H, so control (a) (N_c = N_f) is
an algebraic identity up to the CG / eigen tolerances.
"""
import math
import time
from pathlib import Path

import numpy as np

import pwtool as P
from pwtool import xp, to_np

SQPI = math.sqrt(math.pi)
TIE = 1e-4      # Ry, width of the smooth pair-split weight in std_response


# ------------------------------------------------------------------------------------------------------- occupations
def f_mp(e, ef, sig):
    from scipy.special import erfc
    x = (ef - np.asarray(e)) / sig
    return 0.5 * erfc(-x) + x * np.exp(-x * x) / (2 * SQPI)


def df_mp(e, ef, sig):
    x = (ef - np.asarray(e)) / sig
    return -np.exp(-x * x) * (1.5 - x * x) / SQPI / sig


def lindhard(e1, e2, ef, sig, tol=1e-9):
    """L[n, m] = (f(e1_n) - f(e2_m)) / (e1_n - e2_m), f' on the diagonal limit."""
    f1, f2 = f_mp(e1, ef, sig), f_mp(e2, ef, sig)
    d = e1[:, None] - e2[None, :]
    num = f1[:, None] - f2[None, :]
    deg = np.abs(d) < tol
    out = np.where(deg, 0.0, num / np.where(deg, 1.0, d))
    if deg.any():
        dm = 0.5 * (df_mp(e1, ef, sig)[:, None] + df_mp(e2, ef, sig)[None, :])
        out = np.where(deg, dm, out)
    return out


def cinf_step(t):
    """C-infinity step: 0 for t <= 0, 1 for t >= 1."""
    t = np.asarray(t, float)
    a = np.where(t > 0, np.exp(-1.0 / np.where(t > 0, t, 1.0)), 0.0)
    b = np.where(t < 1, np.exp(-1.0 / np.where(t < 1, 1 - t, 1.0)), 0.0)
    return a / (a + b)


class Switch:
    """s (1 below E_c, 0 above E_froz) and the clamp g (= x below E_g, saturating below E_g + delta < E_c). Ry."""

    def __init__(self, ef, sig, efroz, c=5.0, delta=0.25 / P.RYEV, gap=0.5 / P.RYEV):
        self.eg = ef + c * sig
        self.ec = self.eg + gap
        self.efroz = efroz
        self.delta = delta
        if not self.ec < self.efroz:
            raise ValueError(f"E_c {self.ec * P.RYEV:.3f} eV >= E_froz {self.efroz * P.RYEV:.3f} eV (H1/H2 not satisfiable)")
        y = np.linspace(0, delta, 20001)
        self._gy = y
        self._gv = np.concatenate([[0.0], np.cumsum(0.5 * ((1 - cinf_step(y[1:] / delta)) + (1 - cinf_step(y[:-1] / delta)))
                                                    * np.diff(y))])

    def s(self, x):
        return 1.0 - cinf_step((np.asarray(x) - self.ec) / (self.efroz - self.ec))

    def g(self, x):
        x = np.asarray(x, float)
        u = np.clip(x - self.eg, 0, self.delta)
        return np.where(x <= self.eg, x, self.eg + np.interp(u, self._gy, self._gv))


# -------------------------------------------------------------------------------------------------------- grid states
def kkey(kc, n):
    return tuple(int(round(x * n)) % n for x in kc)


class GridStates:
    """Refined eigenstates of every k of a full-grid nscf (pw.x save), cached on disk as .npz."""

    def __init__(self, S, save, n, cache, nb=None, tol=1e-10):
        self.S, self.save, self.n, self.tol = S, Path(save), n, tol
        self.cache = Path(cache); self.cache.mkdir(parents=True, exist_ok=True)
        x = P.read_xml(save)
        self.kc = [xk @ (S.at / S.alat).T for xk in x["xk"]]
        self.eig_pw = x["eig"]
        self.index = {kkey(k, n): i for i, k in enumerate(self.kc)}
        if len(self.index) != n ** 3:
            raise ValueError(f"{save}: {len(self.index)} distinct k, expected {n ** 3}")
        self.nb = nb or x["eig"].shape[1]
        self.iters = []
        self.mem = {} if n <= 6 else None          # in-memory cache for coarse grids

    def get(self, key):
        """-> (Basis, eig (nb,), C (nb, npw) xp array) for grid key (i, j, l)."""
        if self.mem is not None and key in self.mem:
            return self.mem[key]
        out = self._get(key)
        if self.mem is not None:
            self.mem[key] = out
        return out

    def _get(self, key):
        i = self.index[key]
        f = self.cache / f"k{i + 1}.npz"
        w = P.read_wfc(P.wfc_path(self.save, i + 1))
        B = self.S.basis(self.kc[i], w["mill"])
        if f.exists():
            d = np.load(f)
            return B, d["e"], xp.asarray(d["C"])
        C0 = xp.asarray(w["evc"][:self.nb])
        e, C, it = P.refine(B, C0, self.tol)
        C = P.align(C, C0, e)
        self.iters.append(it)
        np.savez(f, e=e, C=to_np(C))
        return B, e, C


def shift_of(kc, q, kc2):
    """Integer G0 with kc + q = kc2 + G0."""
    g = np.asarray(kc) + np.asarray(q) - np.asarray(kc2)
    r = np.round(g)
    assert np.abs(g - r).max() < 1e-6
    return r.astype(int)


# ---------------------------------------------------------------------------------------------------------- CG solver
def cg(apply_A, b, prec, tol=1e-10, maxit=2000, proj=None):
    """Block CG (independent columns as rows) for Hermitian positive definite A: A x = b, relative residual <= tol.
    proj(v): projector applied to keep iterates in a subspace (Q space for the standard Sternheimer)."""
    x = xp.zeros_like(b)
    r = b.copy()
    if proj is not None:
        r = proj(r)
    bn = xp.sqrt(xp.sum(xp.abs(r) ** 2, 1))
    bn = xp.where(bn == 0, 1.0, bn)
    z = prec(r)
    if proj is not None:
        z = proj(z)
    p = z.copy()
    rz = xp.sum(r.conj() * z, 1).real
    for it in range(maxit):
        Ap = apply_A(p)
        if proj is not None:
            Ap = proj(Ap)
        a = rz / xp.sum(p.conj() * Ap, 1).real
        x = x + a[:, None] * p
        r = r - a[:, None] * Ap
        rel = float(to_np(xp.max(xp.sqrt(xp.sum(xp.abs(r) ** 2, 1)) / bn)))
        if rel <= tol:
            return x, it + 1, rel
        z = prec(r)
        if proj is not None:
            z = proj(z)
        rz2 = xp.sum(r.conj() * z, 1).real
        p = z + (rz2 / rz)[:, None] * p
        rz = rz2
    return x, maxit, rel


# ---------------------------------------------------------------------------------------------------- std response
def std_response(states, q, dV, ef, sig, fmin=1e-13, tol=1e-10, log=print):
    """Delta n_std on the box: explicit pairs over the computed bands (W_nm = L_nm theta(e_m - e_n)) plus the projected
    Sternheimer for the rest space at k+q; times 2 (TR) x 2 (spin) / N^3."""
    n = states.n
    S = states.S
    dVx = xp.asarray(dV)
    acc = xp.zeros(tuple(S.nr), complex)
    t0, its = time.time(), []
    keys = sorted(states.index)
    for key in keys:
        kc = np.array(key) / n
        key2 = kkey(kc + q, n)
        B1, e1, C1 = states.get(key)
        B2, e2, C2 = states.get(key2)
        G0 = shift_of(kc, q, np.array(key2) / n)
        f1 = f_mp(e1, ef, sig)
        occ = np.where(np.abs(f1) > fmin)[0]
        if len(occ) == 0:
            continue
        if e2.max() < ef + 8 * sig:
            raise ValueError(f"computed set too small at {key2}: top {e2.max() * P.RYEV:.3f} eV < E_F + 8 sigma")
        u1 = B1.to_box(C1[occ])                          # (no, box)
        dvu = B2.from_box(dVx[None] * u1, shift=G0)      # (no, npw2): dV u_n in the k+q basis
        M = C2.conj() @ dvu.T                            # (nb2, no): M_mn
        L = lindhard(e1[occ], e2, ef, sig)               # (no, nb2)
        # split weight h(e_m - e_n) with h(d) + h(-d) = 1, smooth over TIE so that pairs degenerate by time reversal
        # (k + q = -k, e.g. q = 1/2) are split consistently; rest-space pairs (d > 3 sigma) get h = 1
        th = 0.5 * (1.0 + np.tanh((e2[None, :] - e1[occ][:, None]) / TIE))
        Wt = xp.asarray(L * th) * M.T                    # (no, nb2): W_nm M_mn
        u2 = B2.to_box(C2, shift=G0)                     # (nb2, box)
        pair = xp.einsum("nm,mxyz->nxyz", Wt, u2)
        # rest: y_n = -(Q (H - e_n) Q)^-1 Q dV u_n, contributes f_n u_n* y_n
        def proj(v):
            return v - (v @ C2.conj().T) @ C2
        en = xp.asarray(e1[occ])
        y, it, rel = cg(lambda v: B2.apply(v) - en[:, None] * v, -dvu,
                        lambda v: v / (1.0 + B2.kin[None, :]), tol=tol, proj=proj)
        its.append(it)
        rest = xp.asarray(f1[occ])[:, None, None, None] * B2.to_box(y, shift=G0)
        acc += xp.sum(u1.conj() * (pair + rest), 0)
    acc *= 4.0 / n ** 3
    log(f"std N={n}: {len(keys)} k, CG its mean {np.mean(its):.0f} max {max(its)}, {time.time() - t0:.0f} s")
    return to_np(acc)


# ------------------------------------------------------------------------------------------------------- Wannier W
def read_umat(path):
    t = Path(path).read_text().split("\n")
    nk, a, b = [int(x) for x in t[1].split()]
    vals, kps, i = [], [], 2
    while len(kps) < nk:
        while not t[i].strip():
            i += 1
        kps.append([float(x) for x in t[i].split()]); i += 1
        m = []
        while len(m) < a * b:
            m.append([float(x) for x in t[i].split()]); i += 1
        arr = np.array(m)
        vals.append((arr[:, 0] + 1j * arr[:, 1]).reshape(a, b).T)   # header (nk, a, b); the file index running over b is fastest
    return np.array(kps), np.array(vals)


class WannierW:
    """w_j(k) = sum_n psi_n(k) V_nj(k), V = U_opt U (wannier90 u_dis.mat, u.mat), on the coarse grid states."""

    def __init__(self, states, seed_dir, seed):
        self.st = states
        kd, Ud = read_umat(Path(seed_dir) / f"{seed}_u_dis.mat") if (Path(seed_dir) / f"{seed}_u_dis.mat").exists() \
            else (None, None)
        ku, U = read_umat(Path(seed_dir) / f"{seed}_u.mat")
        self.nw = U.shape[1]
        self.V = {}
        for i, k in enumerate(ku):
            key = kkey(k, states.n)
            self.V[key] = (Ud[i] @ U[i]) if Ud is not None else U[i]

    def get(self, key):
        B, e, C = self.st.get(key)
        V = xp.asarray(self.V[key])                       # (nbands, nw)
        nbw = V.shape[0]
        w = V.T @ C[:nbw]                                 # (nw, npw)
        return B, w


def interp_matrix(nc, nf):
    """Band-limited Fourier interpolation I (nf^3 x nc^3) from G_c to G_f (crystal grids, Nyquist split)."""
    def one():
        if nc % 2:
            R = np.arange(-(nc - 1) // 2, (nc - 1) // 2 + 1); w = np.ones(len(R))
        else:
            R = np.arange(-nc // 2, nc // 2 + 1); w = np.ones(len(R)); w[0] = w[-1] = 0.5
        kf = np.arange(nf) / nf
        kc = np.arange(nc) / nc
        x = kf[:, None] - kc[None, :]
        return np.real(np.einsum("r,fcr->fc", w, np.exp(2j * np.pi * x[:, :, None] * R[None, None, :]))) / nc
    I1 = one()
    return np.einsum("ad,be,cf->abcdef", I1, I1, I1).reshape(nf ** 3, nc ** 3)


def fact_response(W, q, dV, ef, sig, efroz, nf, tol=1e-10, log=print, return_info=False):
    """Delta n_fact(N_c, N_f): D3 rest solves on G_c, D4 window weights on G_f from interpolated H_W and M, D5 pull-back."""
    st, S = W.st, W.st.S
    nc = st.n
    dVx = xp.asarray(dV)
    keys = sorted(st.index)
    sw = Switch(ef, sig, efroz)
    t0 = time.time()
    # pass 1: H_W(k_c), M(k_c) = w(k_c+q)^+ dV w(k_c)
    HW, M, emin = {}, {}, 1e9
    for key in keys:
        kc = np.array(key) / nc
        key2 = kkey(kc + q, nc)
        B1, w1 = W.get(key)
        B2, w2 = W.get(key2)
        G0 = shift_of(kc, q, np.array(key2) / nc)
        H = to_np(w1.conj() @ B1.apply(w1).T)
        HW[key] = (H + H.conj().T) / 2
        dvw = B2.from_box(dVx[None] * B1.to_box(w1), shift=G0)
        M[key] = to_np(w2.conj() @ dvw.T)                 # (i at k+q, j at k)
        emin = min(emin, float(np.linalg.eigvalsh(HW[key])[0]))
    alpha = sw.ec - emin + 5.0 / P.RYEV
    # fine grid: F(k), D(k)
    I = interp_matrix(nc, nf)
    Hc = np.array([HW[k] for k in keys]); Mc = np.array([M[k] for k in keys])
    keys_f = [(a, b, c) for a in range(nf) for b in range(nf) for c in range(nf)]
    order_c = {k: i for i, k in enumerate(keys)}
    # keys sorted lexicographically == C order of (a, b, c) used by interp_matrix
    assert keys == [(a, b, c) for a in range(nc) for b in range(nc) for c in range(nc)]
    Hf = np.einsum("fc,cij->fij", I, Hc)
    Mf = np.einsum("fc,cij->fij", I, Mc)
    nw = Hc.shape[1]
    Ff = np.zeros_like(Hf); Df = np.zeros_like(Hf)
    eigs = [np.linalg.eigh((h + h.conj().T) / 2) for h in Hf]
    for i, kf in enumerate(keys_f):
        k2 = kkey(np.array(kf) / nf + q, nf)
        j = (k2[0] * nf + k2[1]) * nf + k2[2]
        e1, U1 = eigs[i]
        e2, U2 = eigs[j]
        Ff[i] = (U1 * f_mp(e1, ef, sig)[None, :]) @ U1.conj().T
        L = lindhard(e1, e2, ef, sig)                    # (n at k, m at k+q)
        f1, f2 = f_mp(e1, ef, sig), f_mp(e2, ef, sig)
        s1, s2 = sw.s(e1), sw.s(e2)
        g1, g2 = sw.g(e1), sw.g(e2)
        Rp = f1[:, None] * (1 - s2[None, :]) / (g1[:, None] - e2[None, :] - alpha * s2[None, :])
        RpT = f2[None, :] * (1 - s1[:, None]) / (g2[None, :] - e1[:, None] - alpha * s1[:, None])
        Sw = L - Rp - RpT
        Mt = U2.conj().T @ Mf[i] @ U1                    # (m, n)
        Df[i] = U2 @ (Sw.T * Mt) @ U1.conj().T
    Ft = np.einsum("fc,fij->cij", I, Ff)
    Dt = np.einsum("fc,fij->cij", I, Df)
    t1 = time.time()
    # pass 2: rest solves and density
    acc = xp.zeros(tuple(S.nr), complex)
    its = []
    for key in keys:
        kc = np.array(key) / nc
        key2 = kkey(kc + q, nc)
        B1, w1 = W.get(key)
        B2, w2 = W.get(key2)
        G0 = shift_of(kc, q, np.array(key2) / nc)
        H1 = HW[key]
        eg, Ut = np.linalg.eigh(H1)
        gn = sw.g(eg)
        psi = xp.asarray(Ut.T) @ w1                      # (nw, npw1): window eigenvectors psi~_n
        b = B2.from_box(dVx[None] * B1.to_box(psi), shift=G0)    # dV psi~_n in the k+q basis
        e2w, U2w = np.linalg.eigh(HW[key2])
        Smat2 = xp.asarray((U2w * sw.s(e2w)[None, :]) @ U2w.conj().T)   # s(H_W(k+q)) in the Wannier basis

        def s_apply(v):
            return ((v @ w2.conj().T) @ Smat2.T) @ w2
        rhs = -(b - s_apply(b))
        gx = xp.asarray(gn)
        z, it, rel = cg(lambda v: B2.apply(v) + alpha * s_apply(v) - gx[:, None] * v, rhs,
                        lambda v: v / (1.0 + B2.kin[None, :]), tol=tol)
        its.append(it)
        X = xp.asarray(Ut.conj()) @ z                    # X_j = sum_n z_n Ut_jn*
        Xb = B2.to_box(X, shift=G0)                      # (nw, box) at k+q
        wb = B1.to_box(w1)                               # (nw, box) at k
        w2b = B2.to_box(w2, shift=G0)
        ic = order_c[key]
        Fk = xp.asarray(Ft[ic]); Dk = xp.asarray(Dt[ic])
        # rest: 2 sum_ij F_ij X_i w_j* ; window: sum_ij D_ij w_i(k+q) w_j*(k)
        restp = xp.einsum("ij,ixyz->jxyz", Fk, Xb)
        winp = xp.einsum("ij,ixyz->jxyz", Dk, w2b)
        acc += xp.sum(wb.conj() * (2.0 * restp + winp), 0)
    acc *= 2.0 / nf ** 3
    info = dict(nw=nw, alpha_eV=alpha * P.RYEV, emin_eV=emin * P.RYEV, cg_mean=float(np.mean(its)), cg_max=int(max(its)),
                t_pass1=round(t1 - t0, 1), t_total=round(time.time() - t0, 1), columns=nc ** 3 * nw)
    log(f"fact Nc={nc} Nf={nf} NW={nw} E_froz-E_F={(efroz - ef) * P.RYEV:.2f} eV: CG its mean {info['cg_mean']:.0f}, "
        f"{info['t_total']} s")
    return (to_np(acc), info) if return_info else to_np(acc)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
