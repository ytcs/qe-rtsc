"""QE input generation and output parsing for the funnel tiers (rules: FUNNEL-LOG F0 pre-registration).

Stdlib + numpy only (runs under the system python3 on the pod); structures are plain dicts
{lattice: 3x3 A, species: [..], frac: [[..]..]}.
"""
import math
import re
import xml.etree.ElementTree as ET

import numpy as np

HA = 27.211386245988
RY = HA / 2
BOHR = 0.529177210903
MASS = dict(H=1.008, Li=6.94, Be=9.012, B=10.811, C=12.011, N=14.007, O=15.999, F=18.998, Na=22.990, Mg=24.305,
            Al=26.982, Si=28.086, P=30.974, S=32.06, Cl=35.45, K=39.098, Ca=40.078, Sc=44.956, Ti=47.867, V=50.942,
            Cr=51.996, Mn=54.938, Fe=55.845, Co=58.933, Ni=58.693, Cu=63.546, Zn=65.38, Ga=69.723, Ge=72.63,
            As=74.922, Se=78.971, Br=79.904, Rb=85.468, Sr=87.62, Y=88.906, Zr=91.224, Nb=92.906, Mo=95.95,
            Tc=98.0, Ru=101.07, Rh=102.91, Pd=106.42, Ag=107.87, Cd=112.41, In=114.82, Sn=118.71, Sb=121.76,
            Te=127.60, I=126.90, Cs=132.91, Ba=137.33, La=138.91, Hf=178.49, Ta=180.95, W=183.84, Re=186.21,
            Os=190.23, Ir=192.217, Pt=195.08, Au=196.97, Hg=200.59, Tl=204.38, Pb=207.2, Bi=208.98)
D3 = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn".split())
MAGNETIC = set("Cr Mn Fe Co Ni".split())
FIRST_ROW = set("H B C N O F".split())
TM = set("Sc Ti V Cr Mn Fe Co Ni Cu Y Zr Nb Mo Tc Ru Rh Pd Ag La Hf Ta W Re Os Ir Pt Au".split())
SPD = set("Zn Cd Hg".split())
S_ONLY = set("H Li Na K Rb Cs Be".split())
P_ONLY = set("O F Cl Br I".split())
SD = set("Ca Sr Ba".split())


def from_pmg(d):
    """pymatgen Structure.as_dict() -> plain dict."""
    lat = d["lattice"]["matrix"]
    sp = [s["species"][0]["element"] for s in d["sites"]]
    fr = [s["abc"] for s in d["sites"]]
    return dict(lattice=lat, species=sp, frac=fr)


def recip_norms(lat):
    a = np.array(lat, float)
    b = 2 * np.pi * np.linalg.inv(a).T
    return np.linalg.norm(b, axis=1)


def kgrid(lat, dk):
    return [max(1, math.ceil(x / dk)) for x in recip_norms(lat)]


def ecut_rule(species):
    s = set(species)
    return 80 if (s & FIRST_ROW or s & D3) else 60


def species_order(st):
    out = []
    for e in st["species"]:
        if e not in out:
            out.append(e)
    return out


def pw_input(st, calc, prefix, k, ecut, pseudo="@PSEUDO@", smear=0.02, extra_sys="", extra_ctl="",
             nbnd=None, press=None, nspin=1, xc=None, kblock=None, conv="1.0d-10"):
    sp = species_order(st)
    sys_lines = [f"ibrav = 0", f"nat = {len(st['species'])}", f"ntyp = {len(sp)}", f"ecutwfc = {ecut}",
                 "occupations = 'smearing'", "smearing = 'mp'", f"degauss = {smear}"]
    if nbnd:
        sys_lines.append(f"nbnd = {nbnd}")
    if xc:
        sys_lines.append(f"input_dft = '{xc}'")
    if nspin == 2:
        sys_lines.append("nspin = 2")
        for i, e in enumerate(sp):
            if e in MAGNETIC:
                sys_lines.append(f"starting_magnetization({i + 1}) = 0.5")
    if extra_sys:
        sys_lines.append(extra_sys)
    ctl = [f"calculation = '{calc}'", f"prefix = '{prefix}'", f"pseudo_dir = '{pseudo}'", "outdir = './'"]
    if calc in ("relax", "vc-relax"):
        ctl += ["etot_conv_thr = 1.0d-5", "forc_conv_thr = 1.0d-4", "nstep = 80"]
    if extra_ctl:
        ctl.append(extra_ctl)
    txt = "&control\n" + "".join(f"    {x}\n" for x in ctl) + "/\n&system\n" + "".join(f"    {x}\n" for x in sys_lines)
    txt += f"/\n&electrons\n    conv_thr = {conv}\n    mixing_beta = 0.4\n/\n"
    if calc in ("relax", "vc-relax"):
        txt += "&ions\n/\n"
    if calc == "vc-relax":
        txt += "&cell\n    press_conv_thr = 0.5\n" + (f"    press = {press}\n" if press else "") + "/\n"
    txt += "ATOMIC_SPECIES\n" + "".join(f" {e} {MASS[e]} {e}_ONCV_PBE-1.2.upf\n" for e in sp)
    txt += "CELL_PARAMETERS angstrom\n" + "".join(" " + " ".join(f"{x:.10f}" for x in v) + "\n" for v in st["lattice"])
    txt += "ATOMIC_POSITIONS crystal\n" + "".join(
        f" {e} {f[0]:.10f} {f[1]:.10f} {f[2]:.10f}\n" for e, f in zip(st["species"], st["frac"]))
    txt += kblock if kblock else f"K_POINTS automatic\n{k[0]} {k[1]} {k[2]} 0 0 0\n"
    return txt


def kfull(n):
    pts = [(i / n, j / n, l / n) for i in range(n) for j in range(n) for l in range(n)]
    return f"K_POINTS crystal\n{len(pts)}\n" + "".join(f" {a:.10f} {b:.10f} {c:.10f} {1 / len(pts):.6e}\n" for a, b, c in pts)


def read_xml(path):
    """Final structure, energy (eV), Fermi (eV), eigenvalues (eV, [nk][nb]), k weights, magnetization."""
    r = ET.parse(path).getroot()
    out = r.find("output")
    ast = out.find("atomic_structure")
    alat_vecs = [[float(x) * BOHR for x in ast.find(f"cell/a{i}").text.split()] for i in (1, 2, 3)]
    sp, cart = [], []
    for a in ast.find("atomic_positions").findall("atom"):
        sp.append(a.get("name"))
        cart.append([float(x) * BOHR for x in a.text.split()])
    lat = np.array(alat_vecs)
    frac = (np.array(cart) @ np.linalg.inv(lat)).tolist()
    e = float(out.find("total_energy/etot").text) * HA
    bs = out.find("band_structure")
    ef_node = bs.find("fermi_energy")
    ef = float(ef_node.text) * HA if ef_node is not None else None
    lsda = bs.find("lsda").text.strip() == "true"
    eig, wk = [], []
    for ks in bs.findall("ks_energies"):
        wk.append(float(ks.find("k_point").get("weight")))
        eig.append([float(x) * HA for x in ks.find("eigenvalues").text.split()])
    mag = out.find("magnetization/total")
    m = float(mag.text) if mag is not None else 0.0
    return dict(structure=dict(lattice=lat.tolist(), species=sp, frac=frac), energy=e, ef=ef, eig=eig, wk=wk,
                lsda=lsda, mag=m)


def dos_ef(x, nat, sigma=0.1):
    """N(E_F) in states/eV/atom (both spins), Gaussian broadening sigma eV on the SCF eigenvalues."""
    w = np.array(x["wk"]) / sum(x["wk"])
    tot = 0.0
    for wi, e in zip(w, x["eig"]):
        d = (np.array(e) - x["ef"]) / sigma
        tot += wi * np.sum(np.exp(-d * d)) / (sigma * math.sqrt(math.pi))
    spin = 1.0 if x["lsda"] else 2.0
    return spin * tot / nat


def projections(species):
    out = []
    for e in species_order(dict(species=species)):
        if e in S_ONLY:
            orb = ["s"]
        elif e in P_ONLY:
            orb = ["p"]
        elif e in SPD:
            orb = ["s", "p", "d"]
        elif e in TM or e in SD:
            orb = ["s", "d"]
        else:
            orb = ["s", "p"]
        out.append((e, orb))
    return out


def n_wann(species):
    cnt = {"s": 1, "p": 3, "d": 5}
    pr = dict(projections(species))
    return sum(sum(cnt[o] for o in pr[e]) for e in species)


def n_deep(x, below=12.0):
    """F0 rule (superseded by G1, kept for the record): bands lying entirely below E_F - below eV at every k."""
    e = np.array(x["eig"])
    n = 0
    while n < e.shape[1] and e[:, n].max() < x["ef"] - below:
        n += 1
    return n


# Electrons in the shells the projection set covers (valence s/p for main group, d + s for TM, ...):
# main-group group valence; transition metals d + s; P_ONLY elements without their (unprojected) s pair.
VAL_E = dict(H=1, Li=1, Na=1, K=1, Rb=1, Cs=1, Be=2, Mg=2, Ca=2, Sr=2, Ba=2, B=3, Al=3, Ga=3, In=3, Tl=3,
             C=4, Si=4, Ge=4, Sn=4, Pb=4, N=5, P=5, As=5, Sb=5, Bi=5, S=6, Se=6, Te=6, O=4, F=5, Cl=5, Br=5, I=5,
             Sc=3, Ti=4, V=5, Cr=6, Mn=7, Fe=8, Co=9, Ni=10, Cu=11, Zn=12, Y=3, Zr=4, Nb=5, Mo=6, Tc=7, Ru=8, Rh=9,
             Pd=10, Ag=11, Cd=12, La=3, Hf=4, Ta=5, W=6, Re=7, Os=8, Ir=9, Pt=10, Au=11, Hg=12)


def z_valence(e, pseudo):
    t = open(f"{pseudo}/{e}_ONCV_PBE-1.2.upf").read(20000)
    return float(re.search(r'z_valence\s*=\s*"\s*([-+0-9.eEdD]+)', t).group(1).replace("d", "e").replace("D", "e"))


def n_shell(species, pseudo):
    """G1 (FUNNEL.md section 7) realised by shell counting: bands of shells NOT covered by the projection set.
    SG15 carries no PP_PSWFC, so the projected weight cannot be computed with projwfc.x; a semicore shell of an
    element is by construction not in its (valence-shell) projection set, and a valence shell always is. Hence an
    element's valence s band is never excluded when it carries an s projection (H3S: S 3s kept)."""
    n = 0
    for e in species:
        z = z_valence(e, pseudo)
        sc = z - VAL_E[e]
        if sc < -1e-6 or abs(sc - round(sc)) > 1e-6 or int(round(sc)) % 2:
            raise ValueError(f"semicore count for {e}: z {z}, valence {VAL_E[e]}")
        n += int(round(sc)) // 2
    return n


def n_semicore(x, species, pseudo):
    """G1 band exclusion: (n, info). n = shell count; requires a real gap below the first kept band at every k
    (max_k e[n-1] < min_k e[n]); raises ValueError otherwise (recorded as fail:semicore)."""
    e = np.array(x["eig"])
    n = n_shell(species, pseudo)
    info = dict(n_shell=n, n_deep_F0=n_deep(x))
    if n:
        top, bot = float(e[:, n - 1].max()), float(e[:, n].min())
        info.update(excl_top_eV=round(top - x["ef"], 3), kept_bottom_eV=round(bot - x["ef"], 3))
        if not top < bot:
            raise ValueError(f"no gap between excluded band {n} and band {n + 1}: {top:.3f} >= {bot:.3f} eV")
    return n, info


def frozen_window(x, nsemi, nw):
    e = np.array(x["eig"])[:, nsemi:]
    for w in (2.0, 1.5, 1.0, 0.6):
        if int((e <= x["ef"] + w).sum(axis=1).max()) <= nw:
            return w
    return None


def a2f_metrics(path, muc=0.10):
    """lambda, omega_log, omega_2 (meV), Tc McMillan-Dynes and Allen-Dynes (f1 f2) in K, from prefix.a2f_proj
    (column 2 = a2F for the first phonon smearing), with EPW's own formulas (supercond.f90 estimate_tc_gap)."""
    w, a = [], []
    for ln in open(path):
        p = ln.split()
        try:
            w.append(float(p[0])); a.append(float(p[1]))
        except (ValueError, IndexError):
            continue
    w, a = np.array(w), np.array(a)
    m = w > 0
    w, a = w[m], a[m]
    dw = w[1] - w[0]
    lam = 2 * np.sum(a / w) * dw
    if lam <= 0:
        return dict(lam=float(lam), wlog=0.0, w2=0.0, tc_mcm=0.0, tc_ad=0.0)
    wlog = math.exp(2 * np.sum(a * np.log(w) / w) * dw / lam)
    w2 = math.sqrt(2 * np.sum(a * w) * dw / lam)
    kel = 0.08617333262  # meV per K
    den = lam - muc * (1 + 0.62 * lam)
    expo = math.exp(-1.04 * (1 + lam) / den) if den > 0 else 0.0
    tc = wlog / 1.2 * expo / kel
    l1 = 2.46 * (1 + 3.8 * muc)
    l2 = 1.82 * (1 + 6.3 * muc) * (w2 / wlog)
    f1 = (1 + (lam / l1) ** 1.5) ** (1 / 3)
    f2 = 1 + (w2 / wlog - 1) * lam ** 2 / (lam ** 2 + l2 ** 2)
    return dict(lam=float(lam), wlog=wlog, w2=w2, tc_mcm=tc, tc_ad=f1 * f2 * tc)


def ph_input(prefix, species, nq):
    sp = species_order(dict(species=species))
    masses = "".join(f"  amass({i + 1}) = {MASS[e]}\n" for i, e in enumerate(sp))
    return (f"&inputph\n  prefix = '{prefix}'\n  fildyn = '{prefix}.dyn'\n{masses}  outdir = './'\n  ldisp = .true.\n"
            f"  fildvscf = 'dvscf'\n  nq1 = {nq}, nq2 = {nq}, nq3 = {nq}\n  tr2_ph = 1.0d-14\n  reduce_io = .true.\n/\n")


def epw_input(prefix, species, nk, nq, nkf, nqf, nsemi, nw, froz, ef, scdm=False, muc=0.10):
    sp = species_order(dict(species=species))
    lines = [f"prefix = '{prefix}'"] + [f"amass({i + 1}) = {MASS[e]}" for i, e in enumerate(sp)]
    lines += ["outdir = './'", "ep_coupling = .true.", "elph = .true.", "epbwrite = .true.", "epwwrite = .true.",
              "etf_mem = 1", f"nbndsub = {nw}"]
    if nsemi:
        lines.append(f"bands_skipped = 'exclude_bands = 1-{nsemi}'")
    lines += ["wannierize = .true.", "num_iter = 500"]
    if scdm:
        lines += ["auto_projections = .true.", "scdm_proj = .true.", "scdm_entanglement = 'erfc'",
                  f"scdm_mu = {ef:.4f}", "scdm_sigma = 2.0"]
    else:
        lines.append(f"dis_froz_max = {ef + froz:.4f}")
        i = 0
        for e, orb in projections(species):
            i += 1
            lines.append(f"proj({i}) = '{e}:{';'.join(orb)}'")
        lines.append("wdata(1) = 'dis_num_iter = 1000'")
    lines += ["eps_acoustic = 2.0", "ephwrite = .true.", "fsthick = 0.4", "degaussw = 0.10", "degaussq = 0.5",
              "nqstep = 500", "eliashberg = .true.", "wscut = 1.0", "temps = 1.0", "nstemp = 1",
              f"muc = {muc}", "dvscf_dir = './save'", f"nk1 = {nk}, nk2 = {nk}, nk3 = {nk}",
              f"nq1 = {nq}, nq2 = {nq}, nq3 = {nq}", "mp_mesh_k = .true.",
              f"nkf1 = {nkf}, nkf2 = {nkf}, nkf3 = {nkf}", f"nqf1 = {nqf}, nqf2 = {nqf}, nqf3 = {nqf}"]
    return "&inputepw\n" + "".join(f"  {x}\n" for x in lines) + "/\n"


def t3a_grids(st):
    nat = len(st["species"])
    k = kgrid(st["lattice"], 0.20 if nat <= 4 else 0.27)
    nq = 4 if nat == 1 else (3 if nat <= 4 else 2)
    nk = 2 * nq   # G2 (FUNNEL.md section 7): Wannier/EPW coarse k = 2 q for every cell size (F0: k = q for 3-4 atoms)
    nkf = max(kgrid(st["lattice"], 0.2))
    nkf += nkf % 2
    return dict(k=k, nq=nq, nk=nk, nkf=nkf, nqf=nkf // 2)
