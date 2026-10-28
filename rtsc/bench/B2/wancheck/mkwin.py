# Wannier90 .win for the B2 check: python mkwin.py "name|num_wann|dis_froz_max|proj;proj" (NK env = coarse k)
import itertools, sys
name, nw, froz, projs = sys.argv[1].split("|")
import numpy as np
a = 12.569178 * 0.529177210903  # Angstrom
A = 0.5 * np.array([[-1, 0, 1], [0, 1, 1], [-1, 1, 0]])  # QE ibrav=2, units of a
d = 0.258656
atoms = [("Ir", (0, 0, 0)), ("Mg", (.25, .25, .25)), ("Mg", (-.25, -.25, -.25))] + [
    ("H", tuple(s * d * np.eye(3)[i])) for i in range(3) for s in (1, -1)]
n = int(__import__("os").environ.get("NK", 4))
frac = lambda kc: A @ np.array(kc, float)  # k in 2pi/a cartesian -> fractional of b_i
path = [("G", (0, 0, 0)), ("X", (0, 1, 0)), ("W", (.5, 1, 0)), ("L", (.5, .5, .5)), ("G", (0, 0, 0)), ("K", (.75, .75, 0))]
out = [f"num_wann = {nw}", "num_bands = 28", "exclude_bands = 1-12", "iprint = 2",
       f"dis_froz_max = {froz}", "dis_num_iter = 2000", "num_iter = 500", "write_hr = .true.",
       "bands_plot = .true.", "bands_num_points = 40",
       "begin projections", *[" " + p for p in projs.split(";")], "end projections",
       "begin unit_cell_cart", "ang"] + [" %.10f %.10f %.10f" % tuple(a * v) for v in A] + [
       "end unit_cell_cart", "begin atoms_cart", "ang"] + [
       " %s %.10f %.10f %.10f" % (el, *(a * np.array(p))) for el, p in atoms] + [
       "end atoms_cart", "begin kpoint_path"]
for (l1, k1), (l2, k2) in zip(path, path[1:]):
    out.append(" %s %.6f %.6f %.6f  %s %.6f %.6f %.6f" % (l1, *frac(k1), l2, *frac(k2)))
out += ["end kpoint_path", f"mp_grid = {n} {n} {n}", "begin kpoints"]
out += [" %.10f %.10f %.10f" % (i / n, j / n, k / n) for i, j, k in itertools.product(range(n), repeat=3)]
out += ["end kpoints"]
sys.stdout.write("\n".join(out) + "\n")
