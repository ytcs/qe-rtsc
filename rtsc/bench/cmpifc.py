#!/usr/bin/env python3
"""cmpifc.py SUBDIR REFDIR BINDIR -- dyn from subgrid IFCs (q2r.x + matdyn.x) vs the full-grid reference.

SUBDIR: ph.x run on the q-subgrid (<prefix>.dyn0..N). REFDIR: ph.x run on the full grid. Runs
q2r.x (zasr 'crystal') and matdyn.x (asr 'crystal') in SUBDIR/ifc at the irreducible q of REFDIR's
dyn0 and prints, per q, max |dw| and the gate max(0.5 %, 1 cm-1) (Gamma acoustic modes ungated).
"""
import glob
import os
import re
import subprocess
import sys

import numpy as np

sub, ref, bindir = (os.path.realpath(x) for x in sys.argv[1:4])
prefix = os.path.basename(glob.glob(os.path.join(ref, "*.dyn0"))[0])[:-5]
w = os.path.join(sub, "ifc")
os.makedirs(w, exist_ok=True)
with open(os.path.join(w, "q2r.in"), "w") as fh:
    fh.write(f"&input\n fildyn = '{sub}/{prefix}.dyn'\n zasr = 'crystal'\n flfrc = 'sub.fc'\n/\n")
with open(os.path.join(ref, f"{prefix}.dyn0")) as fh:
    fh.readline()
    nq = int(fh.readline().split()[0])
    qs = [fh.readline().split()[:3] for _ in range(nq)]
amass = re.findall(r"amass\(\d+\)\s*=\s*(\S+)", open(os.path.join(ref, "ph.in")).read())
with open(os.path.join(w, "matdyn.in"), "w") as fh:
    fh.write("&input\n flfrc = 'sub.fc'\n asr = 'crystal'\n flfrq = 'sub.freq'\n fleig = ' '\n fldos = ' '\n"
             " flvec = ' '\n fldyn = ' '\n")
    for i, m in enumerate(amass):
        fh.write(f" amass({i + 1}) = {m}\n")
    fh.write(f"/\n{nq}\n" + "".join(f" {q[0]} {q[1]} {q[2]} 1\n" for q in qs))
for exe, inp in (("q2r.x", "q2r.in"), ("matdyn.x", "matdyn.in")):
    with open(os.path.join(w, inp)) as fi, open(os.path.join(w, inp[:-3] + ".out"), "w") as fo:
        subprocess.run([os.path.join(bindir, exe)], stdin=fi, stdout=fo, stderr=subprocess.STDOUT, cwd=w,
                       check=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
txt = open(os.path.join(w, "sub.freq")).read()
nb = int(re.search(r"nbnd=\s*(\d+)", txt).group(1))
nums = [float(x) for x in txt.split("/", 1)[1].split()]
step = len(nums) // nq          # q line (3 coordinates, plus the path length in recent matdyn) + nb
mat = [np.array(nums[(i + 1) * step - nb:(i + 1) * step]) for i in range(nq)]


def freqs(i):
    with open(os.path.join(ref, f"{prefix}.dyn{i}")) as fh:
        return np.array([float(x) for x in re.findall(r"freq.*=\s*\S+\s*\[THz\]\s*=\s*(\S+)\s*\[cm-1\]", fh.read())])


ok, wmax, relmax = True, 0.0, 0.0
print(f"{'q':>3s} {'max|dw| cm-1':>13s} {'max rel':>9s}")
for i in range(nq):
    a, b = freqs(i + 1), mat[i]
    d = np.abs(a - b)
    note = ""
    if all(abs(float(x)) < 1e-8 for x in qs[i]):
        note = f"  (Gamma acoustic, ungated: {d[:3].max():.3f})"
        a, d = a[3:], d[3:]
    big = np.abs(a) > 1.0
    rel = (d[big] / np.abs(a[big])).max()
    g = bool(np.all(d <= np.maximum(0.005 * np.abs(a), 1.0)))
    ok &= g
    wmax, relmax = max(wmax, d.max()), max(relmax, rel)
    print(f"{i + 1:3d} {d.max():13.3f} {rel:9.2e}  {'PASS' if g else 'FAIL'}{note}")
print(f"overall max |dw| {wmax:.3f} cm-1, max rel {relmax:.2e}: {'PASS' if ok else 'FAIL'}")
