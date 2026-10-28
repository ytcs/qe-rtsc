# Compare Wannier-interpolated bands with DFT on the same path: python -I cmp.py <dir> [EF]
import sys, re, numpy as np
d, EF = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 10.83
e = np.load("dft_path.npy")[:, 12:]                      # bands 13-40
w = np.loadtxt(f"{d}/mg2irh6_band.dat")[:, 1]
nk = len(e); w = np.sort(w.reshape(-1, nk).T, axis=1); nw = w.shape[1]
# match each Wannier band to the nearest DFT band within the frozen manifold only
froz = float(re.search(r"dis_froz_max = (\S+)", open(f"{d}/mg2irh6.win").read()).group(1))
err = []
for k in range(nk):
    ek = e[k][e[k] <= froz]
    err += [np.min(np.abs(w[k] - x)) for x in ek]     # every frozen DFT state must be reproduced
err = np.array(err)
ef = np.concatenate([e[k][(np.abs(e[k] - EF) < 0.5)] for k in range(nk)])
near = []
for k in range(nk):
    for x in e[k][(np.abs(e[k] - EF) < 0.5) & (e[k] <= froz)]:
        near.append(np.min(np.abs(w[k] - x)))
wout = open(f"{d}/mg2irh6.wout").read()
spreads = [float(s) for s in re.findall(r"WF centre and spread\s+\d+\s+\(.*?\)\s+(\S+)", wout.split("Final State")[-1])]
conv = "not satisfied" not in wout
print(f"{d}: nw={nw} froz={froz} max|dE| frozen {err.max()*1e3:.1f} meV, |E-EF|<0.5 {max(near)*1e3:.1f} meV, "
      f"spreads {min(spreads):.2f}-{max(spreads):.2f} total {sum(spreads):.2f} Å², dis converged {conv}")
