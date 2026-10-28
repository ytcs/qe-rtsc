# qe-rtsc

The RTSC campaign's fork of Quantum ESPRESSO, branched from the `qe-7.6`
release tag. EPW is included in this tree. Everything specific to the fork
lives under `rtsc/`:

- `bench/`: frozen benchmark inputs (`mkbench.py` generates them) and the
  timed pipeline runner `run.sh`.
- `pod/`: GPU build script for the NVHPC image on RunPod.
- `pseudo.sh`: fetches the SG15 pseudopotentials the benchmarks use.

Benchmark rules (baseline, accuracy gate, timing) are in MILESTONE-1.md
in the campaign workspace.

On a GPU pod:

    git clone https://github.com/ytcs/qe-rtsc && cd qe-rtsc && git checkout rtsc
    rtsc/pseudo.sh && rtsc/pod/build-gpu.sh . 80      # 80 = A100, 90 = H100

## Branches and tags

- `rtsc`: integration branch. Everything here passes the benchmark
  accuracy gate.
- `qe-7.6`: the vanilla baseline. Every speedup is measured against a build
  of this tag made with the same toolchain on the same node.
- Remotes: `upstream` (gitlab.com/QEF/q-e, where development happens),
  `github-mirror` (github.com/QEF/q-e), `origin` (ours, private).

## Licence

QE ships the GPL v2 text in `License`. Most source headers say "GNU General
Public License" without a version number, and a few say "version 2 or any
later version". Before importing code from another package, check that its
licence is compatible and record the decision here:

| Package | Licence | Status |
|---------|---------|--------|
| Wannier90 | GPL v2 | compatible |
| ABINIT | GPL v3 | **do not merge code** until QE's version terms are settled; use it as an external cross-check |
| python-sscha / CellConstructor | check before import | — |
| phonopy, pymatgen, ASE | BSD / MIT / LGPL | interface-level use only |

Changes are distributed under QE's terms (GPL).
