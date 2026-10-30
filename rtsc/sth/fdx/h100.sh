#!/usr/bin/env bash
# H100 measurement of the cuFFTDx local operator (groups/compute/FFT-LOG.md; pre-registered kill: FP64 FFT-chain gain < 2x).
# Budget <= 1 h. Phase 1 (~15 min) decides the kill; phase 2 (~35 min) is the in-situ ph.x A/B, run only if time is left.
#   curl -sSfL https://raw.githubusercontent.com/ytcs/qe-rtsc/fftdx/rtsc/sth/fdx/h100.sh | bash
# or pipe this file over ssh. Results go to /root/fdx-h100/ (and are printed). Does NOT save any cache archive.
set -uo pipefail
SHA=${SHA:-fftdx}
OUT=/root/fdx-h100; mkdir -p $OUT; cd $OUT
exec > >(tee -a $OUT/h100.log) 2>&1
echo "== $(date -u +%H:%M:%S) start, ref $SHA"
nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,clocks.max.sm --format=csv
nproc
NV=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5
export PATH=$NV/cuda/13.2/bin:$NV/compilers/bin:$PATH
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git zstd curl python3-venv >/dev/null

# ---------------- phase 1: microbenchmark (decides the kill) ----------------
[ -d src ] || git clone -q --depth 1 -b $SHA https://github.com/ytcs/qe-rtsc.git src 2>/dev/null || {
  # fallback: only the files the harness needs
  mkdir -p src/rtsc/sth/fdx
  for f in rtsc/sth/sth_pfft.f90 rtsc/sth/sth_pfft_s.f90 rtsc/sth/fdx/{sth_fftdx.cu,fdxbench.cu,pfwrap.f90,gemmbench.cu,build.sh,deps.sh}; do
    curl -sSfL https://raw.githubusercontent.com/ytcs/qe-rtsc/$SHA/$f -o src/$f; done; chmod +x src/rtsc/sth/fdx/*.sh; }
(cd src && git log --oneline -1 2>/dev/null)
eval "$(FDX_DEPS=/root/rtsc/fdxdeps src/rtsc/sth/fdx/deps.sh)"
MDX=$MDX CUTLASS=$CUTLASS src/rtsc/sth/fdx/build.sh 90 2>&1 | grep -v Wno | tail -5
BIN=src/rtsc/sth/fdx/build
echo "== $(date -u +%H:%M:%S) fdxbench (A cuFFT chain, P pfft, D fftdx; FP64 then FP32; us per column application)"
for s in B1 B2; do $BIN/fdxbench $s 1024 8 20; done
echo "== repeat (spread), B2 x3"
for r in 1 2 3; do NOPF=1 $BIN/fdxbench B2 1024 8 20 | grep -E "A cuFFT|D fftdx"; done
echo "== batch 4096 columns"
for s in B1 B2; do NOPF=1 $BIN/fdxbench $s 4096 8 10 | grep -E "^B|FP|A cuFFT|D fftdx"; done
echo "== split-line plane kernel (FDX_SPLIT=1)"
for s in B1 B2; do FDX_SPLIT=1 NOPF=1 $BIN/fdxbench $s 1024 8 20 | grep -E "FP|D fftdx|occupancy"; done
echo "== per-kernel ablation (FDX_ONLY=1 kz_inv, 2 kxy, 3 kz_fwd)"
for s in B1 B2; do for o in 1 2 3; do echo "-- $s kernel $o"; FDX_ONLY=$o NOPF=1 $BIN/fdxbench $s 1024 8 20 | grep "D fftdx"; done; done
echo "== dense V(G-G') as a GEMM (item 7)"
$BIN/gemmbench
echo "== $(date -u +%H:%M:%S) phase 1 done"

# ---------------- phase 2: in-situ ph.x, B1 frozen q2, fftdx off / on (skip with PHASE2=0) ----------------
[ "${PHASE2:-1}" = 1 ] || exit 0
echo "== $(date -u +%H:%M:%S) warm build (sm_90, branch $SHA, cuFFTDx on)"
mkdir -p /root/rtsc
curl -sSfL https://raw.githubusercontent.com/ytcs/qe-rtsc/$SHA/rtsc/pod/warm.sh -o /root/warm.sh
# warm.sh saves to the cache only on a cache miss; the sm90 archive exists, so nothing is written to /cache
sed -i 's|^\[ -f "\$ARC" \] \|\| CC=\$CC qe-rtsc/rtsc/pod/save.sh|echo "(no save)"|' /root/warm.sh
BRANCH=$SHA CC=90 J=$(nproc) MDX=$MDX CUTLASS=$CUTLASS bash /root/warm.sh 2>&1 | tail -3
cd /root/rtsc && source env.sh
B=/root/rtsc/qe-rtsc/rtsc/bench; R=$OUT/runs; Q=/root/rtsc/qe-rtsc/build-gpu/bin
mkdir -p $R/scf && cd $R/scf && cp $B/B1/frozen/*.in . && sed -i "s|@PSEUDO@|$B/../pseudo/sg15|" *.in
RTSC_STH=1 OMP_NUM_THREADS=8 mpirun --allow-run-as-root --bind-to none -np 1 -x OMP_NUM_THREADS -x RTSC_STH $Q/pw.x -in scf.in > scf.out
echo "== $(date -u +%H:%M:%S) scf done"
mkdir -p /root/rtsc && touch /root/rtsc/TIMING.lock
RTSC_STH_DVB=1 RTSC_STH_TIME=1 $B/phq.sh $R/scf $R/q2-off $Q 2 2
RTSC_STH_DVB=1 RTSC_STH_TIME=1 RTSC_STH_FFTDX=1 RTSC_STH_FFTDX_CHECK=1 $B/phq.sh $R/scf $R/q2-on $Q 2 2
python3 $B/cmpph.py $R/q2-off $R/q2-on
grep -h "rtsc fftdx" $R/q2-on/ph.out | head -3
for d in q2-off q2-on; do echo "-- $d"; grep -hE "sth_kernel|PHONON  |c_bands|solve_linter" $R/$d/ph.out | head -6; done
echo "== $(date -u +%H:%M:%S) H100_DONE"
