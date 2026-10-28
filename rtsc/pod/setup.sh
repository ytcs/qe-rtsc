#!/usr/bin/env bash
# One-shot setup on a fresh RunPod NVHPC pod. Builds on the container disk (/root);
# /workspace is a FUSE network fs and is far too slow for compiling.
#   curl -sSfL https://raw.githubusercontent.com/ytcs/qe-rtsc/rtsc/rtsc/pod/setup.sh | bash
set -euo pipefail
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl git libfftw3-dev pkg-config cmake ninja-build python3 python3-numpy >/dev/null
mkdir -p /root/rtsc && cd /root/rtsc
# Fork (devxlib is vendored: gitlab.com is unreachable from RunPod).
[ -d qe-rtsc ] || git clone -q -b rtsc https://github.com/ytcs/qe-rtsc.git
(cd qe-rtsc && git submodule update -q --init external/wannier90 external/mbd external/fox)
# Vanilla baseline: the same tag, the same submodule pins, plus the vendored devxlib snapshot.
if [ ! -d qe-vanilla ]; then
  git clone -q --depth 1 -b qe-7.6 https://github.com/QEF/q-e.git qe-vanilla
  (cd qe-vanilla && git submodule update -q --init external/wannier90 external/mbd external/fox)
  cp -r qe-rtsc/external/devxlib/. qe-vanilla/external/devxlib/
  (cd qe-vanilla/external/devxlib && git init -q && git add -A && git -c user.email=rtsc@local -c user.name=rtsc commit -qm "devxlib snapshot")
fi
qe-rtsc/rtsc/pseudo.sh
echo "ready: build with  qe-rtsc/rtsc/pod/build-gpu.sh qe-vanilla 80   (80=A100, 90=H100)"
