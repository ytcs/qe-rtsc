#!/usr/bin/env bash
# Fetch the SG15 ONCV pseudopotentials (CC BY-SA 4.0, quantum-simulation.org) used by the benchmarks.
set -euo pipefail
D=${1:-$(dirname "$(realpath "$0")")/pseudo/sg15}
mkdir -p "$D" && cd "$D"
[ -f Mg_ONCV_PBE-1.2.upf ] || curl -sSfL http://www.quantum-simulation.org/potentials/sg15_oncv/sg15_oncv_upf_2020-02-06.tar.gz | tar xz
