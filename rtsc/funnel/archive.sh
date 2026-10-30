#!/usr/bin/env bash
# Archive EPW-reusable T3a outputs per control to /cache/funnel-data/F0c/<id>.tar.zst (<= 1.5 GB each).
#   archive.sh <t3a root> <id...>
R=$1; shift
mkdir -p /cache/funnel-data/F0c
for id in "$@"; do
  D=$R/$id; [ -d $D ] || continue
  cd $D
  L=$(ls -d save x.save x.xml x.chk x.win x.nnkp x.ukk *.in timings.tsv result.json x.dyn* epw_*.a2f_proj epw_*.out k3/x.save k3/nscf.in k3/*.in k3/epw_*.out k3/epw_*.a2f_proj 2>/dev/null)
  tar -I "zstd -T4 -3" -cf /cache/funnel-data/F0c/$id.tar.zst $L 2>/dev/null
  s=$(du -m /cache/funnel-data/F0c/$id.tar.zst | cut -f1)
  if [ "$s" -gt 1500 ]; then rm -f /cache/funnel-data/F0c/$id.tar.zst; echo "$id SKIPPED ($s MB > 1500)"; else echo "$id $s MB"; fi
done
