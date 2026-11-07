"""Ceiling of a transcriptome-first (L1) step in front of STAR, from a STAR_PROFILE build's Log.out.

    python txome_ceiling.py <star_Log.out> [...]

The profile build (-DSTAR_PROFILE) classifies every unique read that STAR mapped itself (not a sequence-cache
hit) by alignment shape and sums its mapping cycles (seed search, windows, stitching):

  g1perfect{U,S}  one-gene concordant (STARsolo Gene), annotated junctions only, no indel, not chrM,
                  0 mismatches, no soft clip; U unspliced, S spliced
  g1mm1{U,S}      same with exactly 1 mismatch
  g1other{U,S}    same gene/junction conditions but soft-clipped or >1 mismatch
  uniqueOther     every other unique read (intronic, intergenic, antisense, multi-gene, novel junctions, chrM)

An L1 step can at best remove the mapping time of the classes it answers, so the share of `map` per class is
the upper bound on its gain (at zero L1 cost and zero cost for the reads it rejects).
"""

import re
import sys

for path in sys.argv[1:]:
    line = next(l for l in open(path) if l.startswith("STAR_PROFILE"))
    total = float(re.search(r" map=([\d.]+)s", line).group(1))
    print(f"== {path}\n  map={total:.1f}s (thread-seconds)")
    for m in re.finditer(r" (g1\w+|uniqueOther|unique|multi|tooShort|cacheHit)=(\d+)reads/([\d.]+)s(\[[^\]]*\])?", line):
        name, n, sec, det = m.group(1), int(m.group(2)), float(m.group(3)), m.group(4) or ""
        us = 1e6 * sec / n if n else 0.0
        print(f"  {name:12s} {n:>10,d} reads {sec:8.1f}s {100 * sec / total:5.1f}% of map {us:6.1f} us/read {det}")
