# SPDX-License-Identifier: MIT
"""First reads and the most frequent prefixes of a FASTQ (read-structure checks)."""
import gzip
import sys
from collections import Counter

path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200000
c = Counter()
with gzip.open(path, "rt") as f:
    for i, ln in enumerate(f):
        if i % 4 == 1:
            if i < 40:
                print(ln.strip())
            c[ln[:60]] += 1
        if i > 4 * n:
            break
for k, v in c.most_common(25):
    print(v, k)
