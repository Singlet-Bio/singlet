#!/usr/bin/env python
"""List samples (with exon/intron .1pz sizes) of remote .singlet bundles: python -P tools/track_c_list.py GSE... """
import io, sys, zipfile, re
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_corpus import HttpFile, BASE

for gse in sys.argv[1:]:
    hf = HttpFile(BASE.format(gse=gse))
    zf = zipfile.ZipFile(io.BufferedReader(hf, buffer_size=1 << 20))
    rows = {}
    for zi in zf.infolist():
        m = re.match(r"samples/([^/]+)/(exon_counts|intron_counts)\.1pz$", zi.filename)
        if m:
            rows.setdefault(m.group(1), {})[m.group(2)] = zi.file_size
    print(f"[{gse}] {hf.size/1e6:.0f} MB, {len(rows)} samples")
    for g, d in sorted(rows.items()):
        print(f"  {g}\texon {d.get('exon_counts',0)/1e6:.1f} MB\tintron {d.get('intron_counts',0)/1e6:.1f} MB")
