#!/usr/bin/env python
"""Build the row -> gene index maps (results/e1_cache/*__genemap.npy) for all corpus files (needs the compiled reader)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_split as E
for s in ["tumor_3p", "pbmc_v2_typical", "pbmc_v3_lowgenes", "pbmc_v2_shallow", "brain_sn_v3_deep", "brain_sn_v3_intronic", "tcell_5p"]:
    for m in ["exon", "intron"]:
        t = time.time()
        g = E.gene_map(s, m)
        print(s, m, len(g), int(g.max()) + 1, f"{time.time() - t:.0f}s", flush=True)
