"""Open-source comparators on a fragments file: MACS3 the way Signac's CallPeaks runs it, and SnapATAC2 per-cell QC.

    python competitors_atac.py FRAGMENTS.tsv.gz GTF OUTDIR [--genome hs|mm] [--cells cells.txt]

Writes OUTDIR/macs3_peaks.narrowPeak, OUTDIR/snapatac2_qc.tsv (barcode, n_fragment, tsse, frac_mito,
nucleosome_signal_signac), OUTDIR/competitors_time.json (wall seconds per step).
Signac CallPeaks: macs2 callpeak -t <fragments> -g <genome> -f BED --nomodel --extsize 200 --shift -100 -n <name>
--keep-dup all (q = 0.05, the MACS default). SnapATAC2: pp.import_fragments + metrics.tsse (ArchR-style TSSe).
"""
import argparse, gzip, json, os, subprocess, sys, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("fragments"); ap.add_argument("gtf"); ap.add_argument("out")
ap.add_argument("--genome", default="hs"); ap.add_argument("--cells", default=None)
ap.add_argument("--skip-macs", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tm = {}
env_bin = os.path.dirname(sys.executable)

if not a.skip_macs:
    t = time.time()
    subprocess.run([f"{env_bin}/macs3", "callpeak", "-t", a.fragments, "-g", a.genome, "-f", "BED", "--nomodel",
                    "--extsize", "200", "--shift", "-100", "--keep-dup", "all", "-n", "macs3", "--outdir", a.out],
                   check=True, stderr=open(f"{a.out}/macs3.log", "w"))
    tm["macs3_callpeak_s"] = time.time() - t

import snapatac2 as snap
t = time.time()
cs = snap.genome.hg38.chrom_sizes if a.genome == "hs" else snap.genome.mm39.chrom_sizes
whitelist = None
if a.cells:
    whitelist = [l.strip() for l in open(a.cells) if l.strip()]
ad = snap.pp.import_fragments(a.fragments, chrom_sizes=cs, min_num_fragments=0 if whitelist else 200,
                              sorted_by_barcode=False, whitelist=whitelist, chrM=["chrM", "MT"])
tm["snapatac2_import_s"] = time.time() - t
t = time.time()
snap.metrics.tsse(ad, gene_anno=a.gtf)
tm["snapatac2_tsse_s"] = time.time() - t
obs = ad.obs.copy()
obs.index.name = "barcode"
obs.index = obs.index.str.replace("-1", "", regex=False)

# Signac NucleosomeSignal on all fragments of the listed cells: (147..294] / < 147, computed here from the file.
mono, nfr = {}, {}
keep = set(obs.index)
with gzip.open(a.fragments, "rt") as fh:
    for line in fh:
        if line[0] == "#":
            continue
        c, s, e, b, _ = line.split("\t", 4)
        b = b.replace("-1", "")
        if b not in keep or c in ("chrM", "MT"):
            continue
        L = int(e) - int(s)
        if L < 147:
            nfr[b] = nfr.get(b, 0) + 1
        elif L <= 294:
            mono[b] = mono.get(b, 0) + 1
obs["nucleosome_signal_signac"] = [mono.get(b, 0) / nfr[b] if nfr.get(b) else np.nan for b in obs.index]
obs.to_csv(f"{a.out}/snapatac2_qc.tsv", sep="\t")
json.dump(tm, open(f"{a.out}/competitors_time.json", "w"), indent=1)
print(json.dumps(tm))
