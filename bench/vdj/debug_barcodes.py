# SPDX-License-Identifier: MIT
"""Run the assembler on the reads of a few barcodes and print every stage (debugging missed contigs).

    python -P debug_barcodes.py FASTQ_DIR REF.tsv.gz WHITELIST BARCODE [BARCODE ...] [--expect JUNCTION ...]
"""
import argparse
import glob
import gzip
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fastq_dir")
    ap.add_argument("ref")
    ap.add_argument("whitelist")
    ap.add_argument("barcodes", nargs="+")
    ap.add_argument("--expect", nargs="*", default=[])
    a = ap.parse_args()
    from singlet.adt import count as AC
    from singlet.adt import seq as S
    from singlet.vdj import assemble as A

    want = {b[:16] for b in a.barcodes}
    r1s, r2s = [], []
    for p1 in sorted(glob.glob(os.path.join(a.fastq_dir, "**", "*_R1_*.fastq.gz"), recursive=True)):
        with gzip.open(p1, "rt") as f1, gzip.open(p1.replace("_R1_", "_R2_"), "rt") as f2:
            for i, (x, y) in enumerate(zip(f1, f2)):
                if i % 4 == 1 and x[:16] in want:
                    r1s.append(x.strip().encode())
                    r2s.append(y.strip().encode())
    print(f"{len(r1s)} read pairs of {len(want)} barcodes")
    ref = A.Reference.load(a.ref)
    wl = AC.load_barcode_list(a.whitelist)
    wl_codes, bad = S.encode_strings(wl, 16)
    # stage: classification of the reverse-complemented R2
    mates = [A._rc(s) for s in r2s]
    codes, off, o = A._classify(ref, mates, 2)
    def vn(i):
        return ref.names[i] if i >= 0 else "-"
    from collections import Counter

    print("V calls", Counter(vn(i) for i in o[:, 0]).most_common(6))
    print("J calls", Counter(vn(i) for i in o[:, 3]).most_common(6))
    print("C calls", Counter(vn(i) for i in o[:, 6]).most_common(6))
    L = (off[1:] - off[:-1])
    cys = np.where(o[:, 0] >= 0, o[:, 2] + ref.anchor[np.maximum(o[:, 0], 0)], -1)
    fw = np.where(o[:, 3] >= 0, o[:, 5] + ref.anchor[np.maximum(o[:, 3], 0)], -1)
    print("reads with Cys in read:", int(((cys >= 0) & (cys + 19 <= L)).sum()), " with F/W in read:",
          int(((fw + 3 - 19 >= 0) & (fw + 3 <= L) & (fw >= 0)).sum()))
    for e in a.expect:
        rc_hits = sum(e in m.decode() for m in mates)
        print(f"expected {e}: in {rc_hits} rc(R2) reads")
        for m, cy, f_, row in list(zip(mates, cys, fw, o))[:2000]:
            s = m.decode()
            k = s.find(e)
            if k >= 0:
                print("  read pos", k, "cys", cy, "fw", f_, "V", vn(row[0]), row[1], "J", vn(row[3]), row[4])
                break
    for frac in (0.02, 0.0):
        res = A.run_library([], [], ref=ref, cb_len=16, umi_len=10, whitelist_codes=np.sort(wl_codes[bad == 0]),
                            chunks=[(r1s, r2s)], params=A.Params(threads=1, umi_read_frac=frac), log=print)
        print("umi_read_frac", frac, res.stats)
        print(res.contigs[["barcode", "chain", "cdr3_nt", "umis", "junction_umis", "reads", "productive", "high_confidence",
                           "kept", "v_gene", "j_gene"]].to_string())


if __name__ == "__main__":
    main()
