# SPDX-License-Identifier: MIT
"""Why given V genes are missing from the reference: their biotype, spliced sequence and frames."""
import sys

from singlet.vdj import reference as R

genome, gtf = sys.argv[1], sys.argv[2]
names = set(sys.argv[3:])
genes = {k: g for k, g in R.parse_gtf(gtf).items() if g["name"] in names}
print("found in GTF parse:", {g["name"]: g["biotype"] for g in genes.values()})
chroms = R._read_fasta(genome, {g["contig"] for g in genes.values()})
for gid, g in genes.items():
    ts = g["transcripts"]
    tid = next((t for t, v in ts.items() if v["canonical"]), None) or sorted(ts)[0]
    seq = R._spliced(chroms[g["contig"]], ts[tid]["exons"], g["strand"])
    print(g["name"], g["biotype"], len(seq), "open frame:", R._open_v_frame(seq))
    for fr in range(3):
        print("  frame", fr, "anchor", R._v_anchor(seq, fr), R.translate(seq, fr))
