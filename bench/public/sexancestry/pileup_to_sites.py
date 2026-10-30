#!/usr/bin/env python3
"""bcftools mpileup query output -> the per-donor depth table singlet.celllabels.ancestry reads.

    python pileup_to_sites.py pile.tsv af_table.parquet out.parquet

pile.tsv: CHROM POS REF ALT(observed, comma list, may contain <*>) AD(comma list: ref then each ALT). The alternate depth is
the depth of the panel's alternate allele (from the AF table), 0 when that allele was not observed; dp is the summed depth.
Output columns: chrom, pos, ref, alt, ad, dp. Private (per-site allele depth): it stays on the compute cluster.
"""
import sys

import numpy as np
import pandas as pd

pile, aft, out = sys.argv[1:4]
p = pd.read_csv(pile, sep="\t", names=["chrom", "pos", "ref", "alts", "AD"], dtype={"chrom": str})
p["chrom"] = p["chrom"].str.replace("^chr", "", regex=True)
af = pd.read_parquet(aft, columns=["chrom", "pos", "ref", "alt"]).astype({"chrom": str})
m = p.merge(af, on=["chrom", "pos", "ref"], how="inner")
ad = np.zeros(len(m), np.int64)
dp = np.zeros(len(m), np.int64)
for i, (alts, AD, alt) in enumerate(zip(m["alts"].to_numpy(), m["AD"].to_numpy(), m["alt"].to_numpy())):
    counts = [int(x) for x in str(AD).split(",") if x not in (".", "")]
    dp[i] = sum(counts)
    al = str(alts).split(",")
    if alt in al:
        j = al.index(alt) + 1
        ad[i] = counts[j] if j < len(counts) else 0
keep = dp > 0
pd.DataFrame({"chrom": m["chrom"][keep], "pos": m["pos"][keep], "ref": m["ref"][keep], "alt": m["alt"][keep],
              "ad": ad[keep], "dp": dp[keep]}).to_parquet(out, index=False)
print(f"{int(keep.sum())} covered panel sites")
