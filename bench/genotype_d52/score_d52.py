#!/usr/bin/env python3
"""PRIVATE (runs on Clipper only). Concordance of the D-52 multi-sample donor VCF (private/genotype/donors.vcf.gz, here the backfilled
copy under $WORK/genotypes/<set>/<pool>/) against the genotype truth of the gate's donor-snps row, beside the engine's per-donor VCFs,
vireo and the plain cellsnp pileup. Reuses S10's scorer (bench/public/genotype/lib/snp_concordance.py, branch bench/public-genotype) and
adds one tool, "singlet-d52". Writes counts only (3x3 genotype-class tables per pool, tool, donor and stratum), never a site or genotype.

usage: score_d52.py LIB GENOTYPES_SET_DIR RUNS TRUTH_DIR OUT.tsv POOL [POOL ...]
"""
import sys
from pathlib import Path

LIB, GT = Path(sys.argv[1]), Path(sys.argv[2])
sys.argv = [sys.argv[0]] + sys.argv[3:]
sys.path.insert(0, str(LIB))
sys.path.insert(0, str(LIB.parent.parent / "capabilities"))
import genotypes as G  # noqa: E402
import pools as P  # noqa: E402
import snp_concordance as SC  # noqa: E402

_tlv, _tdg = SC.tool_labels_and_vcfs, SC.tool_donor_gts


def tool_labels_and_vcfs(runs, pool):
    res = _tlv(runs, pool)
    v = GT / P.rundir(runs, pool).name / "donors.vcf.gz"
    if v.exists() and "singlet" in res:
        res["singlet-d52"] = (res["singlet"][0], v)   # same clusters and labels as the engine's demux
    return res


def tool_donor_gts(tool, path):
    if tool == "singlet-d52":
        keys, gts, dps = G.read_donor_vcf(path)
        return {"donor%d" % j: (keys, gts[j], dps[j]) for j in range(len(gts))}
    return _tdg(tool, path)


SC.tool_labels_and_vcfs, SC.tool_donor_gts = tool_labels_and_vcfs, tool_donor_gts
SC.main()
