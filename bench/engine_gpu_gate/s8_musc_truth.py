# SPDX-License-Identifier: MIT
"""Long-read USA truth for MuSC (mouse muscle stem cells, ONT + Illumina 10x v2): the TRUTH-LR rule of TRUTH_RULE.md (U: >= 10 aligned intronic bases of the gene;
S: not U and an annotated junction +-2 nt; A otherwise; molecule = U > S > A over its primary ONT alignments) on the molecules of truth.v2.parquet.

MuSC has no tier 1 (ONT only); molecules are the tier-2 reads with a truth gene. The truth table's short-read projection exists for dev cells only, so only the long-read class is
built here. Cells: the 176 of split/cells_MuSC.txt (dev/test by split.tsv). Read-only on ground-truth.   usage: s8_musc_truth.py OUT.tsv.gz [--workers 4]
"""
import argparse
import collections
import sys
from multiprocessing import Pool
from pathlib import Path

GT = Path("/mnt/projects/debruinz_project/debruinz/ground-truth")
sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/expression-sota/longread_usa/code")
import truth_usa as T  # noqa: E402

T.GTF = Path("/mnt/projects/debruinz_project/cellarium/reference/GRCm39-2024-A/genes/genes.gtf")
BAMS = [GT / f"work/MuSC/aln/chunk_{i:03d}/aligned.bam" for i in range(4)]
CELLS = {ln.split("\t")[1]: ln.split("\t")[2].strip() for ln in open(GT / "split/split.tsv") if ln.startswith("MuSC\t")}
MOL: dict = {}   # (CB, UB) -> gene index


def build_molecules():
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    M = T.models()
    f = pq.ParquetFile(GT / "truth/MuSC/truth.v2.parquet")
    votes = collections.defaultdict(collections.Counter)
    for rg in range(f.metadata.num_row_groups):
        t = f.read_row_group(rg, columns=["CB", "UB", "tier", "has_truth", "true_gene"])
        t = t.filter(pc.and_(pc.equal(t["tier"], 2), t["has_truth"]))
        for cb, ub, g in zip(t["CB"].to_pylist(), t["UB"].to_pylist(), t["true_gene"].to_pylist()):
            if cb in CELLS and g and g in M.idx:
                votes[(cb, ub)][M.idx[g]] += 1
    tie = 0
    for k, c in votes.items():
        top = c.most_common(2)
        if len(top) > 1 and top[0][1] == top[1][1]:
            tie += 1
            continue
        MOL[k] = top[0][0]
    print(f"[truth] molecules {len(MOL)} (tie dropped {tie}) in {len({k[0] for k in MOL})} cells", file=sys.stderr, flush=True)


def scan(bam):
    import pysam
    M = T.models()
    res = {}
    n = 0
    with pysam.AlignmentFile(str(bam), "rb", check_sq=False, threads=2) as f:
        for r in f.fetch(until_eof=True):
            n += 1
            if r.flag & 0x904:
                continue
            try:
                k = (r.get_tag("CB"), r.get_tag("UB"))
            except KeyError:
                continue
            g = MOL.get(k)
            if g is None or M.chrom[g] != r.reference_name:
                continue
            blocks, juncs, pos, bs = [], [], r.reference_start, r.reference_start
            for op, ln in r.cigartuples or []:
                if op in (0, 2, 7, 8):
                    pos += ln
                elif op == 3:
                    blocks.append((bs, pos)); juncs.append((pos, pos + ln)); pos += ln; bs = pos
            blocks.append((bs, pos))
            ni = M.n_int(g, blocks)
            sp = (M.n_annot(g, juncs) > 0) if juncs else False
            o = res.get(k)
            res[k] = (max(o[0], ni), o[1] or sp, o[2] + 1) if o else (ni, sp, 1)
    print(f"[scan] {bam.parent.name}: {n} records, {len(res)} molecules", file=sys.stderr, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    build_molecules()
    with Pool(min(a.workers, len(BAMS))) as p:
        parts = p.map(scan, BAMS)
    agg = {}
    for part in parts:
        for k, (ni, sp, n) in part.items():
            o = agg.get(k)
            agg[k] = (max(o[0], ni), o[1] or sp, o[2] + n) if o else (ni, sp, n)
    import gzip
    M = T.models()
    cls = lambda ni, sp, thr: "U" if ni >= thr else ("S" if sp else "A")   # noqa: E731
    with gzip.open(a.out, "wt") as w:
        w.write("CB\tUB\tgene\tsplit\tclsLR\tclsLR5\tclsLR20\tnintLR\tn_ont\n")
        for (cb, ub), (ni, sp, n) in agg.items():
            w.write(f"{cb}\t{ub}\t{M.gids[MOL[(cb, ub)]]}\t{CELLS[cb]}\t{cls(ni, sp, 10)}\t{cls(ni, sp, 5)}\t{cls(ni, sp, 20)}\t{ni}\t{n}\n")
    c = collections.Counter(cls(ni, sp, 10) for ni, sp, n in agg.values())
    print(f"[truth] molecules with a long-read class {len(agg)} of {len(MOL)}; classes {dict(c)}", file=sys.stderr)


if __name__ == "__main__":
    main()
