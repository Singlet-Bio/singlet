"""Per-cell chrM allele counts WITH strand from a Singlet tagged BAM (pysam), in the CSC layout the C++ caller reads.

Adapted from bench/modalities/genotype_pool/mt_pileup.py. Same per-base rules as the engine's pileup_mt_bases:
M/=/X CIGAR ops only, base quality >= 10, A/C/G/T only, reference position < 16569, one count per read base (no UMI
dedup), primary alignments only. Two variants are written in one pass over the BAM:

  OUT.q20.bin  MAPQ >= 20  what the FIXED engine counts (pileup_mt_bases gated on high_mapq)
  OUT.q0.bin   any MAPQ    what the pre-fix engine counted (NH>1 multimappers have MAPQ 0-3 and bypassed the filter)
  OUT.q20umi.bin  MAPQ >= 20, collapsed to molecules: per (cell, UB UMI, position) the consensus base of its reads counts
               once (what a UMI-aware engine would count; the engine counts reads). Reads without UB are dropped.

Usage: mt_pileup_strand.py BAM CELLS.txt OUT_PREFIX [ASSIGN.tsv]
  CELLS.txt  one barcode per line (column order of the matrices); the "-1" suffix is ignored
  ASSIGN.tsv Singlet donor/donor_assignments.tsv; if given, OUT_PREFIX.assign.tsv (cell idx, donor_id, status) is written
Binary layout (little endian):
  'MTC1' u32 n_cells u64 nnz_total u64 nnz_rev
  i32 indptr_total[n_cells+1] i32 indices_total[nnz] u16 data_total[nnz]     feature = pos*4 + base (A,C,G,T)
  i32 indptr_rev  [n_cells+1] i32 indices_rev  [nnz] u16 data_rev  [nnz]     reverse-strand subset
"""
import sys, json, struct, collections
from array import array
import numpy as np
import pysam

L = 16569
NB = {"A": 0, "C": 1, "G": 2, "T": 3}
MIN_BQ = 10


def write_bin(path, n_cells, keys):
    """keys: int64 array of ((cell*L + pos)*8 + base + 4*rev), one entry per counted base."""
    uk, cnt = np.unique(keys, return_counts=True)
    code = uk % 8
    base = code % 4
    rev = code // 4
    cp = uk // 8
    pos = cp % L
    cell = cp // L
    k2 = (cell * L + pos) * 4 + base
    u2, inv = np.unique(k2, return_inverse=True)
    tot = np.bincount(inv, weights=cnt).astype(np.int64)
    rv = np.bincount(inv, weights=cnt * rev).astype(np.int64)
    tot = np.minimum(tot, 65535).astype(np.uint16)
    rv = np.minimum(rv, 65535).astype(np.uint16)
    cell2 = u2 // (L * 4)
    feat2 = (u2 % (L * 4)).astype(np.int32)

    def csc(mask):
        c = cell2[mask]
        ip = np.concatenate([[0], np.cumsum(np.bincount(c, minlength=n_cells))]).astype(np.int32)
        return ip, feat2[mask], mask

    ipt, ixt, mt = csc(np.ones(len(u2), bool))
    ipr, ixr, mr = csc(rv > 0)
    with open(path, "wb") as f:
        f.write(b"MTC1")
        f.write(struct.pack("<IQQ", n_cells, len(ixt), len(ixr)))
        f.write(ipt.tobytes()); f.write(ixt.astype(np.int32).tobytes()); f.write(tot.tobytes())
        f.write(ipr.tobytes()); f.write(ixr.astype(np.int32).tobytes()); f.write(rv[mr].tobytes())
    return int(tot.sum()), int(len(ixt))


def molecules(keysU):
    """keysU: int64 ((cell*L+pos)<<27 | umi24<<3 | base<<1 | rev), one per read base.  Returns int64 keys in write_bin's layout
    ((cell*L+pos)*8 + base + 4*rev), one per (cell, umi, pos) molecule, base = read consensus, strand = majority strand."""
    uk, cnt = np.unique(keysU, return_counts=True)
    gb = uk >> 1                      # (cell,pos,umi,base)
    rev = uk & 1
    ugb, inv = np.unique(gb, return_inverse=True)
    c_gb = np.bincount(inv, weights=cnt)
    r_gb = np.bincount(inv, weights=cnt * rev)
    del uk, cnt, inv, gb, rev
    g2 = ugb >> 2                     # (cell,pos,umi)
    b2 = ugb & 3
    idx = np.lexsort((-c_gb, g2))
    g_sorted = g2[idx]
    first = np.concatenate([[True], g_sorted[1:] != g_sorted[:-1]])
    ch = idx[first]
    cp = g2[ch] >> 24                 # cell*L+pos
    r = (r_gb[ch] * 2 > c_gb[ch]).astype(np.int64)
    return (cp * 8 + b2[ch] + 4 * r).astype(np.int64)


def main():
    bam, cells_f, out = sys.argv[1:4]
    assign_f = sys.argv[4] if len(sys.argv) > 4 else None
    cells = [l.strip() for l in open(cells_f) if l.strip()]
    cidx = {c.split("-")[0]: i for i, c in enumerate(cells)}
    n_cells = len(cells)
    f = pysam.AlignmentFile(bam, "rb")
    chrm = [c for c in f.references if c in ("chrM", "MT", "M")][0]
    k20, k0, kU = array("q"), array("q"), array("q")
    st = collections.Counter()
    for r in f.fetch(chrm):
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        st["reads_primary"] += 1
        mq = r.mapping_quality
        try:
            nh = r.get_tag("NH")
        except KeyError:
            nh = 1
        st["mapq_%s" % ("ge20" if mq >= 20 else ("0" if mq == 0 else "1-19"))] += 1
        st["nh_%s_mapq_%s" % ("1" if nh == 1 else ">1", "ge20" if mq >= 20 else "lt20")] += 1
        try:
            cb = r.get_tag("CB")
        except KeyError:
            continue
        ci = cidx.get(cb.split("-")[0])
        if ci is None:
            continue
        st["reads_in_cells"] += 1
        st["reads_in_cells_mapq_ge20" if mq >= 20 else "reads_in_cells_mapq_lt20"] += 1
        q = r.query_qualities
        s = r.query_sequence
        rev = 4 if r.is_reverse else 0
        base_key = ci * L
        umi = None
        if mq >= 20:
            try:
                ub = r.get_tag("UB")
                u = 0
                for ch_ in ub[:12]:
                    u = u * 4 + "ACGT".find(ch_) if ch_ in "ACGT" else -1
                    if u < 0:
                        break
                umi = u if u >= 0 else None
            except KeyError:
                umi = None
            if umi is None:
                st["reads_q20_without_ub"] += 1
        for qp, rp in r.get_aligned_pairs(matches_only=True):
            if rp >= L or (q is not None and q[qp] < MIN_BQ):
                continue
            b = NB.get(s[qp])
            if b is None:
                continue
            k = (base_key + rp) * 8 + b + rev
            k0.append(k)
            if mq >= 20:
                k20.append(k)
                if umi is not None:
                    kU.append(((base_key + rp) << 27) | ((umi & 0xFFFFFF) << 3) | (b << 1) | (1 if rev else 0))
    st["bases_q0"] = len(k0)
    st["bases_q20"] = len(k20)
    print(json.dumps(st, indent=1), flush=True)
    st["bases_q20umi_input"] = len(kU)
    for name, arr in (("q20", k20), ("q0", k0)):
        tot, nnz = write_bin("%s.%s.bin" % (out, name), n_cells, np.frombuffer(arr, dtype=np.int64))
        print(name, "total bases", tot, "nnz", nnz, flush=True)
    mk = molecules(np.frombuffer(kU, dtype=np.int64))
    st["molecule_bases_q20umi"] = len(mk)
    tot, nnz = write_bin("%s.q20umi.bin" % out, n_cells, mk)
    print("q20umi molecule bases", tot, "nnz", nnz, flush=True)
    json.dump(dict(st), open(out + ".pileup_stats.json", "w"), indent=1)
    if assign_f:
        import pandas as pd
        da = pd.read_csv(assign_f, sep="\t")
        da["bc"] = da.cell.astype(str).str.split("-").str[0]
        da = da.drop_duplicates("bc").set_index("bc")
        with open(out + ".assign.tsv", "w") as o:
            o.write("idx\tdonor_id\tstatus\n")
            for i, c in enumerate(cells):
                bc = c.split("-")[0]
                if bc in da.index:
                    d = str(da.loc[bc, "donor_id"])
                    status = str(da.loc[bc, "cell_status"])
                else:
                    d, status = "unassigned", "cell"
                o.write("%d\t%s\t%s\n" % (i, d, status))
    print("done")


if __name__ == "__main__":
    main()
