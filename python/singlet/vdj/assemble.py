# SPDX-License-Identifier: MIT
"""Assemble and annotate V(D)J contigs per cell from a 10x 5' V(D)J (TCR / BCR enriched) library.

Steps (:func:`run_library`):

1. **Orientation.** A probe of the first reads decides which mates are sense: on 10x 5' V(D)J, R1 is cell barcode + UMI
   + template-switch oligo + the transcript's 5' end (sense), R2 is antisense from a fragmentation point. The probe
   classifies R2, its reverse complement and the R1 insert and keeps what hits the segments.
2. **Classification.** Every mate is matched with exact 15-mers to the V / J / C segments of
   :mod:`singlet.vdj.reference` (``_kernels.classify``); the best segment per region and its diagonal give the chain and
   the read positions of Cys104 and Phe/Trp118. Reads with no segment hit are counted and dropped.
3. **Barcodes and UMIs.** Cell barcodes are corrected against the whitelist with the ADT counting rules
   (:func:`singlet.adt.count.correct_barcodes`, Cell Ranger's 1-mismatch posterior), UMIs within a barcode with Cell
   Ranger's ``correct_umis`` (:func:`singlet.adt.count.correct_umis`). Molecules with fewer mates than
   ``umi_read_frac`` of the library's median (>= ``min_umi_reads``) are dropped: index hopping, chimeras, UMI errors.
4. **Assembly.** Per (barcode, chain), the reads that reach the junction are assembled into junction contigs by greedy
   extension through their 19-mer graph weighted by distinct UMIs (``_kernels.assemble_group``). A UMI belongs to the
   contig whose specific k-mers most of its reads carry.
5. **Annotation.** V, J and C genes by k-mer votes of the contig's reads (all mates of its UMIs), D by the longest exact
   match in the junction, the IMGT junction (``cdr3``: Cys104 ... Phe/Trp118, as Cell Ranger and TRUST4 report it) and
   its translation, productivity (in frame, no stop codon, conserved C and F/W, J motif in frame).
6. **Filtering.** A contig is kept when it has >= ``min_umis`` UMIs (>= ``min_junction_umis`` of them over the
   junction) and >= ``min_frac`` of the UMIs of the barcode's top contig of the same chain. Every kept contig is reported with its UMI and read support, so stricter
   cuts can be applied later.

Output: one row per contig (:data:`CONTIG_COLUMNS`), the layout of Cell Ranger's ``filtered_contig_annotations.csv``
where the meaning is the same.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import _kernels as K
from . import reference as R

K_INDEX = 15
K_ASM = 19
TSO = b"TTTCTTATATGGG"
CHUNK = 1_000_000
MAX_PATHS = 12
MAX_JUNCTION = 150
REGION = {"V": 0, "D": 1, "J": 2, "C": 3}
_RC = bytes.maketrans(b"ACGTN", b"TGCAN")

CONTIG_COLUMNS = ["barcode", "contig_id", "chain", "v_gene", "d_gene", "j_gene", "c_gene", "cdr3", "cdr3_nt", "productive",
                  "full_length", "umis", "junction_umis", "reads", "high_confidence", "kept"]


@dataclass
class Params:
    # filters set on the development data (10x pbmc2 TCR / BCR) only; the other benchmark libraries are held out
    min_umis: int = 1  # UMIs of a kept contig (junction UMIs + UMIs of the same chain and V gene)
    min_junction_umis: int = 1  # UMIs whose reads hold the contig's specific junction k-mers
    min_frac: float = 0.20  # of the barcode's top contig of the same chain
    min_seed_umis: int = 1
    seed_frac: float = 0.05  # seeds below this share of the top seed's UMIs are error k-mers
    min_branch_umis: int = 2
    branch_frac: float = 0.20
    min_hits: int = 2  # exact 15-mer hits for a segment call on a read
    min_barcode_umis: int = 2  # barcodes with fewer V(D)J UMIs are not assembled
    min_umi_reads: int = 2  # mates of a molecule, at least
    min_contig_reads: int = 3  # mates of a kept contig
    contig_read_frac: float = 0.05  # mates per UMI of a kept contig, against the library's read-weighted median
    umi_read_frac: float = 0.02  # and at least this share of the library's read-weighted median mates per molecule
    threads: int = 8


class Reference:
    def __init__(self, segs: List[R.Segment]):
        self.segs = segs
        self.names = np.array([s.name for s in segs])
        self.chain = np.array([s.chain for s in segs])
        self.region = np.array([REGION[s.region] for s in segs], dtype=np.int8)
        self.anchor = np.array([s.anchor for s in segs], dtype=np.int32)
        self.frame = np.array([s.frame for s in segs], dtype=np.int32)
        self.km, self.sg, self.ps = K.build_index([s.seq for s in segs], K_INDEX)
        self.chains = list(R.CHAINS)
        self.chain_id = np.array([self.chains.index(c) for c in self.chain], dtype=np.int8)
        self.d_segs = {c: [(s.name, s.seq) for s in segs if s.region == "D" and s.chain == c] for c in R.CHAINS}
        self.j_by_chain = {c: [i for i, s in enumerate(segs) if s.region == "J" and s.chain == c and s.anchor >= 0]
                           for c in R.CHAINS}
        # indistinguishable V genes (identical up to Cys104): one class, named by its first member, voting as one
        eq = R.equivalence(segs)
        first = {}
        self.seg_class = np.arange(len(segs), dtype=np.int32)
        for i, s in enumerate(segs):
            lab = eq.get(s.name)
            if lab and "|" in lab:
                self.seg_class[i] = first.setdefault(lab, i)

    @classmethod
    def load(cls, species_or_path: str = "human"):
        return cls(R.load(species_or_path))


# ----------------------------------------------------------------------------------------------------------- reads
def iter_chunks(r1_paths: Sequence[str], r2_paths: Sequence[str], chunk: int = CHUNK,
                max_reads: Optional[int] = None) -> Iterable[Tuple[List[bytes], List[bytes]]]:
    import dnaio

    n = 0
    a: List[bytes] = []
    b: List[bytes] = []
    for p1, p2 in zip(r1_paths, r2_paths):
        with dnaio.open(p1, file2=p2, mode="r") as rd:
            for x, y in rd:
                a.append(x.sequence.encode())
                b.append(y.sequence.encode())
                n += 1
                if len(a) >= chunk:
                    yield a, b
                    a, b = [], []
                if max_reads is not None and n >= max_reads:
                    if a:
                        yield a, b
                    return
    if a:
        yield a, b


def _codes(seqs: List[bytes]):
    off = np.zeros(len(seqs) + 1, np.int64)
    off[1:] = np.cumsum(np.fromiter((len(s) for s in seqs), np.int64, len(seqs)))
    buf = np.frombuffer(b"".join(seqs), dtype=np.uint8)
    return K.LUT[buf], off


def _rc(s: bytes) -> bytes:
    return s.translate(_RC)[::-1]


def _classify(ref: Reference, seqs: List[bytes], min_hits: int):
    codes, off = _codes(seqs)
    out = K.classify(codes, off, K_INDEX, ref.km, ref.sg, ref.ps, ref.region, min_hits, ref.seg_class)
    return codes, off, out


def probe_orientation(ref: Reference, r1: List[bytes], r2: List[bytes], r1_start: int, min_hits: int = 2) -> Dict:
    """Hit rates of R2, reverse-complemented R2 and the R1 insert (both strands) on a sample of pairs."""
    rates = {}
    for name, seqs in (("r2", r2), ("r2_rc", [_rc(s) for s in r2]), ("r1", [s[r1_start:] for s in r1]),
                       ("r1_rc", [_rc(s[r1_start:]) for s in r1])):
        _, _, o = _classify(ref, seqs, min_hits)
        rates[name] = float(((o[:, 1] > 0) | (o[:, 4] > 0) | (o[:, 7] > 0)).mean()) if len(o) else 0.0
    r2_rc = rates["r2_rc"] >= rates["r2"]
    r1_use = max(rates["r1"], rates["r1_rc"]) >= 0.05
    return {"rates": rates, "r2_rc": bool(r2_rc), "r1_use": bool(r1_use), "r1_rc": bool(rates["r1_rc"] > rates["r1"])}


def _r1_start(r1: List[bytes], cb_umi: int) -> int:
    """Insert start in R1: after barcode + UMI, plus the template-switch oligo when most reads carry it there."""
    if not r1 or len(r1[0]) <= cb_umi + len(TSO):
        return cb_umi
    hit = sum(1 for s in r1[:20000] if s[cb_umi:cb_umi + len(TSO)] == TSO)
    return cb_umi + len(TSO) if hit > 0.3 * min(20000, len(r1)) else cb_umi


# ----------------------------------------------------------------------------------------------------------- library
@dataclass
class LibraryResult:
    contigs: "object"  # pandas DataFrame, CONTIG_COLUMNS
    stats: Dict
    timings: Dict


def run_library(r1_paths: Sequence[str], r2_paths: Sequence[str], *, ref: Reference, cb_len: int, umi_len: int,
                whitelist_codes: np.ndarray, params: Params = Params(), max_reads: Optional[int] = None,
                chunks: Optional[Iterable] = None, log=print) -> LibraryResult:
    import pandas as pd

    from singlet.adt import count as AC
    from singlet.adt import seq as S

    t0 = time.time()
    tm: Dict[str, float] = {}
    st: Dict[str, int] = {"read_pairs": 0, "mates_with_segment": 0, "mates_informative": 0}
    it = iter(chunks) if chunks is not None else iter_chunks(r1_paths, r2_paths, max_reads=max_reads)
    cbu = cb_len + umi_len
    # stored per kept mate
    cbs: List[np.ndarray] = []
    umis: List[np.ndarray] = []
    meta: List[np.ndarray] = []  # (n, 9) classify + chain + cys + fw + informative
    seq_parts: List[np.ndarray] = []
    seq_lens: List[np.ndarray] = []
    orient = None
    r1_start = cbu
    for r1, r2 in it:
        st["read_pairs"] += len(r1)
        if orient is None:
            r1_start = _r1_start(r1, cbu)
            orient = probe_orientation(ref, r1[:100000], r2[:100000], r1_start, params.min_hits)
            orient["r1_insert_start"] = r1_start
            log(f"[vdj] orientation {orient}")
        mates = [(_rc(s) if orient["r2_rc"] else s) for s in r2]
        src = [np.arange(len(r1))]
        if orient["r1_use"]:
            ins = [s[r1_start:] for s in r1]
            mates += [(_rc(s) if orient["r1_rc"] else s) for s in ins]
            src.append(np.arange(len(r1)))
        src = np.concatenate(src)
        codes, off, o = _classify(ref, mates, params.min_hits)
        hits = np.stack([o[:, 1], o[:, 4], o[:, 7]], axis=1)
        anyhit = hits.max(axis=1) > 0
        st["mates_with_segment"] += int(anyhit.sum())
        # chain: of the region with most hits
        best_reg = hits.argmax(axis=1)
        seg_of = np.choose(best_reg, [o[:, 0], o[:, 3], o[:, 6]])
        chain = np.where(anyhit, ref.chain_id[np.maximum(seg_of, 0)], -1).astype(np.int8)
        L = (off[1:] - off[:-1]).astype(np.int32)
        cys = np.where(o[:, 0] >= 0, o[:, 2] + ref.anchor[np.maximum(o[:, 0], 0)], -1)
        cys = np.where((o[:, 0] >= 0) & (ref.anchor[np.maximum(o[:, 0], 0)] >= 0) & (cys >= 0) & (cys + K_ASM <= L), cys, -1)
        fw = np.where(o[:, 3] >= 0, o[:, 5] + ref.anchor[np.maximum(o[:, 3], 0)], -1)
        fw = np.where((o[:, 3] >= 0) & (ref.anchor[np.maximum(o[:, 3], 0)] >= 0) & (fw + 3 - K_ASM >= 0) & (fw + 3 <= L), fw, -1)
        # a J the reference lacks: Phe/Trp118 from the read, upstream of the C region
        todo = np.flatnonzero((o[:, 3] < 0) & (o[:, 6] >= 0) & (o[:, 8] >= 30) & (o[:, 8] <= L))
        if len(todo):
            f2 = K.fw_from_c(codes, off, o[:, 8].astype(np.int64), todo)
            ok2 = (f2 >= 0) & (f2 + 3 - K_ASM >= 0)
            fw[todo[ok2]] = f2[ok2]
        # informative: reaches the junction (V 3' end in the read, or a J hit)
        v_end = np.where(o[:, 0] >= 0, o[:, 2] + np.array([len(ref.segs[i].seq) for i in range(len(ref.segs))],
                                                                  dtype=np.int32)[np.maximum(o[:, 0], 0)], -10 ** 6)
        informative = anyhit & ((o[:, 3] >= 0) | (fw >= 0) | ((o[:, 0] >= 0) & (v_end - 30 < L)))
        st["mates_informative"] += int(informative.sum())
        keep = np.flatnonzero(anyhit)
        r1a = np.array(r1, dtype=object)
        cb = np.array([s[:cb_len] for s in r1a[src[keep]]], dtype=f"S{cb_len}")
        um = np.array([s[cb_len:cbu] for s in r1a[src[keep]]], dtype=f"S{umi_len}")
        cbs.append(cb)
        umis.append(um)
        mm = np.empty((len(keep), 13), np.int32)
        mm[:, :9] = o[keep, :9]
        # gene votes by equivalence class (the anchors above used each read's own segment)
        for col in (0, 3, 6):
            mm[:, col] = np.where(mm[:, col] >= 0, ref.seg_class[np.maximum(mm[:, col], 0)], -1)
        mm[:, 12] = o[keep, 9]  # tie flags: reads whose V / J / C ties between genes do not vote for a gene
        mm[:, 9] = chain[keep]
        mm[:, 10] = cys[keep]
        mm[:, 11] = np.where(informative[keep], fw[keep], -2)  # -2 marks a non-informative mate (no sequence kept)
        meta.append(mm)
        inf_idx = keep[informative[keep]]
        if len(inf_idx):
            starts = off[inf_idx]
            lens = (off[inf_idx + 1] - starts)
            idx = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) + np.arange(lens.sum())
            seq_parts.append(codes[idx])
            seq_lens.append(lens)
    tm["read_and_classify"] = time.time() - t0
    if orient is None:
        return LibraryResult(pd.DataFrame(columns=CONTIG_COLUMNS), st, tm)
    st["orientation"] = orient
    cb = np.concatenate(cbs)
    um = np.concatenate(umis)
    mm = np.concatenate(meta)
    del cbs, umis, meta
    seqc = np.concatenate(seq_parts) if seq_parts else np.zeros(0, np.uint8)
    slen = np.concatenate(seq_lens) if seq_lens else np.zeros(0, np.int64)
    del seq_parts, seq_lens
    soff = np.zeros(len(slen) + 1, np.int64)
    soff[1:] = np.cumsum(slen)
    inf_mask = mm[:, 11] != -2
    seq_row = np.full(len(mm), -1, np.int64)
    seq_row[inf_mask] = np.arange(inf_mask.sum())

    # barcodes
    t1 = time.time()
    ucb, inv = np.unique(cb, return_inverse=True)
    inv = inv.ravel()
    cb_reads = np.bincount(inv, minlength=len(ucb))
    ucb_mat = ucb.view(np.uint8).reshape(len(ucb), cb_len)
    code, valid, n_corr = AC.correct_barcodes(ucb_mat, cb_reads, whitelist_codes)
    st["mates_valid_barcode"] = int(cb_reads[valid].sum())
    st["barcodes_corrected"] = n_corr
    rv = valid[inv]
    bc_code = code[inv]
    # UMIs: per corrected barcode
    ucodes, ubad = S.encode(um.view(np.uint8).reshape(len(um), umi_len))
    ok = rv & (ubad == 0)
    st["mates_usable"] = int(ok.sum())
    sel = np.flatnonzero(ok)
    ubc, bci = np.unique(bc_code[sel], return_inverse=True)
    bci = bci.ravel()
    key = (bci.astype(np.uint64) << np.uint64(2 * umi_len)) | ucodes[sel]
    gu, ginv = np.unique(key, return_inverse=True)
    ginv = ginv.ravel()
    greads = np.bincount(ginv, minlength=len(gu))
    umask = np.uint64((1 << (2 * umi_len)) - 1)
    tgt = AC.correct_umis(gu >> np.uint64(2 * umi_len), gu & umask, greads, umi_len)
    mol = tgt[ginv]  # molecule id (index into gu) per usable mate
    st["umis"] = int(len(np.unique(tgt)))
    # read support per molecule: a V(D)J library sequences each molecule hundreds of times, while UMIs with a few reads
    # are index hopping, PCR chimeras and uncorrected UMI errors (in pbmc2 TCR, 1.5 reads per UMI for contigs Cell
    # Ranger does not report against ~800 for those it does). The cut adapts to the library's depth.
    mates_per_mol = np.bincount(mol, minlength=len(gu))
    mm_ = mates_per_mol[mates_per_mol > 0]
    # the depth a typical read sits at: the read-weighted median (junk molecules can outnumber real ones, e.g. pbmc2 TCR
    # with a median of 4 mates per molecule against ~800 per real molecule)
    srt = np.sort(mm_)
    cw = np.cumsum(srt)
    med = float(srt[np.searchsorted(cw, cw[-1] / 2)]) if len(srt) else 1.0
    umi_thr = max(params.min_umi_reads, int(np.ceil(params.umi_read_frac * med)))
    good = mates_per_mol >= umi_thr
    st["umi_read_threshold"] = umi_thr
    st["umi_reads_median"] = med
    st["umis_low_support"] = int((mm_ < umi_thr).sum())
    keep_m = good[mol]
    sel, bci, mol = sel[keep_m], bci[keep_m], mol[keep_m]
    # one chain per molecule (the majority of its mates): mates of another chain are chimeras or index hops, and as
    # a contig of their own they carry other cells' junctions (pbmc2 TCR: 2 reads each against ~3,800 for real ones)
    chain_m = mm[sel, 9].astype(np.int64)
    ok_c = chain_m >= 0
    mk = mol[ok_c].astype(np.int64) * 8 + chain_m[ok_c]
    uk_, cnt_ = np.unique(mk, return_counts=True)
    best_chain = np.full(len(gu), -1, np.int64)
    o_ = np.lexsort((-cnt_, uk_ // 8))
    sm_ = (uk_ // 8)[o_]
    head = np.ones(len(o_), bool)
    head[1:] = sm_[1:] != sm_[:-1]
    best_chain[sm_[head]] = (uk_ % 8)[o_][head]
    same = best_chain[mol] == chain_m
    st["mates_other_chain"] = int((~same & ok_c).sum())
    sel, bci, mol = sel[same], bci[same], mol[same]
    tm["barcodes_umis"] = time.time() - t1

    # barcodes worth assembling
    cell_of = bci  # dense barcode index per usable mate
    umis_per_bc = np.bincount(np.unique(np.stack([cell_of, mol], 1), axis=0)[:, 0], minlength=len(ubc))
    asm_bc = umis_per_bc >= params.min_barcode_umis
    st["barcodes_assembled"] = int(asm_bc.sum())

    # groups: (barcode, chain) over usable informative mates
    t2 = time.time()
    m_sel = mm[sel]
    s_row = seq_row[sel]
    chain = m_sel[:, 9]
    gi = np.flatnonzero(asm_bc[cell_of] & (s_row >= 0) & (chain >= 0))
    gkey = cell_of[gi].astype(np.int64) * 8 + chain[gi]
    o = np.argsort(gkey, kind="stable")
    gi = gi[o]
    gkey = gkey[o]
    bounds = np.flatnonzero(np.diff(gkey)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(gi)]])

    def work(k):
        rows = gi[starts[k]:ends[k]]
        mols = mol[rows]
        um_local, um_inv = np.unique(mols, return_inverse=True)
        res = K.assemble_group(seqc, soff, s_row[rows], um_inv.ravel().astype(np.int32), m_sel[rows, 10],
                               m_sel[rows, 11], K_ASM, params.min_seed_umis, params.seed_frac, params.min_branch_umis,
                               params.branch_frac, MAX_PATHS, MAX_JUNCTION)
        return k, rows, res

    groups = []
    with ThreadPoolExecutor(max_workers=params.threads) as ex:
        for k, rows, res in ex.map(work, range(len(starts))):
            if len(res[1]):
                groups.append((int(gkey[starts[k]]), rows, res))
    tm["assemble"] = time.time() - t2

    # annotation
    t3 = time.time()
    # per molecule: chain-wise votes from all its usable mates (informative or not)
    rows_all = []
    for gk, rows, (paths, plen, psup, cand) in groups:
        bcix, ch = divmod(gk, 8)
        chname = ref.chains[ch]
        mols = mol[rows]
        P = len(plen)
        # molecule -> contig (majority of its reads' assignments)
        okc = cand >= 0
        if not okc.any():
            continue
        mu, mi = np.unique(mols[okc], return_inverse=True)
        votes = np.zeros((len(mu), P), np.int64)
        np.add.at(votes, (mi.ravel(), cand[okc]), 1)
        mol_cand = votes.argmax(axis=1)
        rows_all.append((bcix, ch, chname, paths, plen, mu, mol_cand, votes))
    tm["annotate_prep"] = time.time() - t3

    # gene votes per (barcode, chain, molecule): one pass over all usable mates
    t4 = time.time()
    contigs = _annotate(ref, rows_all, sel, mm, cell_of, mol, ubc, cb_len, params)
    tm["annotate"] = time.time() - t4
    tm["total"] = time.time() - t0
    st["contigs"] = int(len(contigs))
    st["contigs_kept"] = int(contigs["kept"].sum()) if len(contigs) else 0
    return LibraryResult(contigs, st, tm)


def _best(values: np.ndarray, weights: np.ndarray):
    if len(values) == 0:
        return -1
    u, inv = np.unique(values, return_inverse=True)
    w = np.bincount(inv.ravel(), weights=weights)
    return int(u[w.argmax()])


def _vote(meta: np.ndarray, col: int, tie_bit: int, fallback: bool = True) -> int:
    """Gene with the most k-mer hits over reads whose call for that region is not a tie; with ``fallback``, all reads
    when every read ties (identical genes, e.g. over a short read)."""
    has = meta[:, col] >= 0
    clean = has & ((meta[:, 12] & tie_bit) == 0)
    use = clean if clean.any() or not fallback else has
    return _best(meta[use, col], meta[use, col + 1].astype(float))


def _mode_per_group(g: np.ndarray, n: int, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per group 0..n-1, the value (>= 0) with the largest summed weight, -1 when the group has none."""
    out = np.full(n, -1, np.int64)
    ok = values >= 0
    if not ok.any():
        return out
    g, v, w = g[ok], values[ok].astype(np.int64), weights[ok].astype(np.float64)
    nv = int(v.max()) + 1
    u, inv = np.unique(g * nv + v, return_inverse=True)
    s = np.bincount(inv.ravel(), weights=w)
    gu = u // nv
    o = np.lexsort((-s, gu))  # per group, largest weight first
    first = np.ones(len(o), bool)
    first[1:] = gu[o][1:] != gu[o][:-1]
    out[gu[o][first]] = (u % nv)[o][first]
    return out


def _junction_j(ref: "Reference", chain: str, junction: str, voted: int) -> int:
    """J gene of a junction by its germline 3' end: the J whose sequence up to the end of its F/W118 codon matches the
    junction's end over the longest stretch (one mismatch allowed). The part of J left of F/W118 tells paralogs apart
    (IGHJ4 ...YFDYW vs IGHJ5 ...NWFDPW) where reads past F/W118 often do not. The read vote breaks ties."""
    best, cands = -1, []
    for i in ref.j_by_chain.get(chain, ()):
        s = ref.segs[i]
        g = s.seq[:s.anchor + 3]
        mm = n = 0
        for x, y in zip(reversed(junction), reversed(g)):
            if x != y:
                mm += 1
                if mm > 1:
                    break
            n += 1
        if n > best:
            best, cands = n, [i]
        elif n == best:
            cands.append(i)
    if not cands:
        return voted
    if voted >= 0:
        # the read vote stands unless its J matches the junction's end clearly worse (> 2 nt): BCR junctions carry
        # somatic mutations and D-derived bases where the read votes past F/W118 are the better evidence
        s = ref.segs[voted]
        g = s.seq[:s.anchor + 3]
        mm = n = 0
        for x, y in zip(reversed(junction), reversed(g)):
            if x != y:
                mm += 1
                if mm > 1:
                    break
            n += 1
        if n >= best - 2:
            return voted
    return cands[0]


def _d_gene(ref: Reference, chain: str, junction: str) -> str:
    best, name = 0, ""
    for n, s in ref.d_segs.get(chain, []):
        # longest common substring, exact, >= 6 nt
        L = len(s)
        for ln in range(min(L, len(junction)), 5, -1):
            if ln <= best:
                break
            if any(s[i:i + ln] in junction for i in range(L - ln + 1)):
                best, name = ln, n
                break
    return name


def _annotate(ref: Reference, groups, sel, mm, cell_of, mol, ubc, cb_len, params: Params):
    import pandas as pd

    from singlet.adt import seq as S

    m_sel = mm[sel]
    chain_all = m_sel[:, 9]
    # index usable mates by (barcode, chain)
    key_all = cell_of.astype(np.int64) * 8 + chain_all
    o = np.argsort(key_all, kind="stable")
    ks = key_all[o]
    rows = []
    for bcix, ch, chname, paths, plen, mu, mol_cand, votes in groups:
        k = bcix * 8 + ch
        lo, hi = np.searchsorted(ks, k), np.searchsorted(ks, k, side="right")
        mates = o[lo:hi]
        mmol = mol[mates]
        # molecule -> contig for every molecule of this (barcode, chain); unassigned = -1
        pos = np.searchsorted(mu, mmol)
        pos = np.minimum(pos, len(mu) - 1)
        mc = np.where(mu[pos] == mmol, mol_cand[pos], -1)
        P = len(plen)
        meta = m_sel[mates]
        # V votes of the assigned molecules, per contig
        cinfo = []
        for p in range(P):
            on = mc == p
            v = _vote(meta[on], 0, 2)
            j = _vote(meta[on], 3, 4)
            c = _vote(meta[on], 6, 1, fallback=False)
            cinfo.append([v, j, c, int(len(np.unique(mmol[on]))), int(on.sum())])
        # unassigned molecules: to the contig with the same V gene, else to the only contig
        un = mc < 0
        extra = np.zeros(P, np.int64)
        if un.any():
            umol, uinv = np.unique(mmol[un], return_inverse=True)
            uv = _mode_per_group(uinv.ravel(), len(umol), meta[un, 0], meta[un, 1])
            for i in range(len(umol)):
                tgt = [p for p in range(P) if uv[i] >= 0 and cinfo[p][0] == uv[i]]
                if len(tgt) == 1:
                    extra[tgt[0]] += 1
                elif P == 1 and uv[i] < 0:
                    extra[0] += 1
        for p in range(P):
            junc = "".join("ACGT"[b] for b in paths[p, :plen[p]])
            v, j, c, ju, nreads = cinfo[p]
            junc_ = "".join("ACGT"[b] for b in paths[p, :plen[p]])
            j = _junction_j(ref, chname, junc_, j)
            if chname == "TRB" and j >= 0:
                # V-D2-J2 joining deletes TRBC1 (it lies between the J1 and D2 clusters), so a TRBJ2 rearrangement
                # can only use TRBC2; TRBC1 and TRBC2 differ at a few bases, so reads often cannot tell them apart
                jn = ref.names[j]
                cn = "TRBC2" if jn.startswith("TRBJ2") else ("TRBC1" if c < 0 else ref.names[c])
                c = int(np.flatnonzero(ref.names == cn)[0]) if (ref.names == cn).any() else c
            aa = R.translate(junc)
            inframe = len(junc) % 3 == 0
            prod = (inframe and "*" not in aa and aa[:1] == "C" and aa[-1:] in ("F", "W"))
            if prod and j >= 0:
                js = ref.segs[j]
                # J frame continues the junction frame: the F/W codon of the junction is the J anchor codon
                prod = js.anchor >= 0 and "*" not in R.translate(js.seq[js.anchor:], 0)[:4]
            rows.append({"bcix": bcix, "chain": chname, "v_gene": ref.names[v] if v >= 0 else "",
                         "d_gene": _d_gene(ref, chname, junc) if chname in ("TRB", "TRD", "IGH") else "",
                         "j_gene": ref.names[j] if j >= 0 else "", "c_gene": ref.names[c] if c >= 0 else "",
                         "cdr3": aa, "cdr3_nt": junc, "productive": bool(prod),
                         "full_length": bool(v >= 0 and j >= 0), "junction_umis": ju,
                         "umis": int(ju + extra[p]), "reads": nreads})
    if not rows:
        return pd.DataFrame(columns=CONTIG_COLUMNS)
    df = pd.DataFrame(rows)
    df["barcode"] = [b + "-1" for b in S.decode(ubc[df["bcix"].to_numpy()], cb_len)]
    top = df.groupby(["bcix", "chain"])["umis"].transform("max")
    # read support of a contig: mates per UMI against the library's typical depth (read-weighted median)
    rpu = df["reads"] / df["umis"].clip(lower=1)
    srt = np.sort(rpu.to_numpy())
    w = np.cumsum(srt * df["umis"].to_numpy()[np.argsort(rpu.to_numpy())])
    med = float(srt[np.searchsorted(w, w[-1] / 2)]) if len(srt) and w[-1] > 0 else 0.0
    df["high_confidence"] = ((df["junction_umis"] >= params.min_junction_umis) & (df["umis"] >= params.min_umis)
                             & (df["reads"] >= params.min_contig_reads) & (rpu >= params.contig_read_frac * med))
    df["kept"] = df["high_confidence"] & (df["umis"] >= params.min_frac * top)
    # error variants: a junction within 2 substitutions of a stronger one of the same cell and chain (a second real
    # chain differs in V, J and N region, i.e. at many positions)
    df = df.sort_values(["bcix", "chain", "umis", "reads"], ascending=[True, True, False, False]).reset_index(drop=True)
    var = np.zeros(len(df), bool)
    for _, idx in df.groupby(["bcix", "chain"]).indices.items():
        if len(idx) < 2:
            continue
        seqs = df["cdr3_nt"].to_numpy()[idx]
        for a_ in range(1, len(idx)):
            for b_ in range(a_):
                if not var[idx[b_]] and len(seqs[a_]) == len(seqs[b_]) and \
                        sum(x != y for x, y in zip(seqs[a_], seqs[b_])) <= 2:
                    var[idx[a_]] = True
                    break
    df.loc[var, "kept"] = False
    df = df.sort_values(["bcix", "chain", "umis"], ascending=[True, True, False]).reset_index(drop=True)
    df["contig_id"] = df["barcode"] + "_contig_" + (df.groupby("bcix").cumcount() + 1).astype(str)
    return df[CONTIG_COLUMNS]
