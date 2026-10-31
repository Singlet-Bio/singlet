# SPDX-License-Identifier: MIT
"""Long-read USA truth for PBMC-E pbmc1 (rule: TRUTH_RULE.md, registered before any tool output was read).

  truth_usa.py b      OUT_DIR   [--workers N]   R2-projected class of every T1 read of truth.v2.parquet -> parts + molecules.npz (gene, TRUTH-R2)
  truth_usa.py a      OUT_DIR   [--workers N]   full long-read class: scan the minimap2 BAMs of both platforms for the T1 molecules
  truth_usa.py final  OUT_DIR                   molecules.parquet (CB, UB, gene, classes) + counts
Read-only on /mnt/.../ground-truth (never written). Molecule key = CB (16 nt, 2 bit) << 22 | UMI (11 nt, 2 bit).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

GT = Path("/mnt/projects/debruinz_project/debruinz/ground-truth")
TRUTH = GT / "truth/PBMC-E/pbmc1/truth.v2.parquet"
GTF = Path("/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/genes/genes.gtf")
CHUNKS = ([("pb", GT / f"work/pbmc1/pacbio/chunk_{i:03d}/aligned.bam") for i in range(8)]
          + [("ont", GT / f"work/pbmc1/ont/aln/chunk_{i:03d}/aligned.bam") for i in range(13)])
MIN_INT, TOL = 10, 2   # frozen in TRUTH_RULE.md
U, S, A = 1, 0, 2      # truth_label convention of the ground-truth study
PRI = {U: 3, S: 2, A: 1}
LUT = np.full(256, 255, np.uint8)
for i, c in enumerate(b"ACGT"):
    LUT[c] = i


def enc(strs: list[str], n: int) -> np.ndarray:
    """exact 2-bit code of equal-length ACGT strings; -1 for any with an N or a different length."""
    ok = np.fromiter((len(s) == n for s in strs), bool, len(strs))
    buf = np.frombuffer("".join(s if len(s) == n else "A" * n for s in strs).encode(), np.uint8).reshape(-1, n)
    d = LUT[buf].astype(np.int64)
    ok &= (d < 4).all(axis=1)
    v = np.zeros(len(strs), np.int64)
    for j in range(n):
        v = (v << 2) | np.minimum(d[:, j], 3)
    v[~ok] = -1
    return v


def dec(v: np.ndarray, n: int) -> list[str]:
    d = np.empty((len(v), n), np.uint8)
    for j in range(n):
        d[:, n - 1 - j] = np.frombuffer(b"ACGT", np.uint8)[(v >> (2 * j)) & 3]
    return [r.tobytes().decode() for r in d]


def key1(cb: str, ub: str) -> int:
    t = str.maketrans("ACGT", "0123")
    try:
        return (int(cb.translate(t), 4) << 22) | int(ub.translate(t), 4)
    except ValueError:
        return -1


# ------------------------------------------------------------------------------------------------ gene models
class Models:
    def __init__(self, gtf: Path):
        t0 = time.time()
        rxg, rxt = re.compile(r'gene_id "([^"]+)"'), re.compile(r'transcript_id "([^"]+)"')
        tx: dict[tuple[str, str], list[tuple[int, int]]] = {}
        ginfo: dict[str, tuple[str, str]] = {}
        with open(gtf) as f:
            for ln in f:
                if ln[0] == "#":
                    continue
                p = ln.split("\t", 8)
                if p[2] != "exon":
                    continue
                gid = self.canon(rxg.search(p[8]).group(1))
                tx.setdefault((gid, rxt.search(p[8]).group(1)), []).append((int(p[3]) - 1, int(p[4])))
                ginfo[gid] = (p[0], p[6])
        self.gids = sorted(ginfo)
        self.idx = {g: i for i, g in enumerate(self.gids)}
        self.chrom = [ginfo[g][0] for g in self.gids]
        # same-strand exon union per (chrom, strand)
        ex: dict[tuple[str, str], list[tuple[int, int]]] = {}
        introns: dict[str, list[tuple[int, int]]] = {}
        annot: dict[str, set] = {}
        for (gid, _t), exs in tx.items():
            exs.sort()
            ex.setdefault(ginfo[gid], []).extend(exs)
            for (s0, e0), (s1, e1) in zip(exs, exs[1:]):
                if s1 > e0:
                    introns.setdefault(gid, []).append((e0, s1))
                    annot.setdefault(gid, set()).add((e0, s1))
        merged = {k: self.merge(v) for k, v in ex.items()}
        es = {k: (np.array([a for a, _ in v], np.int64), np.array([b for _, b in v], np.int64)) for k, v in merged.items()}
        n = len(self.gids)
        self.is_ = [None] * n
        self.ie_ = [None] * n
        self.annot = [annot.get(g, set()) for g in self.gids]
        for g in self.gids:
            i = self.idx[g]
            out = []
            Es, Ee = es[ginfo[g]]
            for a, b in self.merge(introns.get(g, [])):
                j = int(np.searchsorted(Ee, a, "right"))
                cur = a
                while j < len(Es) and Es[j] < b:
                    if Es[j] > cur:
                        out.append((cur, Es[j]))
                    cur = max(cur, int(Ee[j]))
                    j += 1
                if cur < b:
                    out.append((cur, b))
            self.is_[i] = np.array([a for a, _ in out], np.int64)
            self.ie_[i] = np.array([b for _, b in out], np.int64)
        print(f"[models] {n} genes, {len(tx)} transcripts, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    @staticmethod
    def canon(g: str) -> str:
        m = re.match(r"(ENSG\d{11})(?:\.\d+)?(_PAR_Y)?", g)
        return (m.group(1) + (m.group(2) or "")) if m else g.split(".")[0]

    @staticmethod
    def merge(iv):
        out = []
        for a, b in sorted(iv):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return [(a, b) for a, b in out]

    def n_int(self, g: int, blocks) -> int:
        Is, Ie = self.is_[g], self.ie_[g]
        n = 0
        if len(Is) == 0:
            return 0
        for s, e in blocks:
            j = int(np.searchsorted(Ie, s, "right"))
            while j < len(Is) and Is[j] < e:
                n += min(e, int(Ie[j])) - max(s, int(Is[j]))
                j += 1
        return n

    def n_annot(self, g: int, juncs) -> int:
        a = self.annot[g]
        c = 0
        for js, je in juncs:
            if (js, je) in a:
                c += 1
                continue
            for ds in range(-TOL, TOL + 1):
                if any((js + ds, je + de) in a for de in range(-TOL, TOL + 1)):
                    c += 1
                    break
        return c

    def classify(self, g: int, blocks, juncs) -> tuple[int, int, int]:
        ni = self.n_int(g, blocks)
        na = self.n_annot(g, juncs) if juncs else 0
        return (U if ni >= MIN_INT else S if na > 0 else A), ni, na


_M: Models | None = None


def models() -> Models:
    global _M
    if _M is None:
        _M = Models(GTF)
    return _M


# ------------------------------------------------------------------------------------------------ b: R2 truth
def work_b(args):
    rg, outdir = args
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    M = models()
    cols = ["CB", "UB", "tier", "has_truth", "true_gene", "block_starts", "block_ends", "junc_starts", "junc_ends"]
    t = pq.ParquetFile(TRUTH).read_row_group(rg, columns=cols)
    t = t.filter(pc.and_(pc.equal(t["tier"], 1), t["has_truth"]))
    d = t.to_pydict()
    key = np.maximum(enc(d["CB"], 16), -1)
    ubk = enc(d["UB"], 11)
    keys = np.where((key >= 0) & (ubk >= 0), (key << 22) | np.maximum(ubk, 0), -1)
    n = len(keys)
    gi = np.full(n, -1, np.int32)
    cls = np.full(n, -1, np.int8)
    nint = np.zeros(n, np.int32)
    nann = np.zeros(n, np.int16)
    for i in range(n):
        g = M.idx.get(d["true_gene"][i] or "", -1)
        if g < 0 or keys[i] < 0:
            continue
        bs, be = d["block_starts"][i], d["block_ends"][i]
        js, je = d["junc_starts"][i], d["junc_ends"][i]
        c, ni, na = M.classify(g, list(zip(bs, be)), list(zip(js, je)))
        gi[i], cls[i], nint[i], nann[i] = g, c, ni, na
    ok = gi >= 0
    np.savez(Path(outdir) / f"b_{rg:04d}.npz", key=keys[ok], gene=gi[ok], cls=cls[ok], nint=nint[ok], nann=nann[ok])
    return rg, int(n), int(ok.sum())


def run_b(out: Path, workers: int):
    import pyarrow.parquet as pq
    nrg = min(pq.ParquetFile(TRUTH).metadata.num_row_groups, int(os.environ.get('LR_RG', 10**9)))  # LR_RG: smoke tests only
    (out / "parts").mkdir(parents=True, exist_ok=True)
    models()  # build before fork
    t0 = time.time()
    tot = [0, 0]
    with Pool(workers) as p:
        for rg, n, k in p.imap_unordered(work_b, [(i, out / "parts") for i in range(nrg)]):
            tot[0] += n
            tot[1] += k
            print(f"[b] rg {rg} T1 reads {n} classified {k} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    parts = [np.load(f) for f in sorted((out / "parts").glob("b_*.npz"))]
    key = np.concatenate([p["key"] for p in parts])
    gene = np.concatenate([p["gene"] for p in parts])
    cls = np.concatenate([p["cls"] for p in parts])
    nint = np.concatenate([p["nint"] for p in parts])
    del parts
    # molecule gene = strict mode of the reads' genes
    o = np.lexsort((gene, key))
    key, gene, cls, nint = key[o], gene[o], cls[o], nint[o]
    run_start = np.r_[True, (key[1:] != key[:-1]) | (gene[1:] != gene[:-1])]
    rs = np.flatnonzero(run_start)
    run_key, run_gene, run_n = key[rs], gene[rs], np.diff(np.r_[rs, len(key)])
    o2 = np.lexsort((-run_n, run_key))
    run_key, run_gene, run_n = run_key[o2], run_gene[o2], run_n[o2]
    first = np.r_[True, run_key[1:] != run_key[:-1]]
    fi = np.flatnonzero(first)
    nxt = np.minimum(fi + 1, len(run_key) - 1)
    tie = (run_key[nxt] == run_key[fi]) & (run_n[nxt] == run_n[fi]) & (fi + 1 < len(run_key))
    mk, mg = run_key[fi][~tie], run_gene[fi][~tie]
    # molecule class from the reads of that gene
    pos = np.searchsorted(mk, key)
    pos = np.minimum(pos, len(mk) - 1)
    keep = (mk[pos] == key) & (mg[pos] == gene)
    pri = np.array([0, 3, 2, 1], np.int8)  # cls -1.. mapped below
    pr = np.where(cls == U, 3, np.where(cls == S, 2, 1)).astype(np.int8)
    mp = np.zeros(len(mk), np.int8)
    np.maximum.at(mp, pos[keep], pr[keep])
    mn = np.zeros(len(mk), np.int32)
    np.maximum.at(mn, pos[keep], nint[keep])
    nr = np.bincount(pos[keep], minlength=len(mk)).astype(np.int32)
    clsB = np.where(mp == 3, U, np.where(mp == 2, S, A)).astype(np.int8)
    np.savez(out / "molecules.npz", key=mk, gene=mg.astype(np.int32), clsB=clsB, nintB=mn, nreadsB=nr)
    json.dump({"t1_reads": tot[0], "t1_reads_classified": tot[1], "molecules": int(len(mk)), "tie_dropped": int(tie.sum()),
               "R2_class_counts": {k: int((clsB == v).sum()) for k, v in (("S", S), ("U", U), ("A", A))},
               "seconds": round(time.time() - t0)}, open(out / "b_summary.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------ a: long-read scan
def work_a(args):
    plat, bam, outdir, ci = args
    import pysam
    M = models()
    mol = np.load(Path(outdir) / "molecules.npz")
    mk, mg = mol["key"], mol["gene"]
    T = str.maketrans("ACGT", "0123")
    ks, gs, nints, nanns, njs = [], [], [], [], []
    c = {"records": 0, "primary": 0, "no_tag": 0, "in_set": 0, "wrong_chrom": 0}
    t0 = time.time()
    with pysam.AlignmentFile(str(bam), "rb", check_sq=False, threads=2) as f:
        for r in f.fetch(until_eof=True):
            c["records"] += 1
            if c["records"] > int(os.environ.get('LR_REC', 10**12)):  # smoke tests only
                break
            if r.flag & 0x904:
                continue
            c["primary"] += 1
            try:
                cb, ub = r.get_tag("CB"), r.get_tag("UB")[:11]  # long-read UMIs are 12 nt; the study truncates to 11 (D-31)
            except KeyError:
                c["no_tag"] += 1
                continue
            try:
                k = (int(cb.split("-")[0].translate(T), 4) << 22) | int(ub.translate(T), 4)
            except ValueError:
                continue
            j = int(np.searchsorted(mk, k))
            if j >= len(mk) or mk[j] != k:
                continue
            c["in_set"] += 1
            g = int(mg[j])
            if M.chrom[g] != r.reference_name:
                c["wrong_chrom"] += 1
                continue
            blocks, juncs = [], []
            pos = r.reference_start
            bs = pos
            for op, ln in r.cigartuples or []:
                if op in (0, 2, 7, 8):
                    pos += ln
                elif op == 3:
                    blocks.append((bs, pos))
                    juncs.append((pos, pos + ln))
                    pos += ln
                    bs = pos
            blocks.append((bs, pos))
            ks.append(k); gs.append(g)
            nints.append(M.n_int(g, blocks))
            nanns.append(M.n_annot(g, juncs) if juncs else 0)
            njs.append(len(juncs))
    np.savez(Path(outdir) / "parts" / f"a_{plat}_{ci:03d}.npz", key=np.array(ks, np.int64), gene=np.array(gs, np.int32),
             nint=np.array(nints, np.int32), nann=np.array(nanns, np.int16), nj=np.array(njs, np.int16))
    c["scored"] = len(ks)
    c["seconds"] = round(time.time() - t0)
    return plat, ci, c


def run_a(out: Path, workers: int):
    models()
    jobs = [(p, b, out, i) for i, (p, b) in enumerate(CHUNKS)]
    stats = []
    with Pool(workers) as pool:
        for plat, ci, c in pool.imap_unordered(work_a, jobs):
            print(f"[a] {plat} chunk {ci}: {c}", file=sys.stderr, flush=True)
            stats.append({"platform": plat, "chunk": ci, **c})
    json.dump(stats, open(out / "a_summary.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------ final
def run_final(out: Path):
    import pandas as pd
    mol = np.load(out / "molecules.npz")
    mk, mg, clsB = mol["key"], mol["gene"], mol["clsB"]
    n = len(mk)
    maxint = {"pb": np.full(n, -1, np.int32), "ont": np.full(n, -1, np.int32)}   # -1 = no read of that platform
    anyS = {"pb": np.zeros(n, bool), "ont": np.zeros(n, bool)}
    nrec = {"pb": np.zeros(n, np.int32), "ont": np.zeros(n, np.int32)}
    for f in sorted((out / "parts").glob("a_*.npz")):
        plat = f.name.split("_")[1]
        p = np.load(f)
        j = np.searchsorted(mk, p["key"])
        np.maximum.at(maxint[plat], j, p["nint"])
        np.logical_or.at(anyS[plat], j, p["nann"] > 0)
        np.add.at(nrec[plat], j, 1)

    def cls_of(mi, sv, thr):
        has = mi >= 0
        return np.where(~has, -1, np.where(mi >= thr, U, np.where(sv, S, A))).astype(np.int8)

    mi_all = np.maximum(maxint["pb"], maxint["ont"])
    any_all = anyS["pb"] | anyS["ont"]
    df = pd.DataFrame({"CB": dec(mk >> 22, 16), "UB": dec(mk & ((1 << 22) - 1), 11), "gene_idx": mg, "clsR2": clsB,
                       "nintR2": mol["nintB"], "nR2": mol["nreadsB"],
                       "clsLR": cls_of(mi_all, any_all, 10), "clsLR5": cls_of(mi_all, any_all, 5), "clsLR20": cls_of(mi_all, any_all, 20),
                       "clsLR_pb": cls_of(maxint["pb"], anyS["pb"], 10), "clsLR_ont": cls_of(maxint["ont"], anyS["ont"], 10),
                       "nintLR": mi_all, "n_pb": nrec["pb"], "n_ont": nrec["ont"]})
    gids = models().gids
    df["gene"] = [gids[i] for i in df.gene_idx]
    df.drop(columns="gene_idx").to_parquet(out / "molecules.parquet", index=False)
    summ = {"molecules": int(n)}
    for c in ("clsR2", "clsLR", "clsLR5", "clsLR20", "clsLR_pb", "clsLR_ont"):
        summ[c] = {k: int((df[c] == v).sum()) for k, v in (("none", -1), ("S", S), ("U", U), ("A", A))}
    ok = df.clsLR >= 0
    summ["confusion_R2_rows_LR_cols"] = {f"{a}->{b}": int(((df.clsR2 == av) & (df.clsLR == bv) & ok).sum())
                                         for a, av in (("S", S), ("U", U), ("A", A)) for b, bv in (("S", S), ("U", U), ("A", A))}
    summ["unspliced_share"] = {"LR": float((df.clsLR[ok] == U).mean()), "R2": float((df.clsR2[ok] == U).mean())}
    json.dump(summ, open(out / "truth_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["b", "a", "final"]), ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    {"b": lambda: run_b(out, a.workers), "a": lambda: run_a(out, a.workers), "final": lambda: run_final(out)}[a.step]()
