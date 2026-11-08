"""B12: where does the Singlet V(D)J assembler lose the reference chains of reference-paired B cells?

    python -P bcr_stage_loss.py LIB REF_TSV OUT_JSON

Runs ``run_library`` on LIB (development libraries only) with hooks that keep the per-mate arrays at three points
(before the molecule read-support cut, before the one-chain-per-molecule cut, before the groups are assembled), and
re-runs ``assemble_group`` on the groups that should hold a missed chain with a copy of the kernel that also returns its
seeds, seed support, seed cut and terminal votes. For every chain of a reference-paired cell (Cell Ranger truth) that the
run gets wrong, it reports the first stage that drops the reference junction:

  outcome   ok | outranked (kept, another contig of the chain has more UMIs) | not_kept (assembled, filtered)
            | near (a contig of the chain within 3 edits) | absent (no contig with the junction)
  stage     (absent and near only) no_mates | barcode | umi_support | other_chain | barcode_not_assembled | no_group
            | seed_absent | seed_no_anchor | seed_weak | terminal_absent | terminal_weak | path_gap | path_pruned
            | path_early_terminal | path_limits
"""
import collections
import inspect
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd

W = "/mnt/projects/debruinz_project/debruinz/singlet"
SRC = os.environ["SRC"]
sys.path.insert(0, SRC + "/bench/vdj")
import compare_vdj as CV  # noqa: E402
from numba import njit  # noqa: E402

from singlet.adt import count as AC  # noqa: E402
from singlet.adt import seq as S  # noqa: E402
from singlet.vdj import _kernels as K  # noqa: E402
from singlet.vdj import assemble as A  # noqa: E402

KA = A.K_ASM
LIB, REFTSV, OUT = sys.argv[1:4]
D = f"{W}/data/vdjc/{LIB}"
GEXC = f"{W}/expression-sota/vdj2/runs/{LIB}/gex_cells.txt"

# ---------- instrumented run_library (source copy with hooks; same code otherwise) ----------
HOOKS = {}
src = textwrap.dedent(inspect.getsource(A.run_library))
for anchor, name in (
    ("    keep_m = good[mol]\n", "umi"),
    ('    st["mates_other_chain"]', "chain"),
    ("    groups = []\n", "grp"),
):
    assert src.count(anchor) == 1, anchor
    src = src.replace(anchor, f"    HOOKS[{name!r}] = dict(locals())\n" + anchor)
ns = dict(A.__dict__)
ns["HOOKS"] = HOOKS
exec(compile(src, "run_library_diag", "exec"), ns)
run_library = ns["run_library"]

# ---------- assemble_group copy that also returns its internals ----------
ksrc = textwrap.dedent(inspect.getsource(K.assemble_group.py_func))
ksrc = ksrc.split("\n", 1)[1]  # drop the @njit line
extra = ", seeds, ssup, seed_min, tu, tc, tthr, np_"
ksrc = ksrc.replace("return paths[:0], plen[:0], psup[:0], cand", "return paths[:0], plen[:0], psup[:0], cand" + extra)
ksrc = ksrc.replace("return paths[:np_], plen[:np_], psup[:np_], cand", "return paths[:np_], plen[:np_], psup[:np_], cand" + extra)
ksrc = ksrc.replace("def assemble_group(", "def assemble_group_diag(")
kns = dict(K.__dict__)
exec(compile(ksrc, "assemble_group_diag", "exec"), kns)
assemble_group_diag = njit(kns["assemble_group_diag"])


def pack(s):
    v = 0
    for ch in s:
        v = (v << 2) | "ACGT".index(ch)
    return v


@njit(cache=False)
def first_hit(codes, off, k, tk):
    """per sequence row: index into sorted target k-mers tk of the first hit, or -1."""
    n = len(off) - 1
    out = np.full(n, -1, np.int64)
    mask = np.uint64((1 << (2 * k)) - 1)
    for r in range(n):
        cur = np.uint64(0)
        valid = 0
        for i in range(off[r], off[r + 1]):
            c = codes[i]
            if c > 3:
                valid = 0
                cur = np.uint64(0)
                continue
            cur = ((cur << np.uint64(2)) | np.uint64(c)) & mask
            valid += 1
            if valid >= k:
                j = np.searchsorted(tk, cur)
                if j < len(tk) and tk[j] == cur:
                    out[r] = j
                    break
    return out


def edit_le(a, b, k=3):
    if abs(len(a) - len(b)) > k:
        return False
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1] <= k


# ---------- run ----------
import dnaio  # noqa: E402
import glob  # noqa: E402

r1 = sorted(glob.glob(f"{D}/fastqs/**/*_R1_*.fastq.gz", recursive=True))
r2 = [p.replace("_R1_", "_R2_") for p in r1]
with dnaio.open(r1[0]) as f:
    r1_len = len(next(iter(f)).sequence)
umi_len = 12 if r1_len in (28, 29, 30, 31) else 10
wl = AC.load_barcode_list(f"{W}/pipeline-benchmark/whitelists/737K-august-2016.txt")
wl_codes, bad = S.encode_strings(wl, 16)
wl_codes = np.sort(wl_codes[bad == 0])
ref = A.Reference.load(REFTSV)
params = A.Params(threads=8)
res = run_library(r1, r2, ref=ref, cb_len=16, umi_len=umi_len, whitelist_codes=wl_codes, params=params,
                  log=lambda *x: print(*x, flush=True))
ct = res.contigs.copy()
ct["barcode"] = ct.barcode.astype(str)

# ---------- reference-paired cells and outcome per chain ----------
refc = CV.load_cellranger(f"{D}/filtered_contig_annotations.csv")
refc = refc[refc.chain.isin(["IGH", "IGK", "IGL"])]
gex = set(l.strip() for l in open(GEXC) if l.strip()) if os.path.exists(GEXC) else None
rt = refc.sort_values("umis", ascending=False, kind="mergesort").drop_duplicates(["barcode", "chain"])
rtop = {(b, c): (n, u) for b, c, n, u in zip(rt.barcode, rt.chain, rt.cdr3_nt, rt.umis)}
kept = ct[ct.kept.astype(bool) & ct.productive.astype(bool)]
if gex is not None:
    kept = kept[kept.barcode.isin(gex)]
mt = kept.sort_values("umis", ascending=False, kind="mergesort").drop_duplicates(["barcode", "chain"])
mtop = dict(zip(zip(mt.barcode, mt.chain), mt.cdr3_nt))
by_bc = {k: g for k, g in ct.groupby(["barcode", "chain"])}
targets = []  # failing chains
n_cells = n_ok = 0
for bc in sorted({b for b, _ in rtop}):
    for x, y in (("IGH", "IGK"), ("IGH", "IGL")):
        if (bc, x) in rtop and (bc, y) in rtop:
            n_cells += 1
            good = True
            for ch in (x, y):
                T, u = rtop[(bc, ch)]
                if mtop.get((bc, ch)) == T:
                    continue
                good = False
                g = by_bc.get((bc, ch))
                out = "absent"
                if g is not None:
                    hit = g[g.cdr3_nt == T]
                    if len(hit):
                        out = "outranked" if bool(hit.kept.iloc[0]) else "not_kept"
                        if out == "not_kept":
                            r_ = hit.iloc[0]
                            out += "_lowconf" if not bool(r_.high_confidence) else "_filter"
                    elif any(edit_le(T, s) for s in g.cdr3_nt):
                        out = "near"
                targets.append(dict(barcode=bc, chain=ch, cdr3_nt=T, ref_umis=float(u), outcome=out))
            n_ok += good
            break
print(f"[diag] {LIB} paired cells {n_cells} correct {n_ok} ({n_ok / max(1, n_cells):.4f}); failing chains {len(targets)}",
      flush=True)

# ---------- stage diagnosis for absent / near ----------
H_umi, H_ch, H_g = HOOKS["umi"], HOOKS["chain"], HOOKS["grp"]
mm = H_umi["mm"]
seq_row = H_umi["seq_row"]
cb_raw = H_umi["cb"]
bc_code = H_umi["bc_code"]
seqc, soff = H_g["seqc"], H_g["soff"]
row2mate = np.flatnonzero(seq_row >= 0)  # seq row -> mate index (mm space)
diag = [t for t in targets if t["outcome"] in ("absent", "near")]
mids = {}
for i, t in enumerate(diag):
    T = t["cdr3_nt"]
    L = len(T)
    mids.setdefault(pack(T[L // 2 - KA // 2 : L // 2 - KA // 2 + KA]), []).append(i)
tk = np.array(sorted(mids), dtype=np.uint64)
hit_row = first_hit(seqc, soff, KA, tk)
rows_hit = np.flatnonzero(hit_row >= 0)
mates_hit = row2mate[rows_hit]
tgt_of_mate = {}
for r_, m_ in zip(rows_hit, mates_hit):
    tgt_of_mate[m_] = mids[int(tk[hit_row[r_]])]
chain_names = ref.chains
sel0, good = H_umi["sel"], H_umi["good"]
mol0 = H_umi["mol"]
sel1, same = H_ch["sel"], H_ch["same"]
selF, molF, cellF, asm_bc, ubc = H_g["sel"], H_g["mol"], H_g["cell_of"], H_g["asm_bc"], H_g["ubc"]


class Pos:
    """index of a mate in a sorted selection array (sel arrays are increasing)."""

    def __init__(self, a):
        self.a = a

    def __contains__(self, m):
        i = np.searchsorted(self.a, m)
        return i < len(self.a) and self.a[i] == m

    def __getitem__(self, m):
        return int(np.searchsorted(self.a, m))


assert np.all(np.diff(sel0) > 0) and np.all(np.diff(sel1) > 0)
pos0, pos1 = Pos(sel0), Pos(sel1)
ubc_str = np.array([b + "-1" for b in S.decode(ubc, 16)])
bc_index = {b: i for i, b in enumerate(ubc_str)}
code_of = {}
for t in diag:
    c, bad_ = S.encode_strings([t["barcode"].split("-")[0]], 16)
    code_of[t["barcode"]] = int(c[0])
cnt = collections.defaultdict(lambda: collections.Counter())
for m_, lst in tgt_of_mate.items():
    raw = cb_raw[m_].decode() + "-1"
    for i in lst:
        t = diag[i]
        if raw != t["barcode"] and int(bc_code[m_]) != code_of[t["barcode"]]:
            continue
        c = cnt[i]
        c["mates_raw_bc"] += 1
        if m_ not in pos0:
            c["lost_barcode_or_umi_seq"] += 1
            continue
        if int(bc_code[m_]) != code_of[t["barcode"]]:
            c["lost_barcode"] += 1
            continue
        if not good[mol0[pos0[m_]]]:
            c["lost_umi_support"] += 1
            continue
        if m_ not in pos1 or not same[pos1[m_]]:
            c["lost_other_chain"] += 1
            continue
        if chain_names[int(mm[m_, 9])] != t["chain"]:
            c["lost_other_chain"] += 1
            continue
        c["mates_in_group"] += 1

# groups: (barcode index, chain) -> (rows in sel space)
gi, starts, ends, gkey = H_g["gi"], H_g["starts"], H_g["ends"], H_g["gkey"]
gstart = {int(gkey[s]): (s, e) for s, e in zip(starts, ends)}
m_sel, s_row = H_g["m_sel"], H_g["s_row"]


def group_inputs(bcix, ch):
    se = gstart.get(bcix * 8 + ch)
    if se is None:
        return None
    rows = gi[se[0] : se[1]]
    um_local, um_inv = np.unique(molF[rows], return_inverse=True)
    return rows, s_row[rows], um_inv.ravel().astype(np.int32), m_sel[rows, 10], m_sel[rows, 11]


def walk(T, reads, umis, term_set):
    """follow the reference junction from its seed with the kernel's per-UMI vote; first failure or 'complete'."""
    need = set()
    for i in range(KA, len(T)):
        ctx = T[i - KA + 1 : i]
        for b in "ACGT":
            need.add(ctx + b)
    need.add(T[:KA])
    km = collections.defaultdict(collections.Counter)
    for s, u in zip(reads, umis):
        for p in range(len(s) - KA + 1):
            w = s[p : p + KA]
            if w in need:
                km[w][u] += 1
    for i in range(KA, len(T) + 1):
        if T[i - KA : i] in term_set and i < len(T):
            return "path_early_terminal", i
        if i == len(T):
            return "complete", i
        ctx = T[i - KA + 1 : i]
        best = {}
        for bi, b in enumerate("ACGT"):
            for u, c in km.get(ctx + b, {}).items():
                o = best.get(u)
                if o is None or c > o[0]:
                    best[u] = [c, bi, False]
                elif c == o[0] and o[1] != bi:
                    o[2] = True
        votes = np.zeros(4, int)
        for c, bi, tie in best.values():
            if not tie:
                votes[bi] += 1
        tb = "ACGT".index(T[i])
        if T[i - KA + 1 : i + 1] not in km:
            return "path_gap", i
        bb = int(np.argmax(votes))
        if votes[bb] == 0:
            rr = [sum(km.get(ctx + b, {}).values()) for b in "ACGT"]
            if int(np.argmax(rr)) != tb:
                return "path_pruned", i
            continue
        if bb != tb:
            thr = max(params.min_branch_umis, params.branch_frac * votes[bb])
            if votes[tb] < thr:
                return "path_pruned", i
    return "complete", len(T)


rows_out = []
for i, t in enumerate(diag):
    c = cnt.get(i, collections.Counter())
    T = t["cdr3_nt"]
    rec = dict(t, **{k: int(v) for k, v in c.items()})
    ch = chain_names.index(t["chain"])
    bcix = bc_index.get(t["barcode"])
    if c["mates_raw_bc"] == 0:
        rec["stage"] = "no_mates"
    elif c["mates_in_group"] == 0:
        lost = {k: c[k] for k in ("lost_barcode_or_umi_seq", "lost_barcode", "lost_umi_support", "lost_other_chain")}
        rec["stage"] = {"lost_barcode_or_umi_seq": "barcode", "lost_barcode": "barcode", "lost_umi_support": "umi_support",
                        "lost_other_chain": "other_chain"}[max(lost, key=lost.get)]
    elif bcix is None or not asm_bc[bcix]:
        rec["stage"] = "barcode_not_assembled"
    else:
        gin = group_inputs(bcix, ch)
        if gin is None:
            rec["stage"] = "no_group"
        else:
            rows, ridx, um, cys, fw = gin
            out = assemble_group_diag(seqc, soff, ridx, um, cys, fw, KA, params.min_seed_umis, params.seed_frac,
                                      params.min_branch_umis, params.branch_frac, A.MAX_PATHS, A.MAX_JUNCTION)
            paths, plen, psup, cand, seeds, ssup, seed_min, tu, tc, tthr, npaths = out
            pstr = ["".join("ACGT"[b] for b in paths[p, : plen[p]]) for p in range(len(plen))]
            rec.update(group_reads=int(len(ridx)), group_umis=int(um.max() + 1) if len(um) else 0, n_paths=int(npaths),
                       seed_min=int(seed_min), tthr=int(tthr))
            reads = ["".join("ACGT"[x] if x < 4 else "N" for x in seqc[soff[r] : soff[r + 1]]) for r in ridx]
            sk, tkm = pack(T[:KA]), pack(T[-KA:])
            si = np.flatnonzero(seeds == np.uint64(sk))
            ti = np.flatnonzero(tu == np.uint64(tkm))
            rec["seed_umis"] = int(ssup[si[0]]) if len(si) else -1
            rec["term_votes"] = int(tc[ti[0]]) if len(ti) else -1
            rec["seed_reads"] = sum(T[:KA] in s for s in reads)
            rec["term_reads"] = sum(T[-KA:] in s for s in reads)
            # read-level anchors: of the reads holding the true seed / terminal k-mer, the V / J call and the anchor offset
            # (anchor position minus the k-mer's true codon position; -999 no anchor)
            mrows = m_sel[rows]
            sv, tv = collections.Counter(), collections.Counter()
            sg, tg = collections.Counter(), collections.Counter()
            for s_, mr in zip(reads, mrows):
                p_ = s_.find(T[:KA])
                if p_ >= 0:
                    sv[int(mr[10]) - p_ if mr[10] >= 0 else -999] += 1
                    sg[ref.names[mr[0]] if mr[0] >= 0 else "-"] += 1
                p_ = s_.find(T[-KA:])
                if p_ >= 0:
                    tv[int(mr[11]) - (p_ + KA - 3) if mr[11] >= 0 else -999] += 1
                    tg[ref.names[mr[3]] if mr[3] >= 0 else "-"] += 1
            rec["seed_anchor_off"] = dict(sv.most_common(4))
            rec["seed_v"] = dict(sg.most_common(3))
            rec["term_anchor_off"] = dict(tv.most_common(4))
            rec["term_j"] = dict(tg.most_common(3))
            rec["term_next"] = collections.Counter(s_[s_.find(T[-KA:]) + KA : s_.find(T[-KA:]) + KA + 12]
                                                   for s_ in reads if T[-KA:] in s_).most_common(2)
            if T in pstr:
                rec["stage"] = "path_found"  # should not happen for absent
            elif not len(si):
                rec["stage"] = "seed_no_anchor" if rec["seed_reads"] else "seed_absent"
            elif ssup[si[0]] < seed_min:
                rec["stage"] = "seed_weak"
            elif not len(ti):
                rec["stage"] = "terminal_absent" if rec["term_reads"] == 0 else "terminal_no_anchor"
            elif tc[ti[0]] < tthr:
                rec["stage"] = "terminal_weak"
            else:
                term_set = {"".join("ACGT"[(int(x) >> (2 * (KA - 1 - j))) & 3] for j in range(KA))
                            for x, v in zip(tu, tc) if v >= tthr}
                st_, at = walk(T, reads, um.tolist(), term_set)
                rec["walk_pos"] = int(at)
                rec["stage"] = "path_limits" if st_ == "complete" else st_
                if npaths >= A.MAX_PATHS:
                    rec["max_paths_hit"] = True
            rec["paths"] = pstr[:6]
    rows_out.append(rec)

summary = {
    "lib": LIB,
    "paired_cells": n_cells,
    "correct": n_ok,
    "failing_chains": len(targets),
    "outcome": collections.Counter(t["outcome"] for t in targets),
    "stage_absent": collections.Counter(r["stage"] for r in rows_out if r["outcome"] == "absent"),
    "stage_near": collections.Counter(r["stage"] for r in rows_out if r["outcome"] == "near"),
}
print(json.dumps(summary, indent=1), flush=True)
json.dump({"summary": summary, "targets": targets, "diag": rows_out, "run_stats": res.stats}, open(OUT, "w"), indent=1,
          default=str)
