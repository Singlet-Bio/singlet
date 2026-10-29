"""Runs one protocol class through the GPU barcode layer and compares it, read by read, with
  (1) the Python/numba reference (src/engine_gpu/barcode_layer_ref.py),
  (2) for the 10x class, W3b's R11 CbCorrector (cb_check_cpu, the CPU code the unsorted stream uses),
  (3) STAR's own CB / UB / UR tags from a STAR BAM of the same reads (bench/barcode_layer/jobs/star_class.sbatch).
Real reads only: the FASTQs come from bench/barcode_layer/jobs/fetch.sbatch. Prints a report and writes <out>/<class>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src" / "engine_gpu"))
import bl_io as io  # noqa: E402
import barcode_layer_ref as ref  # noqa: E402

PB_WL = "/mnt/projects/debruinz_project/debruinz/singlet/pipeline-benchmark/whitelists"
DATA = "/mnt/projects/debruinz_project/debruinz/singlet/data/barcode-layer"

# bc: file of the barcode mate (read names r<i>). qual: 'F' (singlet writes R1 with quality F) or 'file' (the sequencer's).
CLASSES = {
    "pbmc1k_v3_F": dict(kind="simple", dir="pbmc1k_v3", bc="R1.fq", cb=(0, 16), umi=(16, 12), wl="3M-february-2018.txt", qual="F"),
    "pbmc1k_v3_real": dict(kind="simple", dir="pbmc1k_v3", bc="R1.fq", cb=(0, 16), umi=(16, 12), wl="3M-february-2018.txt", qual="file"),
    "dropseq": dict(kind="nowl", dir="dropseq", bc="M1.fq", cb=(0, 12), umi=(12, 8), qual="F", min_reads=5),
    "bd": dict(kind="complex", dir="bd", bc="M1.fq", segs=["0_0_0_8", "0_21_0_29", "0_43_0_51"], umi="0_52_0_59",
               wl=["bd_cls1.txt", "bd_cls2.txt", "bd_cls3.txt"], qual="F"),
    # GSE281535 / SRR31302021 (Parse-style linkers: BC3 at 10-17, a 30-base linker, BC2, ...). 'splitseq' is singlet's protocol row as it stands
    # (BC2 at 33-40, BC1 at 56-63); 'splitseq48' is STAR's published SPLiT-seq recipe (0_10_0_17 0_48_0_55 0_78_0_85).
    "splitseq": dict(kind="complex", dir="splitseq_parse", bc="M2.fq", segs=["0_10_0_17", "0_33_0_40", "0_56_0_63"], umi="0_0_0_9",
                     wl=["splitseq_bc_rd3.txt", "splitseq_bc_rd2.txt", "splitseq_bc_rd1.txt"], qual="F"),
    "splitseq48": dict(kind="complex", dir="splitseq_parse", bc="M2.fq", segs=["0_10_0_17", "0_48_0_55", "0_78_0_85"], umi="0_0_0_9",
                       wl=["splitseq_bc_rd3.txt", "splitseq_bc_rd2.txt", "splitseq_bc_rd1.txt"], qual="F"),
    "indrop": dict(kind="complex", dir="indrop", bc="M2.fq", segs=["0_0_2_-1", "3_1_3_8"], umi="3_9_3_14",
                   wl=["indrop_bc1.txt", "indrop_bc2.txt"], adapter="GAGTGATTGCTTGTGACGCCTT", adapter_mm=1, qual="F"),
}


def sh(cmd: list[str]) -> str:
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"$ {' '.join(cmd)}  [{time.time() - t:.1f}s rc={r.returncode}]", flush=True)
    if r.stdout.strip():
        print(r.stdout.strip(), flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True)
        raise SystemExit(f"command failed: {cmd[0]}")
    return r.stdout


def prep_reads(args, fq, rd, qual_const):
    if args.reuse and os.path.exists(rd):
        return len(io.load_reads(rd)[0]) - 1
    return io.fastq_to_reads(fq, rd, qual_const=qual_const, max_reads=args.max_reads)


def run_h(args, probe, cmd):
    """Runs the harness unless --reuse and its outputs exist."""
    if args.reuse and os.path.exists(probe):
        print(f"(reusing {probe})", flush=True)
        return
    sh(args.exe.split() + cmd)


def pct(a, b):
    return round(100.0 * a / b, 4) if b else None


# ---------------------------------------------------------------------------------------------------------------------
from numba import njit  # noqa: E402


@njit(cache=False)
def resolve_pass_f64(cbs, quals, n_reads, cb_len, wl, counts, qtab, out_cell):
    """Same as ref.resolve_pass but with the posterior in double (what a MATCH_CellRanger build, or a CPU port that forgot the float, would do)."""
    cand = np.empty(3 * cb_len + 4, dtype=np.int64)
    cq = np.empty(3 * cb_len + 4, dtype=np.int64)
    for r in range(n_reads):
        v, pos_n = ref._pack(cbs[r], 0, cb_len)
        out_cell[r] = -1
        if pos_n == -2:
            continue
        if pos_n == -1:
            i = ref._bsearch(wl, v)
            if i >= 0:
                out_cell[r] = i
                continue
        n = 0
        if pos_n >= 0:
            shift = np.uint64(2 * (cb_len - 1 - pos_n))
            for jj in range(4):
                i = ref._bsearch(wl, v ^ (np.uint64(jj) << shift))
                if i >= 0:
                    cand[n] = i
                    cq[n] = quals[r, pos_n]
                    n += 1
        else:
            for ii in range(cb_len):
                for jj in range(1, 4):
                    i = ref._bsearch(wl, v ^ (np.uint64(jj) << np.uint64(2 * ii)))
                    if i >= 0:
                        cand[n] = i
                        cq[n] = quals[r, cb_len - 1 - ii]
                        n += 1
        if n == 1:
            out_cell[r] = cand[0]
        elif n > 1:
            ptot = 0.0
            pmax = 0.0
            best = -1
            for k in range(n):
                c = counts[cand[k]]
                if c > 0:
                    q = cq[k] - 33
                    if q > 33:
                        q = 33
                    pin = np.float64(c) * qtab[q]
                    ptot = ptot + pin
                    if pin > pmax:
                        best = cand[k]
                        pmax = pin
            if ptot > 0.0 and pmax >= 0.975 * ptot:
                out_cell[r] = best


def umi_check_arrays(umm: np.ndarray):
    """convertCheckUMI over a (n, L) base matrix: -23 any N, -24 homopolymer of the UMI length, else 0."""
    c = io._CODE2[umm]
    has_n = (c > 3).any(axis=1)
    homo = (c == c[:, :1]).all(axis=1) & ~has_n
    return np.where(has_n, -23, np.where(homo, -24, 0)).astype(np.int32)


def compare_simple(cfg, work, wl_path, args, rep):
    cb0, cbl = cfg["cb"]
    um0, uml = cfg["umi"]
    rd = f"{work}/reads.bin"
    n = prep_reads(args, f"{DATA}/{cfg['dir']}/{cfg['bc']}", rd, "F" if cfg["qual"] == "F" else None)
    rep["reads"] = n
    keys, L = io.wl_keys(wl_path)
    assert L == cbl
    io.write_u64_list(f"{work}/wl.bin", keys)
    rep["whitelist"] = int(len(keys))
    out = f"{work}/gpu"
    run_h(args, f"{out}.cell.i64", ["mode=simple", f"reads={rd}", f"wl={work}/wl.bin", f"out={out}", f"cb_start={cb0}", f"cb_len={cbl}", f"umi_start={um0}",
        f"umi_len={uml}", "mm1=1", "mm1_multi=1", "nbase=1", "one_exact=0", "pseudocount=1", "has_wl=1"])
    k_cell = io.load_arr(out, "cell", "i64")
    k_st = io.load_arr(out, "status", "i32")
    k_umi = io.load_arr(out, "umi", "u64")
    k_hist = io.load_arr(out, "histexact", "u32")
    off, seq, qual = io.load_reads(rd)
    cbm = io.fixed_matrix(off, seq, cb0, cbl)
    cbq = io.fixed_matrix(off, qual, cb0, cbl, pad=ord("H"))
    umm = io.fixed_matrix(off, seq, um0, uml)
    ucheck = umi_check_arrays(umm)
    ok = ucheck == 0
    rep["umi_bad"] = {"N": int((ucheck == -23).sum()), "homopolymer": int((ucheck == -24).sum())}
    # (1) reference: numba batch on every read
    cb_codes = io._CODE2[cbm]
    t = time.time()
    hist = np.zeros(len(keys), dtype=np.uint32)
    okidx = np.flatnonzero(ok)
    ref.exact_pass(cb_codes[okidx], cbq[okidx].astype(np.int64), len(okidx), cbl, keys, hist)
    hist_exact = hist.copy()
    hist += 1
    r_cell = np.full(n, -1, dtype=np.int64)
    r_code = ucheck.astype(np.int64).copy()
    cell_ok = np.empty(len(okidx), dtype=np.int64)
    code_ok = np.empty(len(okidx), dtype=np.int64)
    ref.resolve_pass(cb_codes[okidx], cbq[okidx].astype(np.int64), len(okidx), cbl, keys, hist, True, ref.qual_table(), cell_ok, code_ok)
    r_cell[okidx] = cell_ok
    r_code[okidx] = code_ok
    rep["ref_numba_seconds"] = round(time.time() - t, 1)
    d_cell = np.empty(len(okidx), dtype=np.int64)
    resolve_pass_f64(cb_codes[okidx], cbq[okidx].astype(np.int64), len(okidx), cbl, keys, hist, ref.qual_table(), d_cell)
    rep["double_precision_posterior_differs_on_reads"] = int((d_cell != cell_ok).sum())
    same_cell = k_cell == r_cell
    same_st = k_st == r_code
    rep["kernel_vs_numba"] = {"reads": n, "cell_identical": int(same_cell.sum()), "status_identical": int(same_st.sum()),
                              "hist_identical": bool((k_hist == hist_exact).all()), "exact_reads": int(hist_exact.sum())}
    # the statuses by class
    vals, cnts = np.unique(k_st, return_counts=True)
    rep["status_counts"] = {int(v): int(c) for v, c in zip(vals, cnts)}
    if not same_cell.all():
        bad = np.flatnonzero(~same_cell)[:5]
        rep["kernel_vs_numba"]["first_cell_diffs"] = [(int(i), int(k_cell[i]), int(r_cell[i]), int(k_st[i]), int(r_code[i])) for i in bad]
    # numba vs plain Python path of the reference on a random subset
    rng = random.Random(7)
    sub = sorted(rng.sample(range(n), min(args.py_subset, n)))
    # pick extra reads that need the posterior or an N, which are rare
    extra = [int(i) for i in np.flatnonzero(np.isin(r_code, [2, -4]))[: args.py_subset // 4]] + [int(i) for i in np.flatnonzero(cbq.min(axis=1) < ord("F"))[:1000]]
    sub = sorted(set(sub) | set(extra))
    wlobj = ref.Whitelist_from_sorted(list(keys))
    lay = ref.SimpleLayout(cb0 + 1, cbl, um0 + 1, uml, ref.flags_from_type("1MM_multi_Nbase_pseudocounts"), wlobj)
    sd = 0
    t = time.time()
    for i in sub:
        s = seq[off[i]:off[i + 1]].tobytes().decode()
        q = qual[off[i]:off[i + 1]].tobytes().decode()
        rec = ref.simple_record(s, q, lay, hist)
        c = rec.cell_key
        if c != k_cell[i]:
            sd += 1
    rep["kernel_vs_python"] = {"reads": len(sub), "cell_identical": len(sub) - sd, "python_seconds": round(time.time() - t, 1)}
    # (2) R11 CbCorrector (CPU)
    if args.cpu_exe:
        cpu = f"{work}/cpu"
        if not (args.reuse and os.path.exists(cpu + ".cell.i64")):
          sh([args.cpu_exe, f"reads={rd}", f"wl={wl_path}", f"out={cpu}", f"cb_start={cb0}", f"cb_len={cbl}", f"umi_start={um0}", f"umi_len={uml}",
            f"qual={'F' if cfg['qual'] == 'F' else 'file'}"])
        c_cell = io.load_arr(cpu, "cell", "i64")
        c_st = io.load_arr(cpu, "status", "i32")
        kst = np.where(k_st <= -23, -3, k_st)           # R11: bad UMI is -3 here
        rep["kernel_vs_R11"] = {"reads": n, "cell_identical": int((c_cell == k_cell).sum()), "status_identical": int((c_st == kst).sum())}
        if not (c_cell == k_cell).all():
            bad = np.flatnonzero(c_cell != k_cell)[:5]
            rep["kernel_vs_R11"]["first_diffs"] = [(int(i), int(k_cell[i]), int(c_cell[i]), int(k_st[i]), int(c_st[i])) for i in bad]
    # (3) STAR
    if args.star_bam and os.path.exists(args.star_bam):
        rep["star"] = compare_star_simple(cfg, args.star_bam, keys, k_cell, k_st, k_umi, cbm, umm, n)


def tag_to_cell(tag_arr: np.ndarray, keys: np.ndarray, L: int):
    """STAR CB strings (S64: b'' no tag, b'-' none, else barcode) -> whitelist index, -1 none/dash, -2 no tag."""
    out = np.full(len(tag_arr), -2, dtype=np.int64)
    has = tag_arr != b""
    dash = tag_arr == b"-"
    out[dash] = -1
    real = has & ~dash
    if real.any():
        m = io.strings_to_matrix(tag_arr[real], L)
        v, pn = io.pack2_matrix(m)
        pos = np.searchsorted(keys, v)
        pos = np.minimum(pos, len(keys) - 1)
        hit = (keys[pos] == v) & (pn == -1)
        res = np.where(hit, pos, -3)         # -3: tag string not in the whitelist
        out[real] = res
    return out


def ham(a: np.ndarray, b: np.ndarray):
    return (a != b).sum(axis=1)


def compare_star_simple(cfg, bam, keys, k_cell, k_st, k_umi, cbm, umm, n):
    cb0, cbl = cfg["cb"]
    um0, uml = cfg["umi"]
    t = time.time()
    st = io.star_tags(bam)
    rep = {"records": int(len(st["idx"])), "parse_seconds": round(time.time() - t, 1)}
    idx = st["idx"]
    present = np.zeros(n, dtype=bool)
    present[idx] = True
    rep["reads_without_record"] = int(n - present.sum())
    order = np.argsort(idx, kind="stable")
    idx = idx[order]
    flag = st["flag"][order]
    cbtag = st["CB"][order]
    ubtag = st["UB"][order]
    urtag = st["UR"][order]
    s_cell = tag_to_cell(cbtag, keys, cbl)
    o_cell = k_cell[idx]
    o_st = k_st[idx]
    unmapped = (flag & 4) != 0
    rep["records_mapped"] = int((~unmapped).sum())
    rep["records_unmapped"] = int(unmapped.sum())
    rep["cb_tag_on_mapped"] = int(((cbtag != b"") & ~unmapped).sum())
    rep["cb_tag_on_unmapped"] = int(((cbtag != b"") & unmapped).sum())
    rep["cb_tag_not_in_whitelist"] = int((s_cell == -3).sum())
    s_has = s_cell >= 0
    o_has = o_cell >= 0
    both_none = ~s_has & ~o_has
    same = s_has & o_has & (s_cell == o_cell)
    diff_cell = s_has & o_has & (s_cell != o_cell)
    ours_only = o_has & ~s_has
    star_only = s_has & ~o_has
    # the comparison universe is reads with a record and (for CB) a mapped alignment or a CB tag decision; report both
    for name, sel in (("all_records", np.ones(len(idx), bool)), ("mapped", ~unmapped), ("with_cb_tag_field", (cbtag != b""))):
        r = {"records": int(sel.sum()), "both_none": int((both_none & sel).sum()), "same_cell": int((same & sel).sum()),
             "different_cell": int((diff_cell & sel).sum()), "ours_cell_star_none": int((ours_only & sel).sum()),
             "star_cell_ours_none": int((star_only & sel).sum())}
        r["CB_identical_pct"] = pct(r["both_none"] + r["same_cell"], r["records"])
        rep[name] = r
    # reasons for disagreements
    bad = ~(both_none | same)
    vals, cnts = np.unique(o_st[bad & ~unmapped] if bad.any() else np.zeros(0, np.int32), return_counts=True)
    rep["disagree_by_our_status_mapped"] = {int(v): int(c) for v, c in zip(vals, cnts)}
    ex = np.flatnonzero(bad & ~unmapped)[:8]
    cbs = io.decode2(io.pack2_matrix(cbm[idx[ex]])[0], cbl) if len(ex) else []
    rep["disagree_examples"] = [
        {"read": int(idx[i]), "raw_cb": cbs[j].decode(), "ours": int(o_cell[i]), "ours_status": int(o_st[i]),
         "star_cb": cbtag[i].decode(), "star_cell": int(s_cell[i])} for j, i in enumerate(ex)]
    # UR / UB
    urm = io.strings_to_matrix(urtag, uml)
    has_ur = urtag != b""
    ours_umm = umm[idx]
    rep["UR"] = {"records_with_UR": int(has_ur.sum()), "identical_to_raw_umi": int((has_ur & (ham(urm, ours_umm) == 0)).sum())}
    rep["UR"]["identical_pct"] = pct(rep["UR"]["identical_to_raw_umi"], rep["UR"]["records_with_UR"])
    has_ub = (ubtag != b"") & (ubtag != b"-")
    sel = has_ub & same
    ubm = io.strings_to_matrix(ubtag[sel], uml)
    h = ham(ubm, ours_umm[sel])
    rep["UB"] = {"records_with_UB_and_same_CB": int(sel.sum()), "UB_equals_raw_umi": int((h == 0).sum()), "UB_one_base_from_raw": int((h == 1).sum()),
                 "UB_more_than_one_base_from_raw": int((h > 1).sum()),
                 "UB_dash_or_absent_with_same_CB": int((same & ~has_ub).sum())}
    rep["UB"]["identical_pct"] = pct(rep["UB"]["UB_equals_raw_umi"], rep["UB"]["records_with_UB_and_same_CB"])
    ub_wrong_cb = has_ub & ~same
    rep["UB"]["UB_present_but_CB_disagrees"] = int(ub_wrong_cb.sum())
    return rep


# ---------------------------------------------------------------------------------------------------------------------
def compare_nowl(cfg, work, args, rep):
    cb0, cbl = cfg["cb"]
    um0, uml = cfg["umi"]
    rd = f"{work}/reads.bin"
    n = prep_reads(args, f"{DATA}/{cfg['dir']}/{cfg['bc']}", rd, "F")
    rep["reads"] = n
    out = f"{work}/gpu"
    mr = cfg["min_reads"]
    run_h(args, f"{out}.cell.i64", ["mode=discover", f"reads={rd}", f"out={out}", f"cb_start={cb0}", f"cb_len={cbl}", f"umi_start={um0}", f"umi_len={uml}", f"min_reads={mr}"])
    k_cell = io.load_arr(out, "cell", "i64")
    k_st = io.load_arr(out, "status", "i32")
    k_umi = io.load_arr(out, "umi", "u64")
    k_cb = io.load_arr(out, "cbval", "u64")
    k_pn = io.load_arr(out, "posn", "i32")
    dk = io.load_arr(out, "disc_keys", "u64")
    dc = io.load_arr(out, "disc_counts", "u32")
    off, seq, qual = io.load_reads(rd)
    cbm = io.fixed_matrix(off, seq, cb0, cbl)
    umm = io.fixed_matrix(off, seq, um0, uml)
    cbv, pn = io.pack2_matrix(cbm)
    ucheck = umi_check_arrays(umm)
    # reference discovery: counts over N-free barcodes of every raw read
    nf = pn == -1
    uv, uc = np.unique(cbv[nf], return_counts=True)
    raw = {int(v): int(c) for v, c in zip(uv, uc) if c >= mr}
    disc = ref.discover_barcodes(raw, mr, True)         # keys: ints; ties by key ascending == barcode string ascending
    rep["distinct_N_free_barcodes"] = int(len(uv))
    rep["discovered"] = len(disc)
    rep["kernel_vs_ref_discovery"] = {"list_identical": bool(len(disc) == len(dk) and (np.array(disc, dtype=np.uint64) == dk).all()),
                                      "counts_identical": bool(len(disc) == len(dc) and all(raw[int(k)] == int(c) for k, c in zip(dk, dc)))}
    index = {b: i for i, b in enumerate(disc)}
    r_cell = np.full(n, -1, dtype=np.int64)
    sorted_disc = np.array(disc, dtype=np.uint64)
    if len(disc):
        so = np.argsort(sorted_disc)
        pos = np.searchsorted(sorted_disc[so], cbv)
        pos = np.minimum(pos, len(disc) - 1)
        hit = (sorted_disc[so][pos] == cbv) & nf & (ucheck == 0)
        r_cell[hit] = so[pos[hit]]
    r_st = np.where(ucheck != 0, ucheck, np.where(nf, 0, -2))
    rep["kernel_vs_numpy"] = {"reads": n, "cell_identical": int((r_cell == k_cell).sum()), "status_identical": int((r_st == k_st).sum())}
    rng = random.Random(3)
    sub = sorted(rng.sample(range(n), min(args.py_subset, n)))
    lay = ref.SimpleLayout(cb0 + 1, cbl, um0 + 1, uml, ref.flags_from_type("Exact"), None)
    discovered_str = {io.decode2(np.array([b], dtype=np.uint64), cbl)[0].decode(): i for i, b in enumerate(disc)}
    sd = 0
    for i in sub:
        s = seq[off[i]:off[i + 1]].tobytes().decode()
        q = qual[off[i]:off[i + 1]].tobytes().decode()
        rec = ref.simple_record(s, q, lay, None, discovered_str)
        if rec.cell_key != k_cell[i]:
            sd += 1
    rep["kernel_vs_python"] = {"reads": len(sub), "cell_identical": len(sub) - sd}
    rep["umi_bad"] = {"N": int((ucheck == -23).sum()), "homopolymer": int((ucheck == -24).sum())}
    if args.star_bam and os.path.exists(args.star_bam):
        t = time.time()
        st = io.star_tags(args.star_bam, tags=("CB", "UR", "CR"))
        idx = st["idx"]
        o = np.argsort(idx, kind="stable")
        idx, flag, cbtag, urtag = idx[o], st["flag"][o], st["CB"][o], st["UR"][o]
        unm = (flag & 4) != 0
        raw_str = io.decode2(cbv[idx], cbl)
        ours_n = pn[idx] != -1
        s_dash = cbtag == b"-"
        s_has = (cbtag != b"") & ~s_dash
        same = s_has & ~ours_n & (cbtag == raw_str)
        bn = ~s_has & ours_n
        rr = {"parse_seconds": round(time.time() - t, 1), "records": int(len(idx)), "records_mapped": int((~unm).sum()),
              "cb_tag_on_mapped": int((s_has & ~unm).sum()), "cb_tag_on_unmapped": int((s_has & unm).sum())}
        for name, sel in (("all_records", np.ones(len(idx), bool)), ("mapped", ~unm), ("with_cb_tag_field", cbtag != b"")):
            r = {"records": int(sel.sum()), "same_barcode": int((same & sel).sum()), "both_N_or_dash": int((bn & sel).sum()),
                 "star_barcode_ours_N": int((s_has & ours_n & sel).sum()), "star_dash_ours_barcode": int((~s_has & ~ours_n & sel).sum()),
                 "different_barcode": int((s_has & ~ours_n & (cbtag != raw_str) & sel).sum())}
            r["CB_identical_pct"] = pct(r["same_barcode"] + r["both_N_or_dash"], r["records"])
            rr[name] = r
        urm = io.strings_to_matrix(urtag, uml)
        has_ur = urtag != b""
        rr["UR"] = {"records_with_UR": int(has_ur.sum()), "identical_to_raw_umi": int((has_ur & (ham(urm, umm[idx]) == 0)).sum())}
        rr["UR"]["identical_pct"] = pct(rr["UR"]["identical_to_raw_umi"], rr["UR"]["records_with_UR"])
        rr["UB"] = "not produced by STAR for CB_samTagOut"
        in_list = k_cell[idx] >= 0
        rr["same_barcode_and_in_discovered_list"] = int((same & in_list).sum())
        rep["star"] = rr


# ---------------------------------------------------------------------------------------------------------------------
def seg_strings(path):
    """segment whitelist -> per length: sorted unique (by 2-bit value) list of strings, STAR's order."""
    by = {}
    for s in io.read_lines(path):
        v, pn = ref.conv(s)
        if pn == 0:
            continue
        by.setdefault(len(s), {})[v] = s       # a later duplicate string has the same value; N-as-A gives the same value
    return {ln: [d[v] for v in sorted(d)] for ln, d in by.items()}


def compare_complex(cfg, work, args, rep):
    rd = f"{work}/reads.bin"
    n = prep_reads(args, f"{DATA}/{cfg['dir']}/{cfg['bc']}", rd, "F")
    rep["reads"] = n
    segs, umi = cfg["segs"], cfg["umi"]
    wls = [f"{PB_WL}/{w}" for w in cfg["wl"]]
    lay = f"{work}/layout.txt"
    with open(lay, "w") as f:
        f.write(f"adapter {cfg.get('adapter', '-')}\nadapter_mm {cfg.get('adapter_mm', 1)}\n")
        p = umi.split("_")
        f.write(f"umi {p[0]} {p[1]} {p[2]} {p[3]}\n")
        for sg, w in zip(segs, wls):
            p = sg.split("_")
            f.write(f"seg {p[0]} {p[1]} {p[2]} {p[3]} {w}\n")
    out = f"{work}/gpu"
    run_h(args, f"{out}.cell.i64", ["mode=complex", f"reads={rd}", f"layout={lay}", f"out={out}"])
    k_cell = io.load_arr(out, "cell", "i64")
    k_st = io.load_arr(out, "status", "i32")
    k_umi = io.load_arr(out, "umi", "u64")
    k_ul = io.load_arr(out, "umilen", "u32")
    off, seq, qual = io.load_reads(rd)
    vals, cnts = np.unique(k_st, return_counts=True)
    rep["status_counts"] = {int(v): int(c) for v, c in zip(vals, cnts)}
    # reference: plain Python on a random subset plus every read of each rare status (capped)
    seg_seqs = [io.read_lines(w) for w in wls]
    sg_objs = []
    for pos, seqs in zip(segs, seg_seqs):
        s = ref.Segment(pos)
        ref.build_segment_wl(s, seqs)
        sg_objs.append(s)
    total = ref.link_segments(sg_objs)
    clay = ref.ComplexLayout(sg_objs, ref.Segment(umi), ref.flags_from_type("Exact"), cfg.get("adapter", "-"), cfg.get("adapter_mm", 1))
    rep["cell_space"] = int(total)
    rng = random.Random(11)
    sub = set(rng.sample(range(n), min(args.py_subset, n)))
    for v in vals:
        if v != 0:
            sub |= set(int(i) for i in np.flatnonzero(k_st == v)[: args.py_subset // 8])
    sub = sorted(sub)
    sd = ss = su = 0
    t = time.time()
    first = []
    for i in sub:
        s = seq[off[i]:off[i + 1]].tobytes().decode()
        q = qual[off[i]:off[i + 1]].tobytes().decode()
        rec = ref.complex_record(s, q, clay)
        ref_cell = rec.cell_key
        ref_st = rec.status
        if ref_cell != k_cell[i]:
            sd += 1
            if len(first) < 5:
                first.append((i, int(k_cell[i]), int(ref_cell), int(k_st[i]), int(ref_st)))
        if ref_st != k_st[i]:
            ss += 1
        if rec.umi_key != int(k_umi[i]) or rec.umi_len != int(k_ul[i]):
            su += 1
    rep["kernel_vs_python"] = {"reads": len(sub), "cell_identical": len(sub) - sd, "status_identical": len(sub) - ss, "umi_identical": len(sub) - su,
                               "python_seconds": round(time.time() - t, 1), "first_cell_diffs": first}
    if args.star_bam and os.path.exists(args.star_bam):
        rep["star"] = compare_star_complex(cfg, args.star_bam, sg_objs, seg_seqs, k_cell, k_st, k_umi, k_ul, off, seq, n)


def decode_cells(cell: np.ndarray, sg_objs, seg_seqs):
    """composite index -> 'a_b_c' strings for fixed-length segments (segment 0 varies fastest)."""
    parts = []
    rem = cell.copy()
    for s, seqs in zip(sg_objs, seg_seqs):
        assert len(s.wl) == 1, "variable-length segment lists are decoded per length"
        ln = next(iter(s.wl))
        lst = np.array([x.encode() for x in seg_strings_from(s, seqs)], dtype=f"S{ln}")
        i = rem % s.total_size
        rem = rem // s.total_size
        parts.append(lst[i])
    out = parts[0]
    for p in parts[1:]:
        out = np.char.add(np.char.add(out, b"_"), p)
    return out


def seg_strings_from(s, seqs):
    d = {}
    for x in seqs:
        v, pn = ref.conv(x)
        if pn == 0:
            continue
        d[v] = x
    return [d[v] for v in sorted(d)]


def compare_star_complex(cfg, bam, sg_objs, seg_seqs, k_cell, k_st, k_umi, k_ul, off, seq, n):
    t = time.time()
    st = io.star_tags(bam, tags=("CB", "UB", "UR"))
    idx = st["idx"]
    o = np.argsort(idx, kind="stable")
    idx, flag, cbtag, ubtag, urtag = idx[o], st["flag"][o], st["CB"][o], st["UB"][o], st["UR"][o]
    unm = (flag & 4) != 0
    rep = {"parse_seconds": round(time.time() - t, 1), "records": int(len(idx)), "records_mapped": int((~unm).sum()),
           "cb_tag_on_mapped": int(((cbtag != b"") & (cbtag != b"-") & ~unm).sum()), "cb_tag_on_unmapped": int(((cbtag != b"") & (cbtag != b"-") & unm).sum())}
    oc = k_cell[idx]
    ours_has = oc >= 0
    o_str = np.full(len(idx), b"-", dtype="S64")
    if ours_has.any():
        o_str[ours_has] = decode_cells(oc[ours_has], sg_objs, seg_seqs).astype("S64")
    s_has = (cbtag != b"") & (cbtag != b"-")
    same = s_has & ours_has & (cbtag == o_str)
    both_none = ~s_has & ~ours_has
    for name, sel in (("all_records", np.ones(len(idx), bool)), ("mapped", ~unm), ("with_cb_tag_field", cbtag != b"")):
        r = {"records": int(sel.sum()), "same_cell": int((same & sel).sum()), "both_none": int((both_none & sel).sum()),
             "different_cell": int((s_has & ours_has & (cbtag != o_str) & sel).sum()), "ours_cell_star_none": int((ours_has & ~s_has & sel).sum()),
             "star_cell_ours_none": int((s_has & ~ours_has & sel).sum())}
        r["CB_identical_pct"] = pct(r["same_cell"] + r["both_none"], r["records"])
        rep[name] = r
    bad = ~(same | both_none)
    vals, cnts = np.unique(k_st[idx][bad & ~unm], return_counts=True)
    rep["disagree_by_our_status_mapped"] = {int(v): int(c) for v, c in zip(vals, cnts)}
    ex = np.flatnonzero(bad & ~unm)[:6]
    rep["disagree_examples"] = [{"read": int(idx[i]), "ours": o_str[i].decode(), "ours_status": int(k_st[idx[i]]), "star_cb": cbtag[i].decode()} for i in ex]
    ul = int(k_ul[0]) if len(k_ul) else 0
    umm = np.zeros((len(idx), ul), dtype=np.uint8)
    dec = io.decode3(k_umi[idx], ul) if ul else np.zeros(len(idx), "S1")
    has_ur = urtag != b""
    rep["UR"] = {"records_with_UR": int(has_ur.sum()), "identical_to_raw_umi": int((has_ur & (dec == urtag)).sum())}
    rep["UR"]["identical_pct"] = pct(rep["UR"]["identical_to_raw_umi"], rep["UR"]["records_with_UR"])
    has_ub = (ubtag != b"") & (ubtag != b"-")
    sel = has_ub & same
    ubm = io.strings_to_matrix(ubtag[sel], ul)
    dm = io.strings_to_matrix(dec[sel], ul)
    h = ham(ubm, dm)
    rep["UB"] = {"records_with_UB_and_same_CB": int(sel.sum()), "UB_equals_raw_umi": int((h == 0).sum()), "UB_one_base_from_raw": int((h == 1).sum()),
                 "UB_more_than_one_base_from_raw": int((h > 1).sum()), "UB_dash_or_absent_with_same_CB": int((same & ~has_ub).sum())}
    rep["UB"]["identical_pct"] = pct(rep["UB"]["UB_equals_raw_umi"], rep["UB"]["records_with_UB_and_same_CB"])
    return rep


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True, choices=sorted(CLASSES))
    ap.add_argument("--exe", required=True)
    ap.add_argument("--cpu-exe", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--star-bam", default=None)
    ap.add_argument("--max-reads", type=int, default=None)
    ap.add_argument("--py-subset", type=int, default=200000)
    ap.add_argument("--data", default=None)
    ap.add_argument("--wl-dir", default=None)
    ap.add_argument("--reuse", action="store_true", help="reuse reads.bin and harness outputs already in --work")
    args = ap.parse_args()
    if args.data:
        DATA = args.data
    if args.wl_dir:
        PB_WL = args.wl_dir
    cfg = CLASSES[args.cls]
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    rep = {"class": args.cls, "kind": cfg["kind"]}
    t0 = time.time()
    if cfg["kind"] == "simple":
        compare_simple(cfg, args.work, f"{PB_WL}/{cfg['wl']}", args, rep)
    elif cfg["kind"] == "nowl":
        compare_nowl(cfg, args.work, args, rep)
    else:
        compare_complex(cfg, args.work, args, rep)
    rep["seconds"] = round(time.time() - t0, 1)
    with open(f"{args.out}/{args.cls}.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
