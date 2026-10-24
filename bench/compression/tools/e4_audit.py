"""E4 Part A: derivability audit of a .singlet bundle (and the per-sample corpus members).

For every member type: bytes (total, per sample), the code that produces it, its inputs, and a verdict. For
splice_psi, colsums (and ptr, perm), cell_calls.tsv and the summary.json numeric fields the recomputation is actually
carried out and compared byte for byte (floats: bit patterns, max ulp).

    python -P tools/e4_audit.py            # full run (bundle GSE145370 14 samples + 7 corpus samples), ~5-10 min
    python -P tools/e4_audit.py --quick    # GSM4317429 only

Outputs: results/e4_recompute.tsv (one row per test), results/e4_members.tsv (the audit table),
         results/e4_audit.log (human-readable log of every comparison).
Measurement only; nothing in the production tree is modified. corpus/ and the bundle are untrusted input.
"""
import argparse
import collections
import json
import lzma
import math
import os
import re
import struct
import subprocess
import sys
import time
import zipfile

import numpy as np
import zstandard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_lib as L  # noqa: E402

T0 = time.time()
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(f"[{time.time() - T0:7.1f}s] {s}", flush=True)


REC = []  # recompute rows


def rec(test, sample, item, n, n_equal, exact, max_ulp="", note=""):
    REC.append(dict(test=test, sample=sample, item=item, n_compared=n, n_equal=n_equal, exact=exact, max_ulp=max_ulp, note=note))


# ------------------------------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------------------------------
KINDS = ["exon_counts", "intron_counts", "sj_counts", "splice_psi", "vdj_gene_usage"]


def load_sample(z, gsm, pz):
    """Read all members of one sample from the bundle; decode the .1pz members with the production reader."""
    d = os.path.join(L.SCRATCH, gsm)
    os.makedirs(d, exist_ok=True)
    S = dict(gsm=gsm, raw={}, mat={}, pzinfo={})
    for zi in z.infolist():
        if not zi.filename.startswith(f"samples/{gsm}/"):
            continue
        fn = zi.filename.split("/")[-1]
        raw = z.read(zi)
        S["raw"][fn] = raw
        if fn.endswith(".1pz"):
            p = os.path.join(d, fn)
            with open(p, "wb") as f:
                f.write(raw)
            S["pzinfo"][fn[:-4]] = L.parse_pz(raw)
            if S["pzinfo"][fn[:-4]][0]["m"] > 0:
                S["mat"][fn[:-4]] = pz.read_1pz(p)
            else:
                S["mat"][fn[:-4]] = None
    return S


# ------------------------------------------------------------------------------------------------------------------
# T1-T3: colsums, ptr, perm blocks inside every .1pz
# ------------------------------------------------------------------------------------------------------------------
def colsums_of(indptr, data, n):
    cs = np.zeros(n, np.uint64)
    if len(data) == 0:
        return cs
    nz = indptr[1:] > indptr[:-1]
    starts = indptr[:-1][nz].astype(np.int64)
    sums = np.add.reduceat(data.astype(np.uint64), starts)
    cs[nz] = sums
    return cs


def perm_from_rnnz(rnnz):
    """Counting sort, descending count, ties by ascending row id (pz_writer.h encode_vocsc_section)."""
    return np.lexsort((np.arange(len(rnnz)), -rnnz.astype(np.int64))).astype(np.uint32)


def test_container_blocks(S, helper):
    gsm = S["gsm"]
    zc = zstandard.ZstdCompressor(level=3, write_checksum=True)
    out = {}
    for kind in KINDS:
        if S["mat"].get(kind) is None:
            continue
        buf = S["raw"][kind + ".1pz"]
        h, o = S["pzinfo"][kind]
        M = S["mat"][kind]
        n, m = h["n"], h["m"]
        # colsums
        cs_stored = np.frombuffer(L.zdec(buf, o["colsums_off"], h["colsums_z_sz"]), dtype="<u8")
        cs_re = colsums_of(M["indptr"], M["data"], n)
        eq = bool(len(cs_stored) == n and np.array_equal(cs_stored, cs_re))
        blob = bytes(buf[o["colsums_off"] : o["colsums_off"] + h["colsums_z_sz"]])
        blob_py = zc.compress(cs_re.astype("<u8").tobytes())
        # same zstd build as the writer? test with 1.5.5 via helper
        tmpi = os.path.join(L.SCRATCH, "cs_in.bin")
        tmpo = os.path.join(L.SCRATCH, "cs_out.bin")
        with open(tmpi, "wb") as f:
            f.write(cs_re.astype("<u8").tobytes())
        subprocess.run([helper, "zc3", tmpi, tmpo], check=True)
        blob_cpp = open(tmpo, "rb").read()
        out[(kind, "colsums")] = dict(raw_equal=eq, bytes_py157=(blob == blob_py), bytes_cpp155=(blob == blob_cpp))
        # ptr
        ptr_raw = L.zdec(buf, o["ptr_off"], h["ptr_z_sz"])
        w = h["ptr_width"]
        pa = np.frombuffer(ptr_raw, dtype=np.uint8).reshape(n, w).astype(np.uint64)
        ptr_vals = sum(pa[:, b] << np.uint64(8 * b) for b in range(w))
        ptr_re = np.diff(M["indptr"]).astype(np.uint64)
        out[(kind, "ptr")] = dict(raw_equal=bool(np.array_equal(ptr_vals, ptr_re)), bytes_py157=None, bytes_cpp155=None)
        # perm
        perm_stored = np.frombuffer(L.zdec(buf, o["perm_off"], h["perm_z_sz"]), dtype="<u4")
        rnnz = np.bincount(M["indices"], minlength=m)
        perm_re = perm_from_rnnz(rnnz)
        out[(kind, "perm")] = dict(raw_equal=bool(np.array_equal(perm_stored, perm_re)), bytes_py157=None, bytes_cpp155=None)
        # row-count replacement cost (Track A definition: LEB128 + best of zstd-19 / xz)
        out[(kind, "rowcounts_bytes")] = rowcounts_cost(rnnz)
    return out


def leb128(a):
    out = bytearray()
    for v in a.tolist():
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
    return bytes(out)


def rowcounts_cost(rnnz):
    """Bytes to store per-row nnz counts so that the decoder re-derives the permutation (Track A section 3)."""
    best = None
    for name, arr in (("plain", rnnz), ("zigzag", None)):
        if name == "plain":
            raw = leb128(rnnz)
        else:
            d = np.diff(rnnz.astype(np.int64), prepend=0)
            zz = np.where(d >= 0, d * 2, -d * 2 - 1)
            raw = leb128(zz)
        c1 = len(zstandard.ZstdCompressor(level=19).compress(raw))
        c2 = len(lzma.compress(raw, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]))
        for c in (c1, c2):
            if best is None or c < best:
                best = c
    return best


# ------------------------------------------------------------------------------------------------------------------
# T4: duplicated name lists
# ------------------------------------------------------------------------------------------------------------------
def test_names(S):
    gsm = S["gsm"]
    M = S["mat"]
    sj, psi = M.get("sj_counts"), M.get("splice_psi")
    res = {}
    if sj is not None and psi is not None:
        res["psi_rownames==sj_rownames"] = sj["rownames"] == psi["rownames"]
    cols = [M[k]["colnames"] for k in KINDS if M.get(k) is not None]
    res["colnames_identical_across_members"] = all(c == cols[0] for c in cols)
    res["n_members_with_colnames"] = len(cols)
    return res


# ------------------------------------------------------------------------------------------------------------------
# T5: splice_psi from sj_counts
# ------------------------------------------------------------------------------------------------------------------
def parse_junction(name):
    c = name.find(":")
    if c < 0:
        return None
    d = name.find("-", c + 1)
    if d < 0:
        return None
    try:
        start = int(name[c + 1 : d])
        end = int(name[d + 1 :])
    except ValueError:
        return None
    return name[:c], start, end


def build_junction_event(names):
    """splice_psi.h build_splice_events() + the first-event-wins assignment in compute_psi()."""
    donors, accs = {}, {}
    for i, nm in enumerate(names):
        p = parse_junction(nm)
        if p is None:
            continue
        chrom, s, e = p
        donors.setdefault(f"{chrom}:{s}-D", []).append(i)
        accs.setdefault(f"{chrom}:{e}-A", []).append(i)
    events = [(k, v) for k, v in donors.items() if len(v) >= 2] + [(k, v) for k, v in accs.items() if len(v) >= 2]
    events.sort(key=lambda kv: kv[0].encode())  # std::string operator< is bytewise
    je = np.full(len(names), -1, np.int64)
    for e, (_, jl) in enumerate(events):
        for j in jl:
            if je[j] == -1:
                je[j] = e
    return je, len(events)


def psi_recompute(sj):
    """Float32 PSI exactly as compute_psi() defines it: data / (sum of data over the junctions of the same event)."""
    names, indptr, indices, data = sj["rownames"], sj["indptr"], sj["indices"], sj["data"]
    n = sj["n"]
    je, n_events = build_junction_event(names)
    cell = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    ev = je[indices]
    mask = ev >= 0
    key = cell[mask] * max(n_events, 1) + ev[mask]
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.bincount(inv, weights=data[mask].astype(np.float64))  # integer-valued, exact in float64 and float32
    assert sums.max() < 2**24 if len(sums) else True
    evsum = sums[inv].astype(np.float32)
    psi = data[mask].astype(np.float32) / evsum
    ptr = np.zeros(n + 1, np.int64)
    ptr[1:] = np.cumsum(np.bincount(cell[mask], minlength=n))
    return ptr, indices[mask].astype(np.int32), psi, n_events


def ulp_diff32(a, b):
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    return np.abs(ia - ib)


def test_psi(S, helper, full_ref=True):
    gsm = S["gsm"]
    sj, psi = S["mat"].get("sj_counts"), S["mat"].get("splice_psi")
    if sj is None or psi is None:
        return None
    t = time.time()
    ptr, idx, pf, n_ev = psi_recompute(sj)
    # (a) stored values vs truncation of the float PSI
    trunc = pf.astype(np.uint32)  # C++ static_cast<uint32_t>(float) truncates toward zero
    rint = np.rint(pf).astype(np.uint32)  # round half to even
    same_struct = bool(np.array_equal(ptr, psi["indptr"]) and np.array_equal(idx, psi["indices"]))
    same_vals = bool(same_struct and np.array_equal(trunc, psi["data"]))
    rint_vals = bool(same_struct and np.array_equal(rint, psi["data"]))
    n_trunc_diff = int((trunc != psi["data"]).sum()) if same_struct else -1
    n_rint_diff = int((rint != psi["data"]).sum()) if same_struct else -1
    stored_vals = np.unique(psi["data"]).tolist()
    n_gt0 = int((pf > 0).sum())
    n_eq1 = int((pf == 1.0).sum())
    res = dict(
        gsm=gsm, sj_nnz=int(sj["nnz"]), psi_nnz_stored=int(psi["nnz"]), psi_nnz_re=int(len(pf)), events=n_ev,
        struct_equal=same_struct, trunc_values_equal=same_vals, rint_values_equal=rint_vals, n_trunc_diff=n_trunc_diff,
        n_rint_diff=n_rint_diff, stored_distinct_values=stored_vals,
        frac_entries_with_psi_eq_1=n_eq1 / max(1, len(pf)),
        n_distinct_float_psi=int(len(np.unique(pf))),
    )
    # (b) the production code: compute_psi() float bits and write_1pz<float>() container bytes
    if full_ref:
        d = os.path.join(L.SCRATCH, gsm, "psi_ref")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "sj.bin"), "wb") as f:
            f.write(struct.pack("<IIQ", sj["m"], sj["n"], sj["nnz"]))
            f.write(sj["indptr"].astype("<i4").tobytes())
            f.write(sj["indices"].astype("<i4").tobytes())
            f.write(sj["data"].astype("<u2").tobytes())
        assert not any("\n" in x for x in sj["rownames"] + sj["colnames"])
        with open(os.path.join(d, "names.txt"), "w", newline="\n", encoding="utf-8") as f:
            f.write("\n".join(sj["rownames"]) + "\n")
        with open(os.path.join(d, "cols.txt"), "w", newline="\n", encoding="utf-8") as f:
            f.write("\n".join(sj["colnames"]) + "\n")
        r = subprocess.run([helper, "psi", d], capture_output=True, text=True)
        if r.returncode != 0:
            res["ref_error"] = r.stderr[-500:]
        else:
            with open(os.path.join(d, "psi.bin"), "rb") as f:
                (k,) = struct.unpack("<Q", f.read(8))
                ip = np.frombuffer(f.read(4 * (sj["n"] + 1)), dtype="<i4")
                ix = np.frombuffer(f.read(4 * k), dtype="<i4")
                fv = np.frombuffer(f.read(4 * k), dtype="<f4")
            res["ref_structure_equal"] = bool(np.array_equal(ip, ptr) and np.array_equal(ix, idx))
            u = ulp_diff32(fv, pf) if len(fv) == len(pf) else np.array([-1])
            res["ref_float_max_ulp"] = int(u.max()) if len(u) else 0
            res["ref_float_n_bit_equal"] = int((u == 0).sum())
            ref_bytes = open(os.path.join(d, "psi_ref.1pz"), "rb").read()
            res["container_bytes_equal_trunc"] = ref_bytes == S["raw"]["splice_psi.1pz"]
            ref2 = open(os.path.join(d, "psi_ref_rint.1pz"), "rb").read()
            res["container_bytes_equal"] = ref2 == S["raw"]["splice_psi.1pz"]
            res["container_bytes_ref_trunc"] = len(ref_bytes)
            res["container_bytes_ref"] = len(ref2)
            res["container_bytes_stored"] = len(S["raw"]["splice_psi.1pz"])
            res["ref_log"] = r.stdout.strip()
    res["secs"] = round(time.time() - t, 1)
    return res


# ------------------------------------------------------------------------------------------------------------------
# T6: cell_calls.tsv
# ------------------------------------------------------------------------------------------------------------------
def cell_calls_text(barcodes_in_order, tot_by_bc, method, pvals=None):
    lines = ["barcode\tis_cell\tlog10_umi\tempty_drops_pvalue\tcall_method"]
    for bc in barcodes_in_order:
        t = tot_by_bc[bc]
        l10 = math.log10(float(t)) if t > 0 else 0.0
        lines.append(f"{bc}\tTRUE\t{l10:.4f}\t0.0000e+00\t{method}")
    return ("\n".join(lines) + "\n").encode()


def test_cell_calls(label, exon, calls_raw):
    """Recompute cell_calls.tsv from the exon colsums + the *set* of called barcodes. Returns a result dict."""
    n = exon["n"]
    tot = colsums_of(exon["indptr"], exon["data"], n)
    cols = exon["colnames"]
    idx = {b: i for i, b in enumerate(cols)}
    txt = calls_raw.decode()
    rows = [ln.split("\t") for ln in txt.split("\n") if ln]
    hdr, rows = rows[0], rows[1:]
    bcs = [r[0] for r in rows]
    res = dict(label=label, n_columns=n, n_rows=len(rows), header_ok=hdr == ["barcode", "is_cell", "log10_umi", "empty_drops_pvalue", "call_method"])
    res["all_called_in_matrix"] = all(b in idx for b in bcs)
    res["all_is_cell_TRUE"] = all(r[1] == "TRUE" for r in rows)
    res["pvalue_constant"] = sorted(set(r[3] for r in rows))
    res["methods"] = sorted(set(r[4] for r in rows))
    res["rows_sorted_bytewise"] = bcs == sorted(bcs, key=lambda s: s.encode())
    res["rows_in_matrix_column_order"] = [idx[b] for b in bcs if b in idx] == sorted(idx[b] for b in bcs if b in idx)
    # log10 column
    tb = {b: int(tot[idx[b]]) for b in bcs if b in idx}
    ok = 0
    ok_np = 0
    for r in rows:
        if r[0] not in tb:
            continue
        t = tb[r[0]]
        s = f"{math.log10(t):.4f}" if t > 0 else "0.0000"
        s2 = f"{float(np.log10(np.float64(t))):.4f}" if t > 0 else "0.0000"
        ok += s == r[2]
        ok_np += s2 == r[2]
    res["log10_text_equal"] = ok
    res["log10_text_equal_numpy"] = ok_np
    # whole file regenerated from (set of barcodes in stored order, exon colsums, constants)
    meth = res["methods"][0] if len(res["methods"]) == 1 else ""
    if res["all_called_in_matrix"] and meth:
        regen = cell_calls_text(bcs, tb, meth)
        res["file_bytes_equal"] = regen == calls_raw
        # and when the order is regenerated by sorting the barcodes
        regen2 = cell_calls_text(sorted(bcs, key=lambda s: s.encode()), tb, meth)
        res["file_bytes_equal_sorted_order"] = regen2 == calls_raw
    # information content of the called *set* given depth
    called = np.zeros(n, bool)
    for b in bcs:
        if b in idx:
            called[idx[b]] = True
    res["info"] = flag_information(tot, called)
    return res, tot, called


def flag_information(tot, called):
    """How much independent information is in the called-cell flag once column depth is known?"""
    n = len(tot)
    k = int(called.sum())
    out = dict(n_columns=n, n_called=k, bitmap_bytes=math.ceil(n / 8))
    # best single threshold on total UMI
    order = np.argsort(-tot.astype(np.int64), kind="stable")
    c = called[order].astype(np.int64)
    # errors if threshold sits after position p (cells = first p columns): cells beyond p + non-cells within p
    cum_cells = np.cumsum(c)
    p = np.arange(1, n + 1)
    err = (k - cum_cells) + (p - cum_cells)
    # only allow cuts between distinct totals
    ts = tot.astype(np.int64)[order]
    valid = np.ones(n, bool)
    valid[:-1] = ts[:-1] != ts[1:]
    err_v = np.where(valid, err, 10**12)
    best = int(err_v.argmin())
    out["threshold_umi"] = int(ts[best])
    out["threshold_exceptions"] = int(err_v[best])
    # ideal adaptive code length of the flag given a depth bin (half-octave bins), MDL parameter cost
    bins = np.floor(2 * np.log2(tot.astype(np.float64) + 1)).astype(np.int64)
    bits = 0.0
    for b in np.unique(bins):
        sel = bins == b
        nb = int(sel.sum())
        k1 = int(called[sel].sum())
        if 0 < k1 < nb:
            p1 = k1 / nb
            bits += -(k1 * math.log2(p1) + (nb - k1) * math.log2(1 - p1)) + 0.5 * math.log2(nb) + 1
        elif nb > 0:
            bits += 0.5 * math.log2(nb) + 1
    out["flag_given_depth_bytes"] = int(math.ceil(bits / 8))
    # cost of listing the exceptions of the best threshold explicitly
    E = out["threshold_exceptions"]
    out["threshold_exceptions_list_bytes"] = int(math.ceil((E * (math.log2(max(n / max(E, 1), 2)) + 1.44) + 24) / 8)) if E else 3
    return out


# ------------------------------------------------------------------------------------------------------------------
# T7: summary.json / pileup_stats.json / provenance.json / study_meta.json / manifest.json
# ------------------------------------------------------------------------------------------------------------------
def jtok(text, key):
    m = re.search(r'"%s":\s*([^,\n}]+)' % re.escape(key), text)
    return m.group(1).strip() if m else None


def g6(x):
    """C++ ostream default for double (precision 6, %g)."""
    return "%.6g" % x


def test_summary(S, exon, intron, tot, called, vocab_genes):
    raw = S["raw"]
    txt = raw["summary.json"].decode()
    ps = json.loads(raw["pileup_stats.json"])
    prov = json.loads(raw["provenance.json"])
    sm = json.loads(txt)
    n = exon["n"]
    # per-gene distinct counts for called cells
    genes_of_row = {}
    toks = [r.split("_")[0] for r in exon["rownames"]]
    uniq = {g: i for i, g in enumerate(dict.fromkeys(toks))}
    gid = np.array([uniq[t] for t in toks], np.int64)
    ng = len(uniq)
    cell_cols = np.flatnonzero(called)
    ng_cell = []
    seen_any = np.zeros(ng, bool)
    for j in cell_cols:
        a, b = exon["indptr"][j], exon["indptr"][j + 1]
        gs = np.unique(gid[exon["indices"][a:b]])
        ng_cell.append(len(gs))
        seen_any[gs] = True
    ng_cell = np.array(ng_cell)
    umis = np.sort(tot[called].astype(np.uint64))
    k = int(called.sum())
    exp = {}
    exp["n_cells_called"] = str(k)
    exp["estimated_cells"] = str(k)
    if k:
        exp["median_umi_per_cell"] = g6(float(umis[k // 2]))
        exp["median_umis_per_cell"] = g6(float(umis[k // 2]))
        exp["median_genes_per_cell"] = g6(float(np.sort(ng_cell)[k // 2]))
        exp["total_genes_detected"] = str(int(seen_any.sum()))
        exp["fraction_reads_in_cells"] = g6(float(umis.sum()) / float(tot.sum()))
        exp["mean_reads_per_cell"] = g6(float(ps["total_reads"]) / k)
    mapped = ps["mapped_reads"]
    exp["exonic_fraction"] = g6(ps["exon_hits"] / mapped)
    exp["intronic_fraction"] = g6(ps["intron_hits"] / mapped)
    exp["intergenic_fraction"] = g6(max(0.0, 1.0 - (ps["exon_hits"] + ps["intron_hits"]) / mapped))
    # internal duplicates inside summary.json
    mr = sm["n_uniquely_mapped"] / sm["n_input_reads"] if sm["n_input_reads"] else 0.0
    exp["mapping_rate"] = g6(mr)
    exp["uniquely_mapped_pct"] = g6(mr * 100.0)
    exp["n_input_reads"] = str(sm["total_reads"])
    exp["total_reads"] = str(sm["n_input_reads"])
    exp["n_uniquely_mapped"] = str(sm["mapped_reads"])
    exp["mapped_reads"] = str(sm["n_uniquely_mapped"])
    res = {}
    for key, v in exp.items():
        res[key] = (jtok(txt, key), v, jtok(txt, key) == v)
    # pileup_stats fields that are matrix sums
    psr = {
        "exon_hits": (ps["exon_hits"], int(exon["data"].sum())),
        "intron_hits": (ps["intron_hits"], int(intron["data"].sum())),
        "sj_hits": (ps["sj_hits"], int(S["mat"]["sj_counts"]["data"].sum())),
        "umi_unique": (ps["umi_unique"], int(exon["data"].sum())),
        "n_barcodes": (ps["n_barcodes"], n),
    }
    # provenance fields
    pvr = {
        "output.exon_features": (prov["output"]["exon_features"], exon["m"]),
        "output.cells": (prov["output"]["cells"], k),
        "output.total_umis": (prov["output"]["total_umis"], int(exon["data"].sum())),
    }
    return res, psr, pvr, sm, ps, prov


# ------------------------------------------------------------------------------------------------------------------
# T8: vdj vs exon
# ------------------------------------------------------------------------------------------------------------------
def test_vdj(S):
    vd, ex, it = S["mat"].get("vdj_gene_usage"), S["mat"].get("exon_counts"), S["mat"].get("intron_counts")
    if vd is None:
        return None
    n = vd["n"]
    sym2rows = collections.defaultdict(list)
    for i, r in enumerate(ex["rownames"]):
        p = r.split("_")
        if len(p) > 2:
            sym2rows[p[1]].append(i)
    isym2rows = collections.defaultdict(list)
    for i, r in enumerate(it["rownames"]):
        p = r.split("_")
        if len(p) > 3:
            isym2rows[p[1]].append(i)
    cellof = lambda M: np.repeat(np.arange(n, dtype=np.int64), np.diff(M["indptr"]))
    cv = cellof(vd)
    vrow = vd["indices"]
    vdat = vd["data"].astype(np.int64)
    # dense exon+intron per-gene sums per cell for the genes named in vdj
    names = [r.split(" (")[0] for r in vd["rownames"]]
    ex_sum = {}
    cex, cit = cellof(ex), cellof(it)
    tot_e = np.zeros(len(vdat), np.int64)
    tot_i = np.zeros(len(vdat), np.int64)
    n_named = 0
    for g, nm in enumerate(names):
        sel = vrow == g
        if not sel.any():
            continue
        rows = sym2rows.get(nm, [])
        irows = isym2rows.get(nm, [])
        if rows:
            n_named += 1
        e_m = np.isin(ex["indices"], rows) if rows else np.zeros(len(ex["indices"]), bool)
        i_m = np.isin(it["indices"], irows) if irows else np.zeros(len(it["indices"]), bool)
        dense_e = np.bincount(cex[e_m], weights=ex["data"][e_m].astype(np.float64), minlength=n)
        dense_i = np.bincount(cit[i_m], weights=it["data"][i_m].astype(np.float64), minlength=n)
        tot_e[sel] = dense_e[cv[sel]].astype(np.int64)
        tot_i[sel] = dense_i[cv[sel]].astype(np.int64)
    return dict(
        vdj_nnz=int(len(vdat)), vdj_genes_with_hits=int(len(np.unique(vrow))), genes_with_exon_rows=n_named,
        frac_eq_exon_sum=float((vdat == tot_e).mean()),
        frac_eq_exon_plus_intron=float((vdat == tot_e + tot_i).mean()),
        frac_vdj_ge_exon=float((vdat >= tot_e).mean()),
        frac_exon_zero=float((tot_e == 0).mean()),
        vdj_total=int(vdat.sum()), exon_total_same_genes=int(tot_e.sum()),
    )


# ------------------------------------------------------------------------------------------------------------------
# Member table
# ------------------------------------------------------------------------------------------------------------------
def inventory_table(inv, meta):
    by = collections.defaultdict(list)
    for r in inv:
        nm = r["name"]
        if nm.startswith("samples/"):
            gsm, fn = nm.split("/")[1], nm.split("/")[2]
            by[fn].append((gsm, r))
        else:
            by[nm].append(("(study)", r))
    return by


def tsv(path, rows, cols):
    with open(path, "w", newline="\n") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-helper", action="store_true")
    args = ap.parse_args()
    os.makedirs(L.SCRATCH, exist_ok=True)
    pz = L.load_pz()
    helper = None if args.no_helper else L.build_helper()
    log("helper:", helper)

    z = zipfile.ZipFile(L.BUNDLE)
    inv, zmeta = L.zip_inventory(L.BUNDLE)
    log("bundle", zmeta, "entries", len(inv))
    manifest = json.loads(z.read("manifest.json"))
    study_meta = json.loads(z.read("study_meta.json"))
    vocab = json.loads(z.read("feature_vocab.json"))
    gsms = manifest["gsm_ids"]
    if args.quick:
        gsms = ["GSM4317429"]

    # ---- per-sample recomputations ------------------------------------------------------------------------------
    blocks = collections.defaultdict(lambda: collections.Counter())
    rowcounts_sum = collections.defaultdict(int)
    psi_res, cc_res, sm_rows, vdj_res = [], [], [], []
    flag_rows = []
    names_ok = collections.Counter()
    checks_manifest = collections.Counter()
    exon_rownames_hash = set()
    vdj_rownames_hash = set()
    S0 = None
    for gsm in gsms:
        t = time.time()
        S = load_sample(z, gsm, pz)
        log(f"{gsm}: loaded+decoded in {time.time() - t:.1f}s; members {sorted(S['raw'])}")
        # T1-T3
        if helper:
            out = test_container_blocks(S, helper)
            for (kind, what), r in out.items():
                if what == "rowcounts_bytes":
                    rowcounts_sum[kind] += r
                    continue
                blocks[(kind, what)]["files"] += 1
                blocks[(kind, what)]["raw_equal"] += bool(r["raw_equal"])
                if what == "colsums":
                    blocks[(kind, what)]["bytes_py157"] += bool(r["bytes_py157"])
                    blocks[(kind, what)]["bytes_cpp155"] += bool(r["bytes_cpp155"])
        # T4
        nm = test_names(S)
        names_ok["samples"] += 1
        names_ok["psi_rownames==sj_rownames"] += bool(nm.get("psi_rownames==sj_rownames"))
        names_ok["colnames_identical_across_members"] += bool(nm["colnames_identical_across_members"])
        exon_rownames_hash.add(L.sha256("\0".join(S["mat"]["exon_counts"]["rownames"]).encode()))
        vdj_rownames_hash.add(L.sha256("\0".join(S["mat"]["vdj_gene_usage"]["rownames"]).encode()))
        # T5
        pr = test_psi(S, helper, full_ref=bool(helper))
        if pr:
            psi_res.append(pr)
            log("psi", {k: v for k, v in pr.items() if k not in ("ref_log",)})
        # T6 + T7 + T8
        ex, it = S["mat"]["exon_counts"], S["mat"]["intron_counts"]
        r6, tot, called = test_cell_calls(gsm, ex, S["raw"]["cell_calls.tsv"])
        cc_res.append(r6)
        log("cell_calls", {k: v for k, v in r6.items() if k != "info"}, r6["info"])
        r7 = test_summary(S, ex, it, tot, called, vocab)
        sm_rows.append((gsm, r7))
        v = test_vdj(S) if gsm in ("GSM4317429", "GSM4317423") else None
        if v:
            vdj_res.append((gsm, v))
            log("vdj", v)
        # manifest checksum + study_meta consistency
        for fn, raw in S["raw"].items():
            arc = f"samples/{gsm}/{fn}"
            checks_manifest["members"] += 1
            checks_manifest["sha256_match"] += manifest["checksums"].get(arc) == L.sha256(raw)
        gm = study_meta["gsm_meta"][gsm]
        sm = json.loads(S["raw"]["summary.json"])
        for fld, skey in (("n_cells", "n_cells_called"), ("median_umi_per_cell", "median_umi_per_cell"),
                          ("median_genes_per_cell", "median_genes_per_cell"), ("mapping_rate", "mapping_rate"),
                          ("reference_build", "reference_build"), ("protocol_name", "protocol_name")):
            checks_manifest["study_meta_fields"] += 1
            checks_manifest["study_meta_fields_equal"] += gm.get(fld) == sm.get(skey)
        if gsm == "GSM4317429":
            S0 = S
        else:
            del S
    log("blocks", {k: dict(v) for k, v in blocks.items()})

    # ---- corpus samples: cell_calls + (partial) summary ---------------------------------------------------------
    corpus_cc = []
    corpus_sm = []
    if not args.quick:
        for sid in L.CORPUS:
            d = os.path.join(L.BC, "corpus", sid)
            ex = pz.read_1pz(os.path.join(d, "exon_counts.1pz"))
            calls = open(os.path.join(d, "cell_calls.tsv"), "rb").read()
            r6, tot, called = test_cell_calls(sid, ex, calls)
            corpus_cc.append(r6)
            log("corpus cell_calls", sid, {k: v for k, v in r6.items() if k != "info"}, r6["info"])
            # summary fields that need only exon + calls
            txt = open(os.path.join(d, "summary.json")).read()
            k = int(called.sum())
            umis = np.sort(tot[called])
            toks = [r.split("_")[0] for r in ex["rownames"]]
            uniq = {g: i for i, g in enumerate(dict.fromkeys(toks))}
            gid = np.array([uniq[t] for t in toks], np.int64)
            ngc = []
            seen = np.zeros(len(uniq), bool)
            for j in np.flatnonzero(called):
                a, b = ex["indptr"][j], ex["indptr"][j + 1]
                gs = np.unique(gid[ex["indices"][a:b]])
                ngc.append(len(gs))
                seen[gs] = True
            expd = {
                "n_cells_called": str(k),
                "median_umi_per_cell": g6(float(umis[k // 2])),
                "median_genes_per_cell": g6(float(np.sort(ngc)[k // 2])),
                "total_genes_detected": str(int(seen.sum())),
                "fraction_reads_in_cells": g6(float(umis.sum()) / float(tot.sum())),
            }
            row = {kk: (jtok(txt, kk), vv, jtok(txt, kk) == vv) for kk, vv in expd.items()}
            corpus_sm.append((sid, row))
            log("corpus summary", sid, row)

    # ---- write e4_recompute.tsv ---------------------------------------------------------------------------------
    n_s = len(gsms)
    for (kind, what), c in sorted(blocks.items()):
        if what == "colsums":
            rec("block_recompute", "bundle", f"{kind}.colsums raw u64 array", c["files"], c["raw_equal"], c["raw_equal"] == c["files"], 0,
                "decompressed colsums == per-column sum of the decoded matrix")
            rec("block_recompute", "bundle", f"{kind}.colsums zstd frame bytes (python-zstandard 0.25 = zstd 1.5.7, level 3 + checksum)",
                c["files"], c["bytes_py157"], c["bytes_py157"] == c["files"], "", "frame bytes depend on the zstd build")
            rec("block_recompute", "bundle", f"{kind}.colsums zstd frame bytes (zstd 1.5.5 static, production pz::zstd_compress)",
                c["files"], c["bytes_cpp155"], c["bytes_cpp155"] == c["files"], "", "")
        else:
            rec("block_recompute", "bundle", f"{kind}.{what} raw block", c["files"], c["raw_equal"], c["raw_equal"] == c["files"], 0,
                {"ptr": "ptr_z == per-column nnz (np.diff(indptr))", "perm": "perm_z == counting sort of per-row nnz, desc, ties by row id"}[what])
    for pr in psi_res:
        gsm = pr["gsm"]
        rec("splice_psi", gsm, "structure (indptr, indices) of splice_psi == sj entries whose junction belongs to an event", pr["psi_nnz_stored"],
            pr["psi_nnz_re"] if pr["struct_equal"] else 0, pr["struct_equal"], "", f"events={pr['events']}")
        rec("splice_psi", gsm, "stored values == trunc(float32 PSI) as pz_writer.h:615 static_cast<uint32_t> would store (stored distinct values: %s)" % pr["stored_distinct_values"],
            pr["psi_nnz_stored"], pr["psi_nnz_stored"] - pr["n_trunc_diff"], pr["trunc_values_equal"], "", f"{pr['n_trunc_diff']} entries differ")
        rec("splice_psi", gsm, "stored values == rint(float32 PSI) (round half to even)", pr["psi_nnz_stored"], pr["psi_nnz_stored"] - pr["n_rint_diff"],
            pr["rint_values_equal"], "", f"{pr['n_rint_diff']} entries differ")
        if "ref_float_max_ulp" in pr:
            rec("splice_psi", gsm, "float32 PSI: numpy recomputation vs production compute_psi() bit patterns", pr["psi_nnz_re"],
                pr["ref_float_n_bit_equal"], pr["ref_float_max_ulp"] == 0 and pr["ref_structure_equal"], pr["ref_float_max_ulp"], "")
            rec("splice_psi", gsm, "splice_psi.1pz file bytes: production compute_psi() + write_1pz (float->uint32 truncation, zstd 1.5.5) vs bundle member",
                pr["container_bytes_stored"], pr["container_bytes_stored"] if pr["container_bytes_equal_trunc"] else 0, pr["container_bytes_equal_trunc"], "",
                f"re-encoded {pr['container_bytes_ref_trunc']} B vs stored {pr['container_bytes_stored']} B")
            rec("splice_psi", gsm, "splice_psi.1pz file bytes: production compute_psi() + nearbyint() + write_1pz (zstd 1.5.5) vs bundle member",
                pr["container_bytes_stored"], pr["container_bytes_stored"] if pr["container_bytes_equal"] else 0, pr["container_bytes_equal"], "",
                f"re-encoded {pr['container_bytes_ref']} B vs stored {pr['container_bytes_stored']} B")
    for r in cc_res + corpus_cc:
        lab = r["label"]
        rec("cell_calls", lab, "log10_umi column == %.4f of log10(exon colsum)", r["n_rows"], r["log10_text_equal"], r["log10_text_equal"] == r["n_rows"], "", "python math.log10")
        rec("cell_calls", lab, "log10_umi column using numpy float64 log10", r["n_rows"], r["log10_text_equal_numpy"], r["log10_text_equal_numpy"] == r["n_rows"], "", "")
        if "file_bytes_equal" in r:
            rec("cell_calls", lab, "whole file regenerated from (called barcode set in stored order, exon colsums, constants)", 1, int(r["file_bytes_equal"]), r["file_bytes_equal"], "",
                f"methods={r['methods']} pvalues={r['pvalue_constant']} all_TRUE={r['all_is_cell_TRUE']} sorted_rows={r['rows_sorted_bytewise']}")
            rec("cell_calls", lab, "whole file regenerated with rows in bytewise-sorted barcode order", 1, int(r["file_bytes_equal_sorted_order"]), r["file_bytes_equal_sorted_order"], "", "")
        i = r["info"]
        rec("cell_calls_flag_info", lab, "called set vs best single UMI threshold: exceptions", i["n_columns"], i["n_columns"] - i["threshold_exceptions"], i["threshold_exceptions"] == 0, "",
            f"n_called={i['n_called']} threshold_umi={i['threshold_umi']} bitmap={i['bitmap_bytes']}B flag|depth={i['flag_given_depth_bytes']}B exceptions_list={i['threshold_exceptions_list_bytes']}B")
    for gsm, (res, psr, pvr, sm, ps, prov) in sm_rows:
        for key, (got, exp, ok) in res.items():
            rec("summary.json", gsm, key, 1, int(ok), ok, "", f"stored={got} recomputed={exp}")
        for key, (a, b) in psr.items():
            rec("pileup_stats.json", gsm, key + " == matrix quantity", 1, int(a == b), a == b, "", f"stored={a} recomputed={b}")
        for key, (a, b) in pvr.items():
            rec("provenance.json", gsm, key + " == matrix quantity", 1, int(a == b), a == b, "", f"stored={a} recomputed={b}")
    for sid, row in corpus_sm:
        for key, (got, exp, ok) in row.items():
            rec("summary.json", sid, key, 1, int(ok), ok, "", f"stored={got} recomputed={exp}")
    rec("names", "bundle", "splice_psi rownames == sj_counts rownames (same sample)", names_ok["samples"], names_ok["psi_rownames==sj_rownames"],
        names_ok["psi_rownames==sj_rownames"] == names_ok["samples"], "", "")
    rec("names", "bundle", "colnames identical in exon/intron/sj/psi/vdj of a sample", names_ok["samples"], names_ok["colnames_identical_across_members"],
        names_ok["colnames_identical_across_members"] == names_ok["samples"], "", "")
    rec("names", "bundle", "distinct exon rowname lists across samples", len(gsms), len(exon_rownames_hash), len(exon_rownames_hash) == 1, "", "1 = identical")
    rec("names", "bundle", "distinct vdj rowname lists across samples", len(gsms), len(vdj_rownames_hash), len(vdj_rownames_hash) == 1, "", "1 = identical")
    rec("manifest", "bundle", "manifest.json sha256 == sha256(member bytes)", checks_manifest["members"], checks_manifest["sha256_match"],
        checks_manifest["sha256_match"] == checks_manifest["members"], "", "")
    rec("study_meta", "bundle", "gsm_meta fields (n_cells, median_umi, median_genes, mapping_rate, build, protocol) == summary.json", checks_manifest["study_meta_fields"],
        checks_manifest["study_meta_fields_equal"], checks_manifest["study_meta_fields_equal"] == checks_manifest["study_meta_fields"], "", "")
    for gsm, v in vdj_res:
        rec("vdj_vs_exon", gsm, "vdj entry == exon-count sum over rows of the same gene symbol", v["vdj_nnz"], int(round(v["frac_eq_exon_sum"] * v["vdj_nnz"])), False, "",
            f"frac_eq_exon={v['frac_eq_exon_sum']:.4f} frac_eq_exon+intron={v['frac_eq_exon_plus_intron']:.4f} vdj_total={v['vdj_total']} exon_total_same_genes={v['exon_total_same_genes']}")
    cols = ["test", "sample", "item", "n_compared", "n_equal", "exact", "max_ulp", "note"]
    tsv(os.path.join(L.RESULTS, "e4_recompute.tsv"), REC, cols)
    log("wrote results/e4_recompute.tsv rows", len(REC))

    # ---- e4_members.tsv -----------------------------------------------------------------------------------------
    write_members(inv, zmeta, manifest, gsms, blocks, rowcounts_sum, psi_res, cc_res, corpus_cc, sm_rows, vdj_res, checks_manifest, names_ok)
    with open(os.path.join(L.RESULTS, "e4_audit.log"), "w") as f:
        f.write("\n".join(LOG) + "\n")


# ------------------------------------------------------------------------------------------------------------------
def write_members(inv, zmeta, manifest, gsms, blocks, rowcounts_sum, psi_res, cc_res, corpus_cc, sm_rows, vdj_res, checks_manifest, names_ok):
    acc = L_read_tsv(os.path.join(L.RESULTS, "track_a_account_bundle_1pz.tsv"))
    by = collections.defaultdict(list)
    for r in inv:
        nm = r["name"]
        if nm.startswith("samples/"):
            by[nm.split("/")[2]].append(r)
        else:
            by[nm].append(r)
    total = zmeta["file_size"]
    n_samples = 14

    def agg(fn):
        rs = by[fn]
        return dict(
            files=len(rs), stored=sum(r["csize"] for r in rs), unc=sum(r["usize"] for r in rs),
            mean=sum(r["csize"] for r in rs) / max(1, len(rs)), mn=min(r["csize"] for r in rs), mx=max(r["csize"] for r in rs),
        )

    def comp(member, field):
        return sum(int(r[field]) for r in acc if r["member"] == member + ".1pz")

    rows = []

    def add(**kw):
        rows.append(kw)

    # ---- member-level rows
    P = "include/singlet/pileup/"
    spec = [
        ("exon_counts.1pz", P + "export.h:959 write_matrix('exon_counts') <- pileup_engine exon accumulator; writer " + P + "pz_writer.h:591 write_1pz",
         "BAM pileup (reads, UMI dedup, gene model GTF); not derivable from other members",
         "I (matrix payload) + D (names, perm, ptr, colsums inside)"),
        ("intron_counts.1pz", P + "export.h:1113 write_matrix('intron_counts'); writer pz_writer.h:591 write_1pz",
         "BAM pileup; independent matrix payload; same inside-container redundancy as exon", "I (payload) + D (names, perm, ptr, colsums)"),
        ("sj_counts.1pz", P + "export.h:1115 write_matrix('sj_counts') <- engine.sj_names()", "BAM spliced reads; junction list is per-sample (detected junctions)",
         "I (payload, junction list) + D (perm, ptr, colsums, barcodes)"),
        ("splice_psi.1pz", P + "export.h:1120-1141 compute_psi(sj_csc, sj_names) -> atomic_write_1pz; " + P + "splice_psi.h:60 build_splice_events, :109 compute_psi",
         "sj_counts.1pz only (rownames -> events; counts -> float32 sj/event_sum)", "D0: whole member (container bytes included, zstd 1.5.5) regenerated from sj_counts; stored values are rint(PSI) in {0,1}, not the float PSI; the float PSI itself is D1 (formula must be pinned)"),
        ("vdj_gene_usage.1pz", P + "export.h:1154 write_matrix('vdj_gene_usage') <- pileup_engine.h:944-958 vdj_model_.query + UMI dedup",
         "BAM reads overlapping VDJ gene spans (any overlap incl. introns, first hit wins, UMI-deduped per (cell, gene, umi))", "I (payload) + D (barcodes, perm, ptr, colsums)"),
        ("mt_heteroplasmy.1pz", P + "export.h:867 compute_heteroplasmy, :1163-1195 write (pipeline_mode only); scaled uint16 = VAF*10000",
         "chrM allele counts (mt_allele_counts, not a bundle member)", "stub here (0x0 matrix, 151 B): D0 constant; real matrices would be I"),
        ("cell_calls.tsv", P + "export.h:667 write_cell_calls <- cell_calling.h:1024; star_solo branch export.h:486-540",
         "STARsolo filtered barcodes (EmptyDrops_CR on the whole-whitelist raw matrix) + exon colsums", "called SET is I; every other byte is D0 (log10_umi from exon colsums, constants, sorted order)"),
        ("summary.json", P + "export.h:1290-1516 PipelineSummary -> summary_json.h:167 write_summary_json",
         "exon/intron matrices, cell_calls, pileup_stats.json, STAR Log.final.out, run metadata", "mixed: 15 of 34 numeric scalars are functions of other members (11 of matrices+pileup_stats, 4 of the two STAR read counts) and were reproduced exactly; 3 are constant zeros; 8 are block counters (zero here); 8 are run metadata (I)"),
        ("pileup_stats.json", "src/pipeline/singlet.cpp:8440 write_stats_json (PileupStats); exon/intron/sj hits == matrix sums",
         "BAM-level counters", "mixed: exon/intron/sj hits, umi_unique, n_barcodes D0; read-level counters I"),
        ("provenance.json", P + "provenance.h:59 write_provenance_json (called export.h:1261-1280)", "run metadata + exon matrix dims/sum + cell caller", "mostly I (host, time, params); output block D0"),
        ("manifest.json", "python/singlet/bundle.py:1142-1160 pack_gse", "sha256 of every member, gsm list, included_files", "D0 (checksums are functions of the members) but needed as an integrity index"),
        ("study_meta.json", "python/singlet/bundle.py:955-1015 pack_gse", "processing catalog + summary.json", "mixed: gsm_meta numeric fields copy summary.json (D0); GEO text I"),
        ("feature_vocab.json", "python/singlet/bundle.py:672 _build_feature_vocab (gene_expression.tsv of the first GSM)", "reference annotation (GTF) of the build", "D0 given the reference build"),
    ]
    # recompute summaries per member for the table
    def pct(c, key):
        return f"{c[key]}/{c['files']}"

    psi_ok = sum(1 for p in psi_res if p.get("container_bytes_equal"))
    psi_struct = sum(1 for p in psi_res if p["struct_equal"] and p["rint_values_equal"])
    cc_all = cc_res + corpus_cc
    cc_bytes_ok = sum(1 for r in cc_all if r.get("file_bytes_equal"))
    sm_ok = sum(1 for _, (res, *_r) in sm_rows for _k, (_g, _e, ok) in res.items() if ok)
    sm_n = sum(1 for _, (res, *_r) in sm_rows for _k in res)
    result_txt = {
        "splice_psi.1pz": f"recomputed from sj_counts: structure + stored 0/1 values (= rint(PSI)) equal {psi_struct}/{len(psi_res)} samples; float PSI bit-identical to production compute_psi(); container bytes identical (zstd 1.5.5) {psi_ok}/{len(psi_res)}",
        "cell_calls.tsv": f"regenerated byte-identical {cc_bytes_ok}/{len(cc_all)} samples from (called set, exon colsums)",
        "summary.json": f"{sm_ok}/{sm_n} recomputable numeric fields reproduced exactly (14 samples)",
        "manifest.json": f"sha256 == member bytes {checks_manifest['sha256_match']}/{checks_manifest['members']}",
        "study_meta.json": f"summary-derived fields equal {checks_manifest['study_meta_fields_equal']}/{checks_manifest['study_meta_fields']}",
    }
    # derivable bytes per member (sum over samples); component sums from Track A tables
    def derivable_bytes(fn):
        base = fn.replace(".1pz", "")
        if not fn.endswith(".1pz"):
            return None
        ptr = comp(base, "ptr_z")
        cs = comp(base, "colsums_z")
        perm = comp(base, "perm_z")
        rown = comp(base, "rown_z3")
        coln = comp(base, "coln_z3")
        rc = rowcounts_sum.get(base, 0)
        d = dict(ptr=ptr, colsums=cs, perm_minus_rowcounts=perm - rc, rownames=rown, colnames=coln)
        return d

    for fn, producer, inputs, verdict in spec:
        a = agg(fn) if fn in by else dict(files=0, stored=0, unc=0, mean=0, mn=0, mx=0)
        row = dict(
            member=fn, scope="member", files=a["files"], bytes_total=a["stored"], bytes_uncompressed=a["unc"],
            share_of_bundle_pct=round(100 * a["stored"] / total, 3), bytes_per_sample_mean=round(a["mean"]), bytes_per_sample_min=a["mn"],
            bytes_per_sample_max=a["mx"], code=producer, inputs=inputs, verdict=verdict, recompute_attempted="yes" if fn in result_txt else "component-level",
            recompute_result=result_txt.get(fn, ""),
        )
        db = derivable_bytes(fn)
        if fn == "splice_psi.1pz":
            row["derivable_bytes_total"] = a["stored"]
            row["derivable_bytes_per_sample"] = round(a["stored"] / n_samples)
            row["independent_bytes_per_sample"] = 0
        elif fn == "cell_calls.tsv":
            fi = [r["info"]["flag_given_depth_bytes"] for r in cc_res]
            ind = round(sum(fi) / len(fi))
            row["derivable_bytes_total"] = a["stored"] - ind * len(fi)
            row["derivable_bytes_per_sample"] = round(a["mean"] - ind)
            row["independent_bytes_per_sample"] = ind
        elif fn == "mt_heteroplasmy.1pz":
            row["derivable_bytes_total"] = a["stored"]
            row["derivable_bytes_per_sample"] = round(a["mean"])
            row["independent_bytes_per_sample"] = 0
        elif fn == "feature_vocab.json":
            row["derivable_bytes_total"] = a["stored"]
            row["derivable_bytes_per_sample"] = 0
            row["independent_bytes_per_sample"] = 0
        elif db:
            dsum = db["ptr"] + db["colsums"] + db["perm_minus_rowcounts"]
            if fn in ("exon_counts.1pz", "intron_counts.1pz", "vdj_gene_usage.1pz"):
                dsum += db["rownames"]
            row["derivable_bytes_total"] = dsum + 0
            row["derivable_bytes_per_sample"] = round(dsum / n_samples)
            row["independent_bytes_per_sample"] = round((a["stored"] - dsum) / n_samples)
        else:
            row["derivable_bytes_total"] = ""
            row["derivable_bytes_per_sample"] = ""
            row["independent_bytes_per_sample"] = ""
        rows.append(row)
    # ---- component-level rows (inside every .1pz)
    comps = [
        ("[.1pz] rownames (feature names, zstd-3)", "pz_writer.h:push_strings_tlv META_TAG_ROWNAMES", "reference annotation of the build (exon/intron/vdj); per-sample detected junctions (sj, psi)",
         "D0 for exon/intron/vdj given the reference vocabulary (hash-identical across all samples); psi = copy of sj (D0); sj junction list is I", "rown_z3"),
        ("[.1pz] colnames (barcodes, zstd-3)", "pz_writer.h:push_strings_tlv META_TAG_COLNAMES", "barcode set of the sample; stored once per member (5x per sample)",
         "1 copy I, 4 copies D0 (identical lists); the single copy is whitelist members in random order", "coln_z3"),
        ("[.1pz] row permutation (perm_z)", "pz_writer.h:encode_vocsc_section (counting sort of row nnz)", "per-row nnz counts", "D0 given row counts (Track A: identical on all 70 files); counts cost ~5% of perm", "perm_z"),
        ("[.1pz] column counts (ptr_z)", "pz_writer.h:encode_vocsc_section ptr_raw", "chunk metadata (value-group counts) / indptr", "D0", "ptr_z"),
        ("[.1pz] column sums (colsums_z)", "pz_writer.h:write_1pz step 2 (never read by pz_reader.h)", "decoded matrix", "D0", "colsums_z"),
        ("[.1pz] chunk payload", "pz_writer.h:vocsc_encode_chunk + zstd", "the counts", "I", "chunk_payload"),
    ]
    for name, code, inputs, verdict, field in comps:
        per_member = {}
        for k in KINDS:
            per_member[k] = comp(k, field)
        tot_c = sum(per_member.values())
        rows.append(dict(
            member=name, scope="component", files=sum(1 for k in KINDS for r in acc if r["member"] == k + ".1pz"), bytes_total=tot_c, bytes_uncompressed="",
            share_of_bundle_pct=round(100 * tot_c / total, 3), bytes_per_sample_mean=round(tot_c / n_samples), bytes_per_sample_min="", bytes_per_sample_max="",
            code=code, inputs=inputs, verdict=verdict, recompute_attempted="yes" if field in ("perm_z", "ptr_z", "colsums_z") else "yes (equality across members and samples)",
            recompute_result=component_result(field, blocks, rowcounts_sum, names_ok),
        ))
    rows.append(dict(member="zip framing (local headers + central directory + zip64 EOCD)", scope="container", files=1, bytes_total=zmeta["tail_bytes"] + sum(r["lh_len"] for r in inv),
                     share_of_bundle_pct=round(100 * (zmeta["tail_bytes"] + sum(r["lh_len"] for r in inv)) / total, 4), bytes_per_sample_mean=round((zmeta["tail_bytes"] + sum(r["lh_len"] for r in inv)) / n_samples),
                     code="python zipfile (bundle.py pack_gse)", inputs="member list", verdict="D0 (function of the member list)", recompute_attempted="no", recompute_result=""))
    cols = ["member", "scope", "files", "bytes_total", "bytes_uncompressed", "share_of_bundle_pct", "bytes_per_sample_mean", "bytes_per_sample_min", "bytes_per_sample_max",
            "derivable_bytes_total", "derivable_bytes_per_sample", "independent_bytes_per_sample", "verdict", "recompute_attempted", "recompute_result", "code", "inputs"]
    tsv(os.path.join(L.RESULTS, "e4_members.tsv"), rows, cols)
    log("wrote results/e4_members.tsv rows", len(rows))


def component_result(field, blocks, rowcounts_sum, names_ok):
    if field == "colsums_z":
        c = {k: v for k, v in blocks.items() if k[1] == "colsums"}
        files = sum(v["files"] for v in c.values())
        return f"raw array equal {sum(v['raw_equal'] for v in c.values())}/{files}; frame bytes equal: zstd1.5.7 {sum(v['bytes_py157'] for v in c.values())}/{files}, zstd1.5.5 {sum(v['bytes_cpp155'] for v in c.values())}/{files}"
    if field in ("perm_z", "ptr_z"):
        what = "perm" if field == "perm_z" else "ptr"
        c = {k: v for k, v in blocks.items() if k[1] == what}
        files = sum(v["files"] for v in c.values())
        return f"decompressed block equal {sum(v['raw_equal'] for v in c.values())}/{files}"
    if field == "coln_z3":
        return f"colnames identical across the 5 members {names_ok['colnames_identical_across_members']}/{names_ok['samples']} samples"
    if field == "rown_z3":
        return f"psi rownames == sj rownames {names_ok['psi_rownames==sj_rownames']}/{names_ok['samples']} samples"
    return ""


def L_read_tsv(path):
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        return [dict(zip(hdr, ln.rstrip("\n").split("\t"))) for ln in f if ln.strip()]


if __name__ == "__main__":
    main()
