"""E4 Part B (latency): transfer-time model for today's layout versus the tiered layout.

Reads results/e4_tiers.tsv (bytes per sample x member x subset x accounting variant) and results/e4_tiers_samples.json.
Writes results/e4_latency.tsv.

Model (stated so that it can be argued with)
  * One HTTP/2 connection, already established (a cold TLS 1.3 connection adds 2 RTT = 100 ms to every row alike).
  * Requests issued in the same round are multiplexed and share the link:  t(round) = RTT + bytes(round) / bandwidth.
    Rounds are strictly sequential (a request whose range depends on a previous answer starts a new round).
    No TCP slow start, no server think time, no decode time. Links: 100 Mbit/s and 1 Gbit/s, RTT 50 ms.
  * The container is the .singlet zip64. Round 1 of every reader is one suffix range request for the last 64 KiB of the
    archive: central directory + end records + manifest.json + study_meta.json (manifest.json is written last, so it sits
    next to the directory). In the proposed layout the per-sample derived-on-read manifests are also placed at the tail,
    so that round 1 also returns the chunk index of every member.
  * Corpus samples are modelled as a one-sample archive holding the members that exist in corpus/<id>/ (exon, intron,
    cell_calls.tsv, summaries); the 14-sample bundle has the same tail window.

Scenarios (the 'cells' are the called cells of the sample, exon + intron interval matrices together)
  first_chunk   the first 1024 called cells decoded and usable (one block)
  all_cells     every called cell
  everything    every column, called cells and empties

Variants
  today              exon_counts.1pz + intron_counts.1pz + cell_calls.tsv as shipped
  tiered_i           member 1 / member 2 written by today's writer (names, permutation, ptr, colsums still inside)
  tiered_ii          new container: names, permutation, ptr, colsums removed (Track A costs); payload as today
  tiered_iii         tiered_ii with the best payload code length on disk (Track D CV code lengths per subset)
  untiered_ii/iii    same containers but one member holding all columns (isolates the effect of tiering)

    python -P tools/e4_latency.py
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_lib as L  # noqa: E402

RTT = 0.050
LINKS = {"100Mbit": 100e6 / 8, "1Gbit": 1e9 / 8}
TAIL = 65536  # suffix window of round 1
HEAD = 8192  # speculative range at the start of a member to read local zip header + 96 B .1pz header
MANIFEST = 1500  # derived-on-read manifest of one sample (recipes, digests, run metadata) in the proposed layout
LOCAL_HDR = 100  # zip local header + zip64 extra per member (bytes)


def f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load():
    with open(os.path.join(L.RESULTS, "e4_tiers.tsv")) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    T = {}
    for r in rows:
        T[(r["sample"], r["member"], r["subset"])] = r
    with open(os.path.join(L.RESULTS, "e4_tiers_samples.json")) as fh:
        info = json.load(fh)
    return T, info


def fixed_bytes(r):
    """Header + chunk index + footer + vocabulary id of one matrix in the new container."""
    return 96 + f(r["ctable"]) + 16 + 8


def unify(T, sample, members, subset, key, extra_rown=0):
    """Bytes of the unified column-block-major member holding `members` (matrices with identical columns) for one subset.
    key = 'bytes_ii' | 'bytes_iii' | 'bytes_ii_canon'. Header and barcodes are counted once, not once per matrix."""
    tot = 0.0
    bwl_once = 0.0
    fixed_once = 0.0
    for mem in members:
        r = T[(sample, mem, subset)]
        v = f(r[key]) if r[key] != "" else f(r["bytes_ii"])
        tot += v - fixed_bytes(r) - f(r["barcodes_wl_bytes"])
        bwl_once = max(bwl_once, f(r["barcodes_wl_bytes"]))
        fixed_once = max(fixed_once, fixed_bytes(r))
        if mem == "sj":
            extra_rown = f(r["rown_z3"])  # per-sample junction list: independent information, kept
    return tot + bwl_once + fixed_once + extra_rown


def first_block_new(T, sample, members, key):
    """Bytes of block 0 (first <=1024 called cells) in the unified cells member (ii/iii)."""
    tot = 0.0
    bwl = 0.0
    rc = 0.0
    for mem in members:
        r = T[(sample, mem, "cells")]
        nb = max(1, int(f(r["n_chunks"])))
        n = f(r["ncols"])
        frac = min(1.0, 1024.0 / n) if n else 1.0
        if key == "bytes_ii":
            payload = f(r["first_chunk_blob"])
        else:
            # block 0 holds the deepest cells (columns keep today's depth-sorted order), so scale the payload by the
            # share that chunk 0 actually has in today's cells file, not by 1/nb
            share = f(r["first_chunk_blob"]) / f(r["chunks_bytes"]) if f(r["chunks_bytes"]) else 1.0 / nb
            payload = (f(r["bytes_iii"]) - fixed_bytes(r) - f(r["side_kj_bytes"]) - f(r["rowcounts_bytes"]) - f(r["barcodes_wl_bytes"])) * share
            payload += f(r["side_kj_bytes"]) * frac
        tot += payload
        rc += f(r["rowcounts_bytes"])  # per-row nnz counts / margins are needed before any chunk can be decoded
        bwl = max(bwl, f(r["barcodes_wl_bytes"]) * frac)
    return tot + rc + bwl + 96 + 8 + 16


def today_layout(sample, kinds):
    """Where do the called cells sit in the shipped column order? Parses only header, chunk table and name block.
    Returns {kind: dict(n_chunks, last_cell_chunk, selective_chunk_bytes, total_chunk_bytes)}."""
    import numpy as np
    sid = sample
    d = os.path.join(L.BC, "corpus", sid)
    calls = set(ln.split("\t")[0] for ln in open(os.path.join(d, "cell_calls.tsv")).read().split("\n")[1:] if ln)
    out = {}
    for kind in kinds:
        if kind in ("exon", "intron"):
            path = os.path.join(d, f"{kind}_counts.1pz")
        else:
            fn = {"sj": "sj_counts", "vdj": "vdj_gene_usage", "psi": "splice_psi"}[kind]
            path = os.path.join(L.SCRATCH, "GSM4317429", fn + ".1pz")
        buf = open(path, "rb").read()
        h, o = L.parse_pz(buf)
        tlv = L.parse_tlv(L.zdec(buf, h["metadata_offset"], h["metadata_z_sz"]))
        called = np.fromiter((b in calls for b in tlv["colnames"]), bool, len(tlv["colnames"]))
        last = int(np.flatnonzero(called).max()) // 1024  # index of the last chunk that holds a called cell
        ct = o["ctable"]
        out[kind] = dict(n_chunks=h["num_chunks"], last_cell_chunk=last + 1, selective_chunk_bytes=int(ct[: last + 1].sum()), total_chunk_bytes=int(ct.sum()),
                         last_called_col=int(np.flatnonzero(called).max()) + 1, n_called=int(called.sum()), n_cols=h["n"],
                         prefix_bytes=96 + h["perm_z_sz"] + h["ptr_z_sz"] + 4 * h["num_chunks"], meta_z=h["metadata_z_sz"], file_bytes=len(buf))
    return out


def t_model(rounds, bw):
    return sum(RTT + b / bw for b in rounds)


LAYOUT = []


def main():
    T, info = load()
    out = []

    def emit(sample, scenario, variant, rounds_bytes, requests):
        for ln, bw in LINKS.items():
            out.append(dict(sample=sample, scenario=scenario, variant=variant, link=ln, rounds=len(rounds_bytes), requests=requests,
                            bytes=int(sum(rounds_bytes)), mb=round(sum(rounds_bytes) / 1e6, 3), t_ms=round(1000 * t_model(rounds_bytes, bw), 1)))

    samples = sorted({k[0] for k in T})
    for s in samples:
        members = [m for m in ("exon", "intron") if (s, m, "orig") in T]
        if len(members) < 2:
            continue
        inf = info[s]
        C = inf["cell_calls_deflated"] if s != "tumor_3p" else inf.get("cell_calls_deflated_bundle", inf["cell_calls_deflated"])
        orig = {m: T[(s, m, "orig")] for m in members}
        E = {m: f(orig[m]["file_bytes_i"]) for m in members}
        # ---------------- today
        prefix = sum(f(orig[m]["prefix_bytes_today"]) for m in members)
        meta = sum(f(orig[m]["meta_z"]) for m in members)
        kb = sum(f(orig[m]["first_cells_chunk_bytes"]) for m in members)
        emit(s, "first_chunk", "today", [TAIL, C + 2 * HEAD, prefix + meta + kb], 1 + 1 + 2 + 6)
        allb = sum(E.values()) + C + len(members) * LOCAL_HDR
        lay = today_layout(s, members)
        LAYOUT.extend(dict(sample=s, member=k, **v) for k, v in lay.items())
        sel = sum(v["selective_chunk_bytes"] for v in lay.values())
        # today_selective: header, then permutation/ptr/chunk table + names + only the chunks that hold called cells
        # (oracle: the reader is assumed to know the last chunk with a called cell; real readers need one more round)
        emit(s, "all_cells", "today_whole", [TAIL, allb], 1 + 3)
        emit(s, "all_cells", "today_selective", [TAIL, C + 2 * HEAD, prefix + meta + sel], 1 + 1 + 2 + 6)
        emit(s, "everything", "today", [TAIL, allb], 1 + 3)
        # ---------------- tiered (i): today's writer on the two subsets
        cells_f = {m: T[(s, m, "cells")] for m in members}
        empt_f = {m: T[(s, m, "empties")] for m in members}
        p_i = sum(f(cells_f[m]["prefix_bytes_today"]) for m in members)
        m_i = sum(f(cells_f[m]["meta_z"]) for m in members)
        c0 = sum(f(cells_f[m]["first_chunk_blob"]) for m in members)
        emit(s, "first_chunk", "tiered_i", [TAIL, 2 * HEAD, p_i + m_i + c0], 1 + 2 + 6)
        cb = sum(f(cells_f[m]["file_bytes_i"]) for m in members) + len(members) * LOCAL_HDR
        eb = sum(f(empt_f[m]["file_bytes_i"]) for m in members) + len(members) * LOCAL_HDR
        emit(s, "all_cells", "tiered_i", [TAIL, cb], 1 + 2)
        emit(s, "everything", "tiered_i", [TAIL, cb + eb], 1 + 4)
        # ---------------- tiered (ii), (iii)
        for key, vname in (("bytes_ii", "tiered_ii"), ("bytes_iii", "tiered_iii")):
            if key == "bytes_iii" and cells_f[members[0]][key] == "":
                continue
            CM = unify(T, s, members, "cells", key) + LOCAL_HDR
            EM = unify(T, s, members, "empties", key) + LOCAL_HDR
            blk = first_block_new(T, s, members, key)
            emit(s, "first_chunk", vname, [TAIL + MANIFEST, blk], 2)
            emit(s, "all_cells", vname, [TAIL + MANIFEST, CM], 2)
            emit(s, "everything", vname, [TAIL + MANIFEST, CM + EM], 3)
            # untiered containers: one member with every column (cells interleaved with empties as today)
            ALL = unify(T, s, members, "all", key) + LOCAL_HDR
            n_all = f(orig[members[0]]["ncols"])
            # first 1024 cells need the chunks that contain them: same chunk set as today, payload scaled by the codec ratio
            frac = 0.0
            for m in members:
                chunks_o = f(orig[m]["chunks_bytes"])
                a = T[(s, m, "all")]
                new_payload_all = (f(a[key]) - fixed_bytes(a) - f(a["rowcounts_bytes"]) - f(a["barcodes_wl_bytes"]) - f(a["side_kj_bytes"]))
                frac += (f(orig[m]["first_cells_chunk_bytes"]) / chunks_o if chunks_o else 0) * new_payload_all
            rc_all = sum(f(T[(s, m, "all")]["rowcounts_bytes"]) for m in members)
            nk = f(orig[members[0]]["first_cells_n_chunks"])
            bar = f(T[(s, members[0], "all")]["barcodes_wl_bytes"]) * min(1.0, nk * 1024 / n_all)
            if key == "bytes_ii":  # (iii) per-chunk code lengths for a mixed cell/empty region are not on disk
                emit(s, "first_chunk", "un" + vname, [TAIL + MANIFEST, frac + rc_all + bar + 120], 2)
            # all called cells from the single member: columns keep today's (roughly depth-sorted) order, so the reader
            # fetches chunks 0..last chunk that holds a called cell (the manifest carries the chunk index)
            n_sel = min(n_all, lay[members[0]]["last_cell_chunk"] * 1024)
            c_sel = n_sel / n_all
            sel_total = 0.0
            for m in members:
                a = T[(s, m, "all")]
                pay = f(a[key]) - fixed_bytes(a) - f(a["rowcounts_bytes"]) - f(a["barcodes_wl_bytes"]) - f(a["side_kj_bytes"])
                fr = lay[m]["selective_chunk_bytes"] / lay[m]["total_chunk_bytes"]
                sel_total += pay * fr + f(a["rowcounts_bytes"]) + f(a["side_kj_bytes"]) * c_sel
            sel_total += f(T[(s, members[0], "all")]["barcodes_wl_bytes"]) * c_sel + fixed_bytes(T[(s, members[0], "all")]) + LOCAL_HDR
            if key == "bytes_ii":  # (iii) per-chunk code lengths for a mixed cell/empty region are not on disk
                emit(s, "all_cells", "un" + vname, [TAIL + MANIFEST, sel_total], 2)
            if key == "bytes_ii":  # (iii) per-chunk code lengths for a mixed cell/empty region are not on disk
                emit(s, "everything", "un" + vname, [TAIL + MANIFEST, ALL], 2)
    # ---------------- the bundle sample with every modality (tumor_3p + sj + vdj; psi derived, mt stub dropped)
    s = "tumor_3p"
    if (s, "sj", "cells") in T:
        inf = info[s]
        mem4 = ["exon", "intron", "sj", "vdj"]
        orig = {m: T[(s, m, "orig")] for m in mem4}
        today_all = sum(f(orig[m]["file_bytes_i"]) for m in mem4) + inf["psi_bytes"] + inf["mt_bytes"] + inf["json_bytes"] + inf["cell_calls_deflated_bundle"] + 10 * LOCAL_HDR
        lay5 = today_layout(s, ["exon", "intron", "sj", "psi", "vdj"])
        other = inf["mt_bytes"] + inf["json_bytes"] + inf["cell_calls_deflated_bundle"] + 6 * LOCAL_HDR
        sel5 = sum(v["prefix_bytes"] + v["meta_z"] + v["selective_chunk_bytes"] for v in lay5.values())
        emit(s + "+sj+vdj+psi", "all_cells", "today_whole", [TAIL, today_all], 1 + 9)
        emit(s + "+sj+vdj+psi", "all_cells", "today_selective", [TAIL, inf["cell_calls_deflated_bundle"] + 5 * HEAD, sel5], 1 + 1 + 5 + 15)
        emit(s + "+sj+vdj+psi", "everything", "today", [TAIL, today_all], 1 + 9)
        cells_i = sum(f(T[(s, m, "cells")]["file_bytes_i"]) for m in mem4) + 4 * LOCAL_HDR
        empt_i = sum(f(T[(s, m, "empties")]["file_bytes_i"]) for m in mem4) + 4 * LOCAL_HDR
        emit(s + "+sj+vdj+psi", "all_cells", "tiered_i", [TAIL, cells_i], 1 + 4)
        emit(s + "+sj+vdj+psi", "everything", "tiered_i", [TAIL, cells_i + empt_i], 1 + 8)
        CM = unify(T, s, mem4, "cells", "bytes_ii") + LOCAL_HDR
        EM = unify(T, s, mem4, "empties", "bytes_ii") + LOCAL_HDR
        emit(s + "+sj+vdj+psi", "all_cells", "tiered_ii", [TAIL + MANIFEST, CM], 2)
        emit(s + "+sj+vdj+psi", "everything", "tiered_ii", [TAIL + MANIFEST, CM + EM], 3)
        # (iii) only for exon/intron have payload models; sj and vdj stay at (ii)
        CM3 = unify(T, s, mem4, "cells", "bytes_iii") + LOCAL_HDR
        EM3 = unify(T, s, mem4, "empties", "bytes_iii") + LOCAL_HDR
        emit(s + "+sj+vdj+psi", "all_cells", "tiered_iii", [TAIL + MANIFEST, CM3], 2)
        emit(s + "+sj+vdj+psi", "everything", "tiered_iii", [TAIL + MANIFEST, CM3 + EM3], 3)
    # 'today' for all_cells = the faster of the two reader strategies on each link
    best = {}
    for r in out:
        if r["scenario"] == "all_cells" and r["variant"] in ("today_whole", "today_selective"):
            k = (r["sample"], r["link"])
            if k not in best or r["t_ms"] < best[k]["t_ms"]:
                best[k] = r
    for k, r in best.items():
        out.append(dict(r, variant="today"))
    with open(os.path.join(L.RESULTS, "e4_today_layout.tsv"), "w", newline="\n") as fh:
        lc = ["sample", "member", "n_cols", "n_called", "last_called_col", "n_chunks", "last_cell_chunk", "selective_chunk_bytes", "total_chunk_bytes"]
        fh.write("\t".join(lc) + "\n")
        for r in LAYOUT:
            fh.write("\t".join(str(r[c]) for c in lc) + "\n")
    cols = ["sample", "scenario", "variant", "link", "rounds", "requests", "bytes", "mb", "t_ms"]
    # speedup vs today
    base = {(r["sample"], r["scenario"], r["link"]): r["t_ms"] for r in out if r["variant"] == "today"}
    bytes_base = {(r["sample"], r["scenario"], r["link"]): r["bytes"] for r in out if r["variant"] == "today"}
    for r in out:
        b = base.get((r["sample"], r["scenario"], r["link"]))
        r["speedup_t"] = round(b / r["t_ms"], 2) if b else ""
        bb = bytes_base.get((r["sample"], r["scenario"], r["link"]))
        r["bytes_ratio"] = round(r["bytes"] / bb, 3) if bb else ""
    cols += ["speedup_t", "bytes_ratio"]
    with open(os.path.join(L.RESULTS, "e4_latency.tsv"), "w", newline="\n") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in out:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    print("wrote results/e4_latency.tsv", len(out))


if __name__ == "__main__":
    main()
