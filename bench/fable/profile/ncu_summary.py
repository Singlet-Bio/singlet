#!/usr/bin/env python3
"""E1 profile: summarise the Nsight Compute exports of one e1_ncu job directory (stdlib only).

Usage: ncu_summary.py OUTDIR [OUTDIR2 ...]   (OUTDIR = $W/jobs/fable/profile/results/ncu-<jobid>)
Inputs per label L (A, M, B, monog, monol): ncu_L_raw.csv (--page raw --csv), ncu_L_<kernel>_src.csv (--page source --csv
--print-source cuda,sass). Writes OUTDIR/e1_ncu_summary.json, OUTDIR/e1_ncu_kernels.csv, OUTDIR/e1_ncu_top_lines.csv and prints tables.
"""
import csv, glob, json, os, re, sys

KEYS = [  # (column, metric name candidates)
    ("duration_ms", ["gpu__time_duration.sum"]),
    ("regs", ["launch__registers_per_thread"]),
    ("block", ["launch__block_size"]),
    ("grid", ["launch__grid_size"]),
    ("waves", ["launch__waves_per_multiprocessor"]),
    ("occ_theor_pct", ["sm__maximum_warps_per_active_cycle_pct"]),
    ("occ_achieved_pct", ["sm__warps_active.avg.pct_of_peak_sustained_active"]),
    ("lim_regs_blocks", ["launch__occupancy_limit_registers"]),
    ("lim_warps_blocks", ["launch__occupancy_limit_warps"]),
    ("lim_smem_blocks", ["launch__occupancy_limit_shared_mem"]),
    ("lim_blocks", ["launch__occupancy_limit_blocks"]),
    ("local_bytes_per_thread", ["launch__stack_size"]),
    ("sm_busy_pct", ["sm__throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("mem_busy_pct", ["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("l1_thru_pct", ["l1tex__throughput.avg.pct_of_peak_sustained_active"]),
    ("l2_thru_pct", ["lts__throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("dram_thru_pct", ["dram__throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("dram_read_GB", ["dram__bytes_read.sum"]),
    ("dram_write_GB", ["dram__bytes_write.sum"]),
    ("l1_hit_pct", ["l1tex__t_sector_hit_rate.pct"]),
    ("l2_hit_pct", ["lts__t_sector_hit_rate.pct"]),
    ("issue_slots_busy_pct", ["sm__inst_issued.avg.pct_of_peak_sustained_active"]),
    ("issue_active_pct", ["smsp__issue_active.avg.pct_of_peak_sustained_active"]),
    ("ipc", ["sm__inst_executed.avg.per_cycle_active"]),
    ("warp_cyc_per_inst", ["smsp__average_warp_latency_per_inst_issued.ratio", "smsp__average_warps_issue_stalled_per_issue_active.ratio"]),
    ("warps_active_per_sched", ["smsp__warps_active.avg.per_cycle_active"]),
    ("warps_eligible_per_sched", ["smsp__warps_eligible.avg.per_cycle_active"]),
    ("simt_threads_per_inst", ["smsp__thread_inst_executed_per_inst_executed.ratio"]),
    ("branch_uniform_pct", ["smsp__sass_average_branch_targets_threads_uniform.pct"]),
    ("inst_executed", ["smsp__inst_executed.sum", "sm__inst_executed.sum"]),
    ("gld_sectors", ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"]),
    ("gld_requests", ["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"]),
    ("gld_hit_sectors", ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"]),
    ("gst_sectors", ["l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"]),
    ("gst_requests", ["l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"]),
    ("lld_sectors", ["l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"]),
    ("lld_requests", ["l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum"]),
    ("lst_sectors", ["l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"]),
    ("lst_requests", ["l1tex__t_requests_pipe_lsu_mem_local_op_st.sum"]),
    ("gatom_requests", ["l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum"]),
    ("gred_requests", ["l1tex__t_requests_pipe_lsu_mem_global_op_red.sum"]),
    ("l2_read_sectors", ["lts__t_sectors_srcunit_tex_op_read.sum"]),
    ("l2_write_sectors", ["lts__t_sectors_srcunit_tex_op_write.sum"]),
    ("sm_cycles_active_avg", ["sm__cycles_active.avg"]),
    ("sm_cycles_elapsed_avg", ["sm__cycles_elapsed.avg"]),
    ("sm_cycles_active_max", ["sm__cycles_active.max"]),
    ("sm_clock_ghz", ["sm__cycles_elapsed.avg.per_second", "gpc__cycles_elapsed.avg.per_second"]),
]
KEEP = re.compile(r"(stall|excessive|sector|request|hit_rate|throughput|occupancy|warps_|issue|inst_executed|registers|cycles_|dram__bytes|"
                  r"bytes_pipe_lsu|local|spill|time_duration|waves|branch|thread_inst)")


def num(s):
    if s is None:
        return None
    t = s.strip().replace(",", "")
    if t in ("", "n/a", "N/A"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def scale(v, unit, want):
    """convert to ms (time) or GB (bytes) or GHz (frequency)"""
    if v is None:
        return None
    u = (unit or "").strip()
    if want == "ms":
        f = {"ns": 1e-6, "nsecond": 1e-6, "usecond": 1e-3, "us": 1e-3, "msecond": 1, "ms": 1, "second": 1e3, "s": 1e3}.get(u)
        return v * f if f else v
    if want == "GB":
        f = {"byte": 1e-9, "Kbyte": 1e-6, "Mbyte": 1e-3, "Gbyte": 1, "KB": 1e-6, "MB": 1e-3, "GB": 1, "B": 1e-9}.get(u)
        return v * f if f else v
    if want == "GHz":
        f = {"hz": 1e-9, "Hz": 1e-9, "Khz": 1e-6, "KHz": 1e-6, "Mhz": 1e-3, "MHz": 1e-3, "Ghz": 1, "GHz": 1, "cycle/second": 1e-9,
             "cycle/nsecond": 1, "cycle/usecond": 1e-3}.get(u)
        return v * f if f else v
    return v


def read_raw(path):
    rows = list(csv.reader(open(path, errors="replace")))
    hi = next((i for i, r in enumerate(rows) if r and r[0] == "ID"), None)
    if hi is None:
        return []
    hdr = rows[hi]
    units = rows[hi + 1] if hi + 1 < len(rows) else []
    data = [r for r in rows[hi + 1:] if r and r[0].strip().isdigit()]
    out = []
    for r in data:
        d = {"_name": r[hdr.index("Kernel Name")] if "Kernel Name" in hdr else "", "_all": {}}
        for i, h in enumerate(hdr):
            if i < len(r):
                d["_all"][h] = (r[i], units[i] if i < len(units) else "")
        out.append(d)
    return out


def kernel_label(name):
    m = re.search(r"(k_p[123]|k_lane)", name)
    return m.group(1) if m else name[:20]


def summarise(k):
    a = k["_all"]
    s = {"kernel": kernel_label(k["_name"])}
    for col, cands in KEYS:
        for c in cands:
            if c in a:
                v, u = a[c]
                v = num(v)
                if col == "duration_ms":
                    v = scale(v, u, "ms")
                elif col.endswith("_GB"):
                    v = scale(v, u, "GB")
                elif col == "sm_clock_ghz":
                    v = scale(v, u, "GHz")
                s[col] = v
                break
    sr = lambda a_, b_: (s.get(a_) / s.get(b_)) if s.get(a_) is not None and s.get(b_) else None
    s["gld_sectors_per_req"] = sr("gld_sectors", "gld_requests")
    s["gst_sectors_per_req"] = sr("gst_sectors", "gst_requests")
    s["lld_sectors_per_req"] = sr("lld_sectors", "lld_requests")
    s["lst_sectors_per_req"] = sr("lst_sectors", "lst_requests")
    s["gld_l1_hit_pct"] = 100 * s["gld_hit_sectors"] / s["gld_sectors"] if s.get("gld_hit_sectors") is not None and s.get("gld_sectors") else None
    s["sm_active_share"] = sr("sm_cycles_active_avg", "sm_cycles_elapsed_avg")
    # sectors per active thread of a request: 1.0 = every active thread in its own sector (fully uncoalesced)
    if s.get("gld_sectors_per_req") and s.get("simt_threads_per_inst"):
        s["gld_sectors_per_active_thread"] = s["gld_sectors_per_req"] / s["simt_threads_per_inst"]
    if s.get("gst_sectors_per_req") and s.get("simt_threads_per_inst"):
        s["gst_sectors_per_active_thread"] = s["gst_sectors_per_req"] / s["simt_threads_per_inst"]
    st = {}
    for name, (v, u) in a.items():
        m = re.match(r"smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$", name)
        if m and num(v) is not None:
            st[m.group(1)] = num(v)
    tot = sum(st.values())
    s["stalls_cyc_per_issue"] = dict(sorted(st.items(), key=lambda x: -x[1]))
    s["stall_total_cyc_per_issue"] = tot
    s["top_stalls"] = [(r, round(v, 3), round(100 * v / tot, 1)) for r, v in sorted(st.items(), key=lambda x: -x[1])[:5]] if tot else []
    ps = {}
    for name, (v, u) in a.items():
        m = re.match(r"smsp__pcsamp_warps_issue_stalled_(\w+)$", name)
        if m and not m.group(1).endswith("not_issued") and num(v) is not None:
            ps[m.group(1)] = num(v)
    pt = sum(ps.values())
    s["pcsamp_top"] = [(r, int(v), round(100 * v / pt, 1)) for r, v in sorted(ps.items(), key=lambda x: -x[1])[:6]] if pt else []
    s["excessive"] = {n: num(v) for n, (v, u) in a.items() if "excessive" in n and num(v) is not None}
    s["kept_metrics"] = {n: [v, u] for n, (v, u) in a.items() if KEEP.search(n)}
    return s


def read_source(path, topn=10):
    """Source view CSV of ncu 2026.2 (--page source --csv --print-source cuda,sass): per file a "File Path" row, a "Function Name"
    row, a header row ("Line No","Source","Address","Source",metrics...) and one row per CUDA line (Address "-") followed by its
    SASS rows. Rows of the same (file, line) from several kernel instances are summed. Returns the top lines by warp-stall samples."""
    if not os.path.exists(path):
        return None
    rows = list(csv.reader(open(path, errors="replace")))
    hdr, cur_file, agg = None, "", {}
    for r in rows:
        if not r:
            continue
        if r[0] == "File Path" or r[0] == "File Name":
            cur_file = r[1].strip() if len(r) > 1 else ""
            continue
        if r[0] == "Line No":
            hdr = r
            continue
        if hdr is None or not r[0].strip().isdigit() or (len(r) > 2 and r[2].strip() not in ("-", "")):
            continue
        key = (cur_file, int(r[0]))
        d = agg.setdefault(key, {"_src": r[1], "_v": {}})
        for i in range(4, min(len(hdr), len(r))):
            v = num(r[i])
            if v is not None:
                d["_v"][i] = d["_v"].get(i, 0) + v
    if not agg:
        return {"error": "no rows parsed", "head": rows[:6]}
    cols = {h: i for i, h in enumerate(hdr) if i >= 4}
    scol = next((h for h in cols if h.startswith("Warp Stall Sampling (All")), None)
    ncol = next((h for h in cols if h.startswith("Warp Stall Sampling (Not")), None)
    si, ni = cols.get(scol), cols.get(ncol)
    tot = sum(d["_v"].get(si, 0) for d in agg.values())
    want = [h for h in cols if any(k in h for k in ("Instructions Executed", "Avg. Threads Executed", "Divergent Branches", "L2 Theoretical Sectors",
                                                      "L1 Wavefronts", "Excessive"))]
    stl = [h for h in cols if h.startswith("stall_") and "(Not Issued)" not in h]
    ie, te = cols.get("Instructions Executed"), cols.get("Thread Instructions Executed")
    lines = []
    for (f, ln), d in sorted(agg.items(), key=lambda kv: -kv[1]["_v"].get(si, 0)):
        v = d["_v"].get(si, 0)
        if v <= 0:
            continue
        st = sorted(((h[6:], d["_v"].get(cols[h], 0)) for h in stl), key=lambda x: -x[1])
        inst, tinst = d["_v"].get(ie), d["_v"].get(te)
        lines.append(dict(file=os.path.basename(f) or "?", line=ln, samples=v, share_pct=round(100 * v / tot, 3) if tot else None,
                          not_issued=d["_v"].get(ni) if ni is not None else None, inst_executed=inst,
                          threads_per_inst=round(tinst / inst, 2) if inst and tinst else None,
                          l2_sectors_global=d["_v"].get(cols.get("L2 Theoretical Sectors Global", -1)),
                          l2_sectors_global_excessive=d["_v"].get(cols.get("L2 Theoretical Sectors Global Excessive", -1)),
                          l2_sectors_local=d["_v"].get(cols.get("L2 Theoretical Sectors Local", -1)),
                          stall1="%s %.0f%%" % (st[0][0], 100 * st[0][1] / v) if st and v else "",
                          stall2="%s %.0f%%" % (st[1][0], 100 * st[1][1] / v) if len(st) > 1 and v else "",
                          source=d["_src"].strip()[:140]))
    top = lines[:topn]
    by_file = {}
    for (f, ln), d in agg.items():
        by_file[os.path.basename(f)] = by_file.get(os.path.basename(f), 0) + d["_v"].get(si, 0)
    return {"stall_column": scol, "total_samples": tot, "columns": hdr, "top": top, "lines": lines,
            "samples_by_file_pct": {k: round(100 * v / tot, 2) for k, v in sorted(by_file.items(), key=lambda x: -x[1])} if tot else {}}


def main():
    allk = []
    for out in sys.argv[1:]:
        res = {"dir": out, "labels": {}}
        for raw in sorted(glob.glob(os.path.join(out, "ncu_*_raw.csv"))):
            L = os.path.basename(raw)[4:-8]
            ks = [summarise(k) for k in read_raw(raw)]
            src = {}
            for k in ("k_p1", "k_p2", "k_p3", "k_lane"):
                s = read_source(os.path.join(out, "ncu_%s_%s_src.csv" % (L, k)))
                if s:
                    src[k] = s
            res["labels"][L] = {"kernels": ks, "source": src}
            for k in ks:
                allk.append((L, k))
        cols = ["duration_ms", "regs", "occ_theor_pct", "occ_achieved_pct", "lim_regs_blocks", "lim_warps_blocks", "warp_cyc_per_inst",
                "issue_slots_busy_pct", "ipc", "simt_threads_per_inst", "l1_hit_pct", "l2_hit_pct", "dram_thru_pct", "dram_read_GB", "dram_write_GB",
                "gld_sectors_per_req", "gld_sectors_per_active_thread", "gst_sectors_per_req", "gst_sectors_per_active_thread", "lld_sectors_per_req",
                "lst_sectors_per_req", "lld_requests", "lst_requests", "gld_requests", "gst_requests", "sm_active_share", "warps_eligible_per_sched",
                "warps_active_per_sched", "sm_busy_pct", "mem_busy_pct", "l2_thru_pct", "sm_clock_ghz", "branch_uniform_pct"]
        with open(os.path.join(out, "e1_ncu_kernels.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "kernel"] + cols + ["top_stalls", "pcsamp_top"])
            for L, k in allk:
                w.writerow([L, k["kernel"]] + [k.get(c) for c in cols] + [k.get("top_stalls"), k.get("pcsamp_top")])
        with open(os.path.join(out, "e1_ncu_lines.csv"), "w", newline="") as f:
            keys = ["file", "line", "samples", "share_pct", "not_issued", "inst_executed", "threads_per_inst", "l2_sectors_global",
                    "l2_sectors_global_excessive", "l2_sectors_local", "stall1", "stall2", "source"]
            w = csv.writer(f)
            w.writerow(["label", "kernel"] + keys)
            for L, d in res["labels"].items():
                for k, s in d["source"].items():
                    for t in s.get("lines", []):
                        w.writerow([L, k] + [t.get(c) for c in keys])
                    s.pop("lines", None)
        json.dump(res, open(os.path.join(out, "e1_ncu_summary.json"), "w"), indent=1, default=str)
        with open(os.path.join(out, "e1_ncu_top_lines.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "kernel", "rank", "file", "line", "stall_samples", "share_pct", "not_issued", "threads_per_inst", "stall1", "stall2", "source"])
            for L, d in res["labels"].items():
                for k, s in d["source"].items():
                    for i, t in enumerate(s.get("top", [])):
                        w.writerow([L, k, i + 1, t["file"], t["line"], t["samples"], t["share_pct"], t["not_issued"], t.get("threads_per_inst"), t.get("stall1"),
                                    t.get("stall2"), t["source"]])
        for L, d in res["labels"].items():
            for k in d["kernels"]:
                print("== %s %s" % (L, k["kernel"]))
                print("   " + "  ".join("%s=%s" % (c, ("%.4g" % k[c]) if isinstance(k.get(c), float) else k.get(c)) for c in cols if k.get(c) is not None))
                print("   stalls (cycles per issued instruction, share):", k.get("top_stalls"))
                print("   pc samples:", k.get("pcsamp_top"))
                if k.get("excessive"):
                    print("   excessive:", k["excessive"])
            for kk, s in d["source"].items():
                print("-- %s %s top lines by %s (total %s); samples by file %s" % (L, kk, s.get("stall_column"), s.get("total_samples"), s.get("samples_by_file_pct")))
                if "error" in s:
                    print("   ", s["error"], s.get("head"))
                for i, t in enumerate(s.get("top", [])):
                    print("   %2d %5.1f%% %s:%d thr/inst %s [%s, %s]  %s" % (i + 1, t["share_pct"] or 0, t["file"], t["line"], t.get("threads_per_inst"),
                                                                    t.get("stall1"), t.get("stall2"), t["source"][:100]))


if __name__ == "__main__":
    main()
