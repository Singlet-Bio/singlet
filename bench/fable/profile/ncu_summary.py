#!/usr/bin/env python3
"""E1 profile: summarise the Nsight Compute exports of one e1_ncu job directory (stdlib only).

Usage: ncu_summary.py OUTDIR [OUTDIR2 ...]   (OUTDIR = $W/jobs/fable/profile/results/ncu-<jobid>)
Inputs per label L (A, M, B, monog, monol): ncu_L_raw.csv (--page raw --csv), ncu_L_<kernel>_src.csv (--page source --csv
--print-source cuda). Writes OUTDIR/e1_ncu_summary.json, OUTDIR/e1_ncu_kernels.csv, OUTDIR/e1_ncu_top_lines.csv and prints tables.
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
    """CUDA-source view CSV: rows of (line number, source, metrics...). Returns the top lines by warp-stall samples."""
    if not os.path.exists(path):
        return None
    rows = list(csv.reader(open(path, errors="replace")))
    hdr = None
    cur_file = ""
    lines = []
    for r in rows:
        if not r:
            continue
        if len(r) == 1 and ("/" in r[0] or r[0].endswith((".cu", ".cuh", ".h"))):
            cur_file = r[0].strip()
            continue
        if r[0] in ("#", "Line", "# Line") or ("Source" in r and any("Stall" in c or "Samples" in c for c in r)):
            hdr = r
            continue
        if hdr and r[0].strip().isdigit():
            d = dict(zip(hdr, r))
            d["_file"] = cur_file
            lines.append(d)
    if not lines:
        return {"error": "no rows parsed", "head": rows[:5]}
    scol = next((c for c in hdr if "Warp Stall Sampling (All" in c), None) or next((c for c in hdr if "Sampl" in c), None)
    ncol = next((c for c in hdr if "Not-issued" in c or "Not Issued" in c), None)
    tot = sum(num(l.get(scol)) or 0 for l in lines) if scol else 0
    lines.sort(key=lambda l: -(num(l.get(scol)) or 0))
    extra = [c for c in hdr if any(k in c for k in ("Excessive", "L2 Theoretical Sectors", "Instructions Executed", "Thread Instructions",
                                                      "L1 Conflicts", "Memory L2", "Global Load", "Local"))]
    top = []
    for l in lines[:topn]:
        v = num(l.get(scol)) or 0
        top.append(dict(file=os.path.basename(l["_file"]) or "?", line=int(l[hdr[0]]), samples=v, share_pct=round(100 * v / tot, 2) if tot else None,
                        not_issued=num(l.get(ncol)) if ncol else None, source=(l.get("Source", "") or "").strip()[:140],
                        **{c: l.get(c) for c in extra}))
    return {"stall_column": scol, "total_samples": tot, "columns": hdr, "top": top}


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
        json.dump(res, open(os.path.join(out, "e1_ncu_summary.json"), "w"), indent=1, default=str)
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
        with open(os.path.join(out, "e1_ncu_top_lines.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "kernel", "rank", "file", "line", "stall_samples", "share_pct", "not_issued", "source"])
            for L, d in res["labels"].items():
                for k, s in d["source"].items():
                    for i, t in enumerate(s.get("top", [])):
                        w.writerow([L, k, i + 1, t["file"], t["line"], t["samples"], t["share_pct"], t["not_issued"], t["source"]])
        for L, d in res["labels"].items():
            for k in d["kernels"]:
                print("== %s %s" % (L, k["kernel"]))
                print("   " + "  ".join("%s=%s" % (c, ("%.4g" % k[c]) if isinstance(k.get(c), float) else k.get(c)) for c in cols if k.get(c) is not None))
                print("   stalls (cycles per issued instruction, share):", k.get("top_stalls"))
                print("   pc samples:", k.get("pcsamp_top"))
                if k.get("excessive"):
                    print("   excessive:", k["excessive"])
            for kk, s in d["source"].items():
                print("-- %s %s top lines by %s (total %s)" % (L, kk, s.get("stall_column"), s.get("total_samples")))
                if "error" in s:
                    print("   ", s["error"], s.get("head"))
                for i, t in enumerate(s.get("top", [])):
                    print("   %2d %5.1f%% %s:%d  %s" % (i + 1, t["share_pct"] or 0, t["file"], t["line"], t["source"][:110]))


if __name__ == "__main__":
    main()
