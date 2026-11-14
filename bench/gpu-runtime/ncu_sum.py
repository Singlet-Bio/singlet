import csv, sys, os, re, glob, collections
M = ["gpu__time_duration.sum", "launch__grid_size", "launch__block_size", "launch__registers_per_thread", "launch__shared_mem_per_block_static", "launch__shared_mem_per_block_dynamic",
     "launch__waves_per_multiprocessor", "sm__maximum_warps_per_active_cycle_pct", "sm__warps_active.avg.pct_of_peak_sustained_active",
     "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps", "launch__occupancy_limit_blocks",
     "sm__throughput.avg.pct_of_peak_sustained_elapsed", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
     "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct", "smsp__issue_active.avg.pct_of_peak_sustained_active", "smsp__thread_inst_executed_per_inst_executed.ratio",
     "memory_l2_theoretical_sectors_global", "memory_l2_theoretical_sectors_global_ideal", "derived__memory_l2_theoretical_sectors_global_excessive"]
ST = ["barrier", "branch_resolving", "dispatch_stall", "drain", "imc_miss", "lg_throttle", "long_scoreboard", "math_pipe_throttle", "membar", "mio_throttle", "misc", "no_instruction", "not_selected", "selected", "short_scoreboard", "sleeping", "tex_throttle", "wait"]
def num(s):
    try: return float(s.replace(",", ""))
    except Exception: return None
for d in sorted(glob.glob(sys.argv[1] + "/*/")):
    p = d + "raw.csv"
    if not os.path.exists(p): continue
    try: rows = list(csv.reader(open(p, newline="")))
    except Exception: continue
    if len(rows) < 3: print("==", os.path.basename(d.rstrip("/")), "no kernel profiled"); continue
    H = rows[0]; U = rows[1]; ix = {h: i for i, h in enumerate(H)}
    data = [r for r in rows[2:] if len(r) == len(H)]
    by = collections.OrderedDict()
    for r in data: by.setdefault(r[ix["Kernel Name"]], []).append(r)
    for kn, rs in by.items():
        print("== %s  %s  (n=%d launches)" % (os.path.basename(d.rstrip("/")), kn[:60], len(rs)))
        def g(m):
            if m not in ix: return None
            v = [num(r[ix[m]]) for r in rs]; v = [x for x in v if x is not None]
            return sum(v) / len(v) if v else None
        t = g("gpu__time_duration.sum"); u = U[ix["gpu__time_duration.sum"]] if "gpu__time_duration.sum" in ix else ""
        print("   duration %s %s; grid %s block %s regs %s smem static %s dyn %s waves %s" % (t, u, g("launch__grid_size"), g("launch__block_size"), g("launch__registers_per_thread"), g("launch__shared_mem_per_block_static"), g("launch__shared_mem_per_block_dynamic"), g("launch__waves_per_multiprocessor")))
        print("   occupancy: theoretical %s %%, achieved %s %%; block limits (blocks/SM): regs %s smem %s warps %s blocks %s" % tuple(["%.1f" % x if x is not None else "-" for x in (g("sm__maximum_warps_per_active_cycle_pct"), g("sm__warps_active.avg.pct_of_peak_sustained_active"))] + [g(m) for m in ("launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps", "launch__occupancy_limit_blocks")]))
        print("   SM %% %s  mem(compute-memory) %% %s  DRAM %% %s  issue active %% %s  threads/warp-inst %s" % tuple("%.1f" % x if x is not None else "-" for x in (g("sm__throughput.avg.pct_of_peak_sustained_elapsed"), g("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"), g("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"), g("smsp__issue_active.avg.pct_of_peak_sustained_active"), g("smsp__thread_inst_executed_per_inst_executed.ratio"))))
        print("   L1 hit %s %%  L2 hit %s %%  global sectors %s ideal %s excessive %s" % tuple("%.4g" % x if x is not None else "-" for x in (g("l1tex__t_sector_hit_rate.pct"), g("lts__t_sector_hit_rate.pct"), g("memory_l2_theoretical_sectors_global"), g("memory_l2_theoretical_sectors_global_ideal"), g("derived__memory_l2_theoretical_sectors_global_excessive"))))
        st = [(g("smsp__average_warps_issue_stalled_%s_per_issue_active.ratio" % s), s) for s in ST]; st = sorted([x for x in st if x[0] is not None], reverse=True)[:4]
        print("   top stalls (warps per issue): " + ", ".join("%s %.2f" % (s, v) for v, s in st))
    dt = d + "details.txt"
    if os.path.exists(dt):
        for l in open(dt, errors="replace"):
            if re.search(r"uncoalesced|Uncoalesced|excessive|local memory|spill|occupancy.*limit|tail effect|imbalance", l) and "OPT" in l or "est. speedup" in l.lower() and False: print("   rule:", l.strip()[:230])
