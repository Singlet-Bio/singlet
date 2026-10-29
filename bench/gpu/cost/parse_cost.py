"""Parse one run dir (run.log, monitor.tsv, time.txt, node.txt, wall.txt) of run_single.sbatch / run_pack.sbatch into a dict.
Stage core-seconds come from the cgroup CPU counter (monitor.tsv, every 2 s) between the times at which marker lines appeared
in run.log (the monitor also logs run.log's size, so a marker's time is the first sample whose size covers its byte offset).
Only valid when the monitor's cgroup holds this run alone (run_single.sbatch).  Whole-run numbers (wall, user+sys, RSS) come
from /usr/bin/time and are valid for packed runs too."""
import json, re, sys, os

MARK = {  # name -> regex on run.log (first match)
 "star_start": rb"\.\.\.\.\. started mapping",
 "sort_start": rb"\.\.\.\.\. started sorting BAM",
 "star_done": rb"\[singlet\] STAR completed in",
 "nh_start": rb"\[nonhost\] Loading screening indices",
 "nh_loaded": rb"\[nonhost\] Loading host Bloom filter",
 "export_start": rb"\[singlet\] Exporting feature matrices",
 "end": rb"\[singlet\] Total:",
}

def parse(d):
    o = {"dir": d}
    log = open(os.path.join(d, "run.log"), "rb").read()
    txt = log.decode(errors="replace")
    f = lambda p, c=float: (lambda m: c(m.group(1)) if m else None)(re.search(p, txt))
    o["pairs"] = f(r"\.1fq: (\d+) reads", int)
    o["bam_records"] = f(r"\[singlet\] Pileup: (\d+) reads", int)
    o["exit_ok"] = bool(re.search(r"\[singlet\] Total:", txt)) and "\nexit " not in txt
    eng = {"refs_s": r"References loaded: ([\d.]+)s", "fq_pass1_s": r"Scanned \d+ blocks \(([\d.]+)s\)",
           "star_s": r"STAR completed in ([\d.]+)s", "bam_index_s": r"BAM indexed in ([\d.]+)s",
           "cb_refresh_s": r"CB-tag barcode refresh: .* in ([\d.]+)s", "pileup_s": r"\[singlet\] Pileup: \d+ reads, ([\d.]+)s",
           "export_s": r"\[export\] Total: ([\d.]+)s", "total_s": r"\[singlet\] Total: ([\d.]+)s",
           "genome_prefetch_s": r"Genome prefetch: \d+ MB in ([\d.]+)s"}
    for k, p in eng.items():
        o[k] = f(p)
    tt = open(os.path.join(d, "time.txt")).read() if os.path.exists(os.path.join(d, "time.txt")) else ""
    m = re.search(r"Elapsed \(wall clock\) time.*: (?:(\d+):)?(\d+):([\d.]+)", tt)
    if m: o["wall_s"] = int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    for k, p in (("user_s", r"User time \(seconds\): ([\d.]+)"), ("sys_s", r"System time \(seconds\): ([\d.]+)"),
                 ("max_rss_gb", r"Maximum resident set size \(kbytes\): (\d+)")):
        m = re.search(p, tt)
        if m: o[k] = float(m.group(1)) / (1048576 if k == "max_rss_gb" else 1)
    if "user_s" in o: o["core_s"] = round(o["user_s"] + o["sys_s"], 1)
    nd = open(os.path.join(d, "node.txt")).read() if os.path.exists(os.path.join(d, "node.txt")) else ""
    for k, p in (("node", r"node=(\S+)"), ("cpu_model", r"Model name:\s*(.+)"), ("cpus_on_node", r"^CPU\(s\):\s*(\d+)"),
                 ("cotenant_cpus", r"cotenant_cpus=(\d+)"), ("loadavg", r"loadavg=(.+)")):
        m = re.search(p, nd, re.M)
        if m: o[k] = m.group(1).strip()
    mp = os.path.join(d, "monitor.tsv")
    if os.path.exists(mp):
        rows = [l.split("\t") for l in open(mp).read().splitlines()[1:] if l.count("\t") >= 3]
        rows = [(float(r[0]), float(r[1]), float(r[3]), int(r[4]) if len(r) > 4 and r[4].isdigit() else None) for r in rows if r[3]]
        if rows:
            o["cgroup_peak_mem_gb"] = round(max(r[1] for r in rows) / 1024, 1)
            o["cgroup_core_s"] = round(rows[-1][2] - rows[0][2], 1)
            if rows[0][3] is not None and rows[-1][3]:
                def tm(pat):
                    m = re.search(pat, log)
                    if not m: return None
                    for r in rows:
                        if r[3] >= m.end(): return r
                T = {k: tm(p) for k, p in MARK.items()}
                if not T.get("end"): T["end"] = rows[-1]
                T["t0"] = rows[0]
                def win(a, b):
                    if T.get(a) and T.get(b): return round(T[b][0] - T[a][0], 1), round(T[b][2] - T[a][2], 1)
                st = {}
                for name, a, b in (("pre_sort_incl_fq_refs_star_map", "t0", "sort_start"), ("sort", "sort_start", "star_done"),
                                   ("index_cbrefresh_pileup", "star_done", "nh_start"), ("nonhost_load_3_indices", "nh_start", "nh_loaded"),
                                   ("nonhost_classify_em_write", "nh_loaded", "export_start"), ("export_incl_demux", "export_start", "end")):
                    w = win(a, b)
                    if w: st[name] = {"wall_s": w[0], "core_s": w[1]}
                o["stage_windows"] = st
                nl = st.get("nonhost_load_3_indices")
                if nl: o["nonhost_load_core_s_monitor"] = nl["core_s"]; o["nonhost_load_wall_s_monitor"] = nl["wall_s"]
    return o

if __name__ == "__main__":
    print(json.dumps(parse(sys.argv[1]), indent=1))
