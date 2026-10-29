#!/usr/bin/env python3
"""Parse one `singlet --pipeline` run directory (run.log, time.txt, monitor.tsv, node.txt, mem.txt) into stage seconds.

Stage definitions (sorted path):
  star_map   STAR 'started mapping' -> 'started Solo counting' (includes writing the unsorted BAM bins)
  star_solo  'started Solo counting' -> 'started sorting BAM'
  sort       'started sorting BAM' -> end of STAR (STAR completed - sum of the above)
  index, cb_refresh, pileup_pass ('[parallel] Processing'), pileup_total ('[singlet] Pileup:'), strand_rerun (if logged)
Usage: parse_run.py RUN_DIR > stages.json
"""
import json, re, sys, os, datetime

def secs(ts):  # 'Oct 26 06:54:41'
    return datetime.datetime.strptime("2026 " + ts, "%Y %b %d %H:%M:%S").timestamp()

def main(d):
    log = open(os.path.join(d, "run.log"), errors="replace").read()
    o = {"dir": d}
    star = {}
    for m in re.finditer(r"^(\w{3} +\d+ \d\d:\d\d:\d\d) \.\.\.\.\. (started [\w ]+|finished successfully)", log, re.M):
        star[m.group(2)] = secs(re.sub(r" +", " ", m.group(1)))
    m = re.search(r"STAR completed in ([\d.]+)s", log); o["star_total"] = float(m.group(1)) if m else None
    t = lambda k: star.get(k)
    if t("started mapping") and t("started Solo counting"): o["star_map"] = t("started Solo counting") - t("started mapping")
    if t("started Solo counting") and t("started sorting BAM"): o["star_solo"] = t("started sorting BAM") - t("started Solo counting")
    if o.get("star_total") and t("started STAR run") and t("started sorting BAM"):
        o["sort"] = o["star_total"] - (t("started sorting BAM") - t("started STAR run"))
    def f(pat, key, cast=float):
        m = re.search(pat, log); 
        if m: o[key] = cast(m.group(1))
    f(r"BAM indexed in ([\d.]+)s", "index")
    f(r"CB-tag barcode refresh: .* in ([\d.]+)s", "cb_refresh")
    f(r"\[parallel\] Processing: ([\d.]+)s", "pileup_pass")
    f(r"\[singlet\] Pileup: .*?, ([\d.]+)s", "pileup_total")
    f(r"\[1fq-pass1\] Scanned \d+ blocks \(([\d.]+)s\)", "fq_pass1")
    f(r"\[singlet\] References loaded: ([\d.]+)s", "ref_load")
    f(r"\[export\] CSC conversion: ([\d.]+)s", "export_csc")
    f(r"in ([\d.]+)s\n.*?\[export\]", "x", float) if False else None
    o["strand_rerun_logged"] = bool(re.search(r"strand.*(re-?run|retry|flip)", log, re.I))
    m = re.search(r"Pileup: ([\d]+) reads", log) or re.search(r"Done: (\d+) reads", log)
    if m: o["reads"] = int(m.group(1))
    tt = open(os.path.join(d, "time.txt"), errors="replace").read() if os.path.exists(os.path.join(d, "time.txt")) else ""
    m = re.search(r"Elapsed \(wall clock\) time.*: (?:(\d+):)?(\d+):([\d.]+)", tt)
    if m: o["wall"] = int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    for k, pat in (("user_s", r"User time \(seconds\): ([\d.]+)"), ("sys_s", r"System time \(seconds\): ([\d.]+)"),
                   ("maxrss_kb", r"Maximum resident set size \(kbytes\): (\d+)")):
        m = re.search(pat, tt)
        if m: o[k] = float(m.group(1))
    if "user_s" in o: o["cpu_s"] = o["user_s"] + o.get("sys_s", 0)
    mem = os.path.join(d, "mem.txt")
    if os.path.exists(mem):
        o.update({k: v for k, v in (l.strip().split("=", 1) for l in open(mem) if "=" in l)})
    mon = os.path.join(d, "monitor.tsv")
    if os.path.exists(mon):
        rows = [l.split("\t") for l in open(mon).read().splitlines()[1:] if l.count("\t") >= 5]
        if rows:
            o["cgroup_peak_mem_mb"] = max(int(r[1]) for r in rows)
            o["cgroup_read_mb"] = int(rows[-1][4]); o["cgroup_write_mb"] = int(rows[-1][5])
    nd = os.path.join(d, "node.txt")
    if os.path.exists(nd):
        s = open(nd).read(); m = re.search(r"node=(\S+)", s); o["node"] = m.group(1) if m else None
        m = re.search(r"cotenant_cpus=(\d+)", s); o["cotenant_cpus"] = int(m.group(1)) if m else None
        m = re.search(r"loadavg=(.*)", s); o["loadavg"] = m.group(1).strip() if m else None
    print(json.dumps(o, indent=1))

main(sys.argv[1])
