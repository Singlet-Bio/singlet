"""M10: host work per sample (W) from existing CPU run logs (docs/gpu-engine/exploration.md, section 9, M10).

    python m10_hostwork.py OUT.json RUN_DIR [RUN_DIR ...]      # each RUN_DIR holds run.log (+ time.txt, stages.json, monitor.tsv)

Read-only. For every run it extracts the stage timers the CPU engine prints (wall seconds), the thread count, the read
count, and -- when monitor.tsv (cumulative cgroup CPU seconds every ~5 s) exists -- the measured core-seconds of the
post-mapping windows, anchored on STAR's "finished successfully" timestamp and the end of the monitor:

    STAR end | index | CB refresh | pileup | non-host (residual) | export (callers + writers) | end

What stays on the CPU in the GPU design (exploration.md section 6): called_ed (here called_v2), ambient, doublets, demux,
sex + ancestry, non-host EM, and the export (everything the engine prints between "Exporting feature matrices" and
"[export] Total"). The GPU removes the BAM sort, index, CB refresh and pileup. Callers without timers in the logs are listed
under "no_timer" and are covered only inside the export window.
"""

import calendar
import glob
import json
import os
import re
import sys

NUM = r"([0-9][0-9.]*(?:e[+-]?[0-9]+)?)"
TAG = re.compile(r"^\[([A-Za-z0-9_\-]+)\]")
TIME = re.compile(r"time=" + NUM + r"s")
MONTH = {m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}
DATE = re.compile(r"^([A-Z][a-z]{2}) +(\d+) (\d\d):(\d\d):(\d\d) \.\.\.\.\. (.*)$")
EDT = 4 * 3600        # all runs are Oct 2026, before the DST change: local = UTC - 4 h


def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def first(rx, text, cast=float):
    m = re.search(rx, text, re.M)
    return cast(m.group(1)) if m else None


def monitor(path):
    pts = []
    for ln in read(path).splitlines():
        p = ln.split("\t")
        if len(p) >= 4 and p[0].isdigit() and p[3].replace(".", "").isdigit():
            pts.append((int(p[0]), float(p[3])))
    return pts


def cpu_at(pts, t):
    if not pts:
        return None
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    lo, hi = 0, len(pts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pts[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (t0, c0), (t1, c1) = pts[lo], pts[hi]
    return c0 + (c1 - c0) * (t - t0) / max(t1 - t0, 1)


def analyse(d):
    log = read(os.path.join(d, "run.log"))
    if not log:
        return None
    tm = read(os.path.join(d, "time.txt"))
    r = {"dir": d, "reads": first(r"\.1fq: (\d+) reads", log, int)}
    if r["reads"] is None:
        r["reads"] = first(r"Pileup: (\d+) reads", log, int)
    r["threads"] = first(r"STAR pid=\d+, (\d+) threads", log, int) or first(r"--threads (\d+)", tm, int)
    r["binary"] = (re.search(r"(singlet-[\w\-.]+)", tm) or [None, None])[1]
    r["user_s"] = first(r"User time \(seconds\): " + NUM, tm)
    r["sys_s"] = first(r"System time \(seconds\): " + NUM, tm)
    r["total_wall"] = first(r"\[singlet\] Total: " + NUM + "s", log)
    st = {
        "pass1": first(r"\[1fq-pass1\] Scanned .*\(" + NUM + "s\\)", log),
        "ref_load": first(r"\[singlet\] References loaded: " + NUM + "s", log),
        "star": first(r"STAR completed in " + NUM + "s", log),
        "index": first(r"BAM indexed in " + NUM + "s", log),
        "cb_refresh": first(r"CB-tag barcode refresh: .* in " + NUM + "s", log),
        "pileup": first(r"\[singlet\] Pileup: .*, " + NUM + "s", log),
        "export_csc": first(r"\[export\] CSC conversion: " + NUM + "s", log),
        "export_total": first(r"\[export\] Total: " + NUM + "s", log),
    }
    r["stages"] = st
    # per-tag timers printed as time=Xs
    tag_t, no_timer = {}, set()
    for ln in log.splitlines():
        m = TAG.match(ln)
        if not m:
            continue
        t = TIME.search(ln)
        if t:
            tag_t[m.group(1)] = tag_t.get(m.group(1), 0.0) + float(t.group(1))
        else:
            no_timer.add(m.group(1))
    # demux: sum of "(X s)" auto-K fits, and the "selected K in X s" line
    ak = [float(x) for x in re.findall(r"\[demux\] auto-K K=\d+: .*\(" + NUM + r" s\)", log)]
    sel = first(r"\[demux\] auto-K: selected K=\d+ in " + NUM + r" s", log)
    r["demux"] = {"auto_k_fits_s": sum(ak) if ak else None, "auto_k_selected_s": sel,
                  "other_s_matches": re.findall(r"\[demux\][^\n]*? ([0-9.]+) ?s\b", log)[:12]}
    r["timers"] = tag_t
    r["tags_without_timer"] = sorted(no_timer - set(tag_t))
    r["called_ed_present"] = "called_ed" in log
    r["cell_caller"] = first(r"cell_caller=(\w+)", log, str)
    nh = [ln for ln in log.splitlines() if ln.startswith("[nonhost]")]
    r["nonhost_lines"] = len(nh)
    r["nonhost_timer_lines"] = [ln[:160] for ln in nh if re.search(r"[0-9.]+ ?s\b", ln)][:3]
    r["nonhost_unmapped_reads"] = first(r"EM screening (\d+) unmapped reads", log, int)

    # timestamps -> epoch; measured core-seconds in windows
    mon = monitor(os.path.join(d, "monitor.tsv"))
    fin = None
    for ln in log.splitlines():
        m = DATE.match(ln)
        if m and m.group(6).startswith("finished successfully"):
            fin = calendar.timegm((2026, MONTH[m.group(1)], int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), 0, 0, 0)) + EDT
    if mon and fin and st["export_total"] is not None:
        t_end = mon[-1][0]
        a = fin
        idx, cbr, pil = st["index"] or 0, st["cb_refresh"] or 0, st["pileup"] or 0
        b, c, e = a + idx, a + idx + cbr, a + idx + cbr + pil
        PAD = 10.0     # seconds after "[export] Total" (rrna, provenance, validation); counted as export
        xs = t_end - st["export_total"] - PAD
        w = {"index": (a, b), "cb_refresh": (b, c), "pileup": (c, e), "nonhost_residual": (e, max(e, xs)), "export": (xs, t_end)}
        r["core_s"] = {k: round(cpu_at(mon, y) - cpu_at(mon, x), 1) for k, (x, y) in w.items()}
        r["window_wall_s"] = {k: round(y - x, 1) for k, (x, y) in w.items()}
        r["core_s"]["post_star_total"] = round(cpu_at(mon, t_end) - cpu_at(mon, fin), 1)
        r["core_s"]["run_total"] = round(mon[-1][1] - mon[0][1], 1)
    return r


def main():
    out = sys.argv[1]
    dirs = []
    for a in sys.argv[2:]:
        dirs.extend(sorted(glob.glob(a)))
    runs = []
    for d in dirs:
        r = analyse(d)
        if r:
            runs.append(r)
    with open(out, "w") as f:
        json.dump(runs, f, indent=1)
    print(f"{len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
