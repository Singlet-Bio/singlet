"""Stage wall times of one singlet --pipeline run, as JSON.  python stage_times.py <run.log> <star_Log.out> <time.txt>

Seconds per stage from the engine's own stage lines (pass 1, reference load, STAR, BAM index, CB-tag refresh, pileup,
export, total), STAR's phase timestamps (genome load, mapping, Solo counting, BAM sort), and /usr/bin/time's wall
clock and peak RSS. Fields that a run did not log are left out.
"""

import json
import re
import sys
from datetime import datetime

run = open(sys.argv[1], errors="replace").read()
star = open(sys.argv[2], errors="replace").read() if len(sys.argv) > 2 else ""
tv = open(sys.argv[3], errors="replace").read() if len(sys.argv) > 3 else ""
out = {}

ENGINE = {
    "pass1_s": r"\[1fq-pass1\] Scanned \d+ blocks \(([\d.]+)s\)",
    "refs_s": r"\[singlet\] References loaded: ([\d.]+)s",
    "genome_prefetch_s": r"\[singlet\] Genome prefetch: \d+ MB in ([\d.]+)s",
    "star_s": r"\[singlet\] STAR completed in ([\d.]+)s",
    "bam_index_s": r"\[singlet\] BAM indexed in ([\d.]+)s",
    "cb_refresh_s": r"CB-tag barcode refresh: .* in ([\d.]+)s",
    "pileup_s": r"\[singlet\] Pileup: \d+ reads, ([\d.]+)s",
    "export_s": r"\[export\] Total: ([\d.]+)s",
    "total_s": r"\[singlet\] Total: ([\d.]+)s",
}
for k, pat in ENGINE.items():
    m = re.search(pat, run)
    if m:
        out[k] = float(m.group(1))

# STAR phases: "Oct 26 06:53:32 ..... started STAR run" (the engine copies STAR's stdout into run.log)
ts = {}
for m in re.finditer(r"^(\w{3} +\d+ \d\d:\d\d:\d\d) \.\.\.\.\. (.+?)\s*$", run, re.M):
    ts.setdefault(m.group(2), datetime.strptime("2000 " + m.group(1), "%Y %b %d %H:%M:%S"))


def span(a, b):
    if a in ts and b in ts:
        return (ts[b] - ts[a]).total_seconds()


for k, (a, b) in {"star_genome_load_s": ("loading genome", "started mapping"),
                  "star_mapping_s": ("started mapping", "finished mapping"),
                  "star_solo_s": ("started Solo counting", "finished Solo counting"),
                  "star_sort_s": ("started sorting BAM", "finished successfully")}.items():
    v = span(a, b)
    if v is not None:
        out[k] = v

for k, pat in {"fastassign": r"FASTASSIGN: (\w+)", "seqcache": r"(SEQCACHE: on, \d+ MB)",
               "seqcache_hit_rate": r"SEQCACHE: lookups=\d+ hits=\d+ hit_rate=([\d.]+)"}.items():
    m = re.search(pat, star)
    if m:
        out[k] = m.group(1)

m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", tv)
if m:
    p = [float(x) for x in m.group(1).split(":")]
    out["wall_s"] = p[-1] + 60 * p[-2] + (3600 * p[-3] if len(p) > 2 else 0)
m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", tv)
if m:
    out["max_rss_gb"] = round(int(m.group(1)) / 1048576, 2)
print(json.dumps(out, indent=1))
