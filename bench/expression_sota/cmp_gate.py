# SPDX-License-Identifier: MIT
"""Compare a run of the is_cell branch with S5's run of the pin: every file byte-identical except cell_calls.tsv (and provenance/logs);
for cell_calls.tsv show exactly what changed. python cmp_gate.py NEW_DIR PIN_DIR"""
import hashlib, os, sys, csv
new, pin = sys.argv[1:3]
SKIP = {"run.log", "time.txt", "node.txt", "provenance.json", "binary_commit.txt", "binary_sha256.txt", "command.txt", "stream_path.txt", "stream_ok.txt", "summary.json", "pileup_stats.json", "metrics_summary.csv", "encode.log", "encode.time.txt", "encode.time.txt"}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
same = diff = 0; dl = []; missing = []
for root, dirs, files in os.walk(pin):
    dirs[:] = [d for d in dirs if not d.startswith("star__") and d != "possorted_tmp"]
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), pin)
        if f in SKIP or f.endswith(".bam") or "_STARtmp" in rel: continue
        q = os.path.join(new, rel)
        if not os.path.exists(q): missing.append(rel); continue
        if sha(os.path.join(pin, rel)) == sha(q): same += 1
        else: diff += 1; dl.append(rel)
print(f"identical {same}; different {diff}: {dl}; missing in new: {missing[:10]}")
a = list(csv.reader(open(os.path.join(pin, "cell_calls.tsv")), delimiter="\t")); b = list(csv.reader(open(os.path.join(new, "cell_calls.tsv")), delimiter="\t"))
print("header pin:", a[0]); print("header new:", b[0]); assert len(a) == len(b)
ia, ib = {h: i for i, h in enumerate(a[0])}, {h: i for i, h in enumerate(b[0])}
n_is = sum(r[ia["is_cell"]] == "TRUE" for r in a[1:]); m_is = sum(r[ib["is_cell"]] == "TRUE" for r in b[1:])
n_cv2 = sum(r[ia["called_v2"]] == "1" for r in a[1:]); m_cv2 = sum(r[ib["called_v2"]] == "1" for r in b[1:])
kn = sum(r[ib["called_knee"]] == "1" for r in b[1:])
other = sum(1 for x, y in zip(a[1:], b[1:]) for k in ("barcode", "called_v2", "umi_exon", "umi_total", "fdr", "pvalue", "log_prob", "call_method") if x[ia[k]] != y[ib[k]])
print(f"pin: is_cell {n_is} called_v2 {n_cv2}; new: is_cell {m_is} called_knee {kn} called_v2 {m_cv2}; other-column diffs {other}; new is_cell==called_v2 on all rows: {all((r[ib['is_cell']]=='TRUE')==(r[ib['called_v2']]=='1') for r in b[1:])}; new called_knee == pin called/is_cell: {all(x[ia['called']]==y[ib['called_knee']] for x,y in zip(a[1:],b[1:]))}")
