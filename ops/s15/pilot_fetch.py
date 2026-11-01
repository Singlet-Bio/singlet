#!/usr/bin/env python3
"""S15: fetch every run of one GSM (S9 fetch_run, full .sra, shared cap) and decode each to .1fq with the pinned binary. Keeps .sra + .1fq under P.
    pilot_fetch.py GSM P SRC BIN WLDIR CAPDIR"""
import csv, json, os, subprocess, sys, urllib.parse, urllib.request
gsm, P, SRC, BIN, WL, CAP = sys.argv[1:7]
sys.path.insert(0, SRC)
import fetch_worker as F, ratecap
q = urllib.parse.quote('sample_alias="%s"' % gsm)
u = "https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&format=tsv&fields=run_accession,read_count,library_layout&query=" + q
rows = list(csv.DictReader(urllib.request.urlopen(u, timeout=120).read().decode().splitlines(), delimiter="\t"))
runs = sorted((r["run_accession"], int(r["read_count"] or 0)) for r in rows)
assert runs, "no ENA runs for " + gsm
os.makedirs(P + "/sra", exist_ok=True); os.makedirs(P + "/stage", exist_ok=True)
json.dump({"gsm": gsm, "runs": runs, "pairs": sum(n for _, n in runs)}, open(P + "/runs.json", "w"))
env = dict(os.environ, SINGLET_WHITELIST_DIR=WL)
lim = ratecap.SharedRate(CAP, "s15-%s" % gsm)
for acc, n in runs:
    fq = "%s/stage/%s.1fq" % (P, acc)
    if os.path.exists(fq + ".done"):
        continue
    have = "%s/sra/%s.sra" % (P, acc)
    if os.path.exists(have) and os.environ.get("REUSE_SRA"):
        sra = have
    else:
      lim.set_active(True)
      sra, info = F.fetch_run(acc, P + "/sra", 8, lambda m: print(m, flush=True), lim)
      lim.set_active(False)
    if os.environ.get("NODECODE"):
        open(fq + ".done", "w").close(); print("fetched only", acc, n, flush=True); continue
    rc = subprocess.run([BIN, "download", sra, "--quality", "binned", "--vdb-threads", "2", "--no-profile", "--whitelist-dir", WL, "-o", fq], env=env).returncode
    if rc != 0:
        print("decode rc", rc, acc)
        sys.exit(20)
    open(fq + ".done", "w").close()
    print("ok", acc, n, flush=True)
if not os.environ.get("NODECODE"):
    open(P + "/stage/.wl_ok", "w").close()   # decoded with --whitelist-dir (see ensure_wl in pilot_stage.sh)
