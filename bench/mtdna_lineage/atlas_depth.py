"""Add the full sequencing depth available in SRA/ENA to atlas_survey.csv rows (all runs of each GSM).

  python atlas_depth.py atlas_survey.csv OUT.csv
"""
import csv
import sys
import time
import urllib.parse
import urllib.request

API = "https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&fields=run_accession,read_count,base_count&format=tsv&query="


def runs(gsm):
    q = urllib.parse.quote(f'sample_alias="{gsm}"')
    for t in range(4):
        try:
            txt = urllib.request.urlopen(API + q, timeout=60).read().decode()
            rows = [l.split("\t") for l in txt.strip().split("\n")[1:] if l]
            return [(r[0], int(r[1] or 0), int(r[2] or 0)) for r in rows]
        except Exception:
            time.sleep(2 ** t)
    return None


rows = list(csv.DictReader(open(sys.argv[1])))
cols = list(rows[0].keys()) + ["sra_runs", "sra_reads", "sra_bases"]
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), fieldnames=cols)
w.writeheader()
for i, r in enumerate(rows):
    rr = runs(r["gsm"])
    r["sra_runs"] = len(rr) if rr is not None else ""
    r["sra_reads"] = sum(x[1] for x in rr) if rr else ""
    r["sra_bases"] = sum(x[2] for x in rr) if rr else ""
    w.writerow(r)
    if i % 100 == 0:
        print(i, file=sys.stderr, flush=True)
