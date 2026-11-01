import csv, sys, urllib.request, collections
csv.field_size_limit(10**9)
O = sys.argv[1]
first = {}
for r in csv.DictReader(open(O + "/runs.tsv"), delimiter="\t"):
    if r["tag"] in ("", "nan") or r["conf"] in ("", "0.0", "nan") or True:
        first.setdefault(r["gsm"], r["run"])
nc = [r["gsm"] for r in csv.DictReader(open(O + "/nocall.tsv"), delimiter="	")]
for g in nc:
    run = first[g]
    try:
        u = "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=%s&result=read_run&fields=submitted_format,submitted_bytes,fastq_bytes,library_layout,instrument_model&format=tsv" % run
        t = urllib.request.urlopen(u, timeout=60).read().decode().strip().splitlines()
        print(g, run, t[-1].replace("\t", " | ") if len(t) > 1 else "EMPTY " + str(t), flush=True)
    except Exception as e:
        print(g, run, "ERR", str(e)[:60], flush=True)
