#!/usr/bin/env python3
"""S15: list ENA runs (sample alias, experiment title, library strategy, reads) for studies. ena_titles.py GSE..."""
import csv, sys, urllib.parse, urllib.request
F = "run_accession,sample_alias,experiment_title,library_strategy,library_source,read_count,library_name"
for gse in sys.argv[1:]:
    u = "https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&format=tsv&limit=0&fields=%s&query=%s" % (F, urllib.parse.quote('study_alias="%s"' % gse))
    rows = list(csv.DictReader(urllib.request.urlopen(u, timeout=300).read().decode().splitlines(), delimiter="\t"))
    print("##", gse, len(rows), "runs")
    for r in sorted(rows, key=lambda r: r["sample_alias"]):
        print("\t".join([r["run_accession"], r["sample_alias"], r["experiment_title"][:110], r["library_strategy"], r["read_count"], r["library_name"][:40]]))
