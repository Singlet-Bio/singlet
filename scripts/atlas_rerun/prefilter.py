#!/usr/bin/env python3
"""Pre-filter runs from metadata, before a byte of `.sra` is fetched (S9). Standard library only.

    prefilter.py annotate RUNS.txt OUT.jsonl      run accessions (one per line) -> ENA + NCBI SRA metadata per run
    prefilter.py eval ANNOTATED.jsonl LABELS.json  accept/reject vs the observed outcome (parked or staged) of real fetches

Why: SRA files are columnar, so the first reads of a run cannot be had with a byte range; a non-10x or non-GEX run is only discovered after the
whole file is downloaded and decoded (the old flow screened 1 M reads by VDB streaming). 21% of the bytes of a random "single-cell" sample were wasted
that way. The signals that are free:

  * ENA read_run: library_strategy (RNA-Seq), library_source (TRANSCRIPTOMIC SINGLE CELL), instrument_platform (ILLUMINA), library_layout,
    and the free text (experiment/study title, library name) for CITE-seq/HTO/ATAC/TCR/BCR/etc. keywords;
  * NCBI SRA XML (efetch db=sra), the part ENA does not give: the per-READ statistics of the run, `<Statistics nreads=N><Read index=i average=L stdev=s/>`.
    A 10x gene-expression run is 2-4 reads per spot, one of which is a fixed-length barcode+UMI read of 26 (v1/v2, 5' v1/v2), 28 (v3, v3.1, 5' v2, HT, multiome GEX)
    or 16/24/... nt (stdev 0), and one cDNA read of >= 40 nt. Reads of length 8/10 are index reads; they do not matter.

`accept(meta)` returns (True, "") or (False, reason). The rules are deliberately conservative (a false reject loses a sample, a false accept only wastes
the bytes the old flow wasted), and validated against real outcomes in docs/research-beta/download-and-staging.md §3.
"""

import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ENA_FIELDS = ("run_accession,experiment_accession,study_accession,library_strategy,library_source,library_selection,library_layout,"
              "instrument_platform,instrument_model,read_count,base_count,experiment_title,study_title,library_name")
BARCODE_LENS = {16, 24, 26, 28}  # 10x barcode(16)+UMI(10|12) reads: v1/v2 26, v3/v3.1/HT/5'v2/multiome 28; 24 and 16 are older/short UMI variants
NON_GEX = re.compile(r"\b(atac|chromatin|cite-?seq|adt|hto|hashtag|antibody|feature[- ]barcod|crispr|perturb|vdj|v\(d\)j|tcr|bcr|"
                     r"immune[- ]profil|spatial|visium|methyl|cut&?(tag|run)|chip-?seq|bisulfite|hi-?c|multiome\s+atac)\b", re.I)


def get(url, tries=4, timeout=90):
    err = None
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "s9-prefilter"}), timeout=timeout).read()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 * (i + 1))
    raise err


def ena_rows(runs):
    out = {}
    for i in range(0, len(runs), 100):
        chunk = runs[i:i + 100]
        q = urllib.parse.quote(" OR ".join('run_accession="%s"' % r for r in chunk))
        d = json.loads(get("https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&query=%s&fields=%s&format=json&limit=0" % (q, ENA_FIELDS)))
        for r in d:
            out[r["run_accession"]] = r
    return out


def sra_stats(runs):
    """efetch db=sra XML -> {run: {'nreads', 'reads': [(average, stdev), ...], 'spots', 'bases'}}. NCBI allows 3 requests/s without a key."""
    out = {}
    for i in range(0, len(runs), 50):
        chunk = runs[i:i + 50]
        xml = get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&retmode=xml&id=" + ",".join(chunk))
        root = ET.fromstring(xml)
        for run in root.iter("RUN"):
            acc = run.get("accession")
            st = run.find("Statistics")
            if acc is None or st is None:
                continue
            reads = sorted(((int(r.get("index")), float(r.get("average") or 0), float(r.get("stdev") or 0)) for r in st.findall("Read")))
            out[acc] = {"nreads": len(reads), "reads": [(a, s) for _, a, s in reads],
                        "spots": int(run.get("total_spots") or 0), "bases": int(run.get("total_bases") or 0)}
        time.sleep(0.4)
    return out


def annotate(runs):
    ena, sra = ena_rows(runs), sra_stats(runs)
    return [{"run": r, "ena": ena.get(r), "sra": sra.get(r)} for r in runs]


def accept(m):
    """-> (True, '') | (False, reason). m = one annotate() record."""
    e, s = m.get("ena"), m.get("sra")
    if not e:
        return False, "not_in_ena"
    if e.get("library_strategy") != "RNA-Seq":
        return False, "strategy_%s" % e.get("library_strategy")
    if e.get("instrument_platform") != "ILLUMINA":
        return False, "platform_%s" % e.get("instrument_platform")
    if "SINGLE CELL" not in (e.get("library_source") or ""):
        return False, "source_%s" % e.get("library_source")
    text = " ".join(e.get(k) or "" for k in ("experiment_title", "study_title", "library_name"))
    if NON_GEX.search(text):
        return False, "text_%s" % NON_GEX.search(text).group(0).lower()
    if not s:
        return True, ""  # no SRA statistics: let the decode gate decide
    informative = [(a, sd) for a, sd in s["reads"] if a >= 20]  # drop 8/10-nt index reads
    if len(informative) < 2:
        return False, "single_read_spot"
    if not any(abs(a - round(a)) < 0.01 and sd < 0.01 and int(round(a)) in BARCODE_LENS for a, sd in informative):
        return False, "no_fixed_barcode_read_%s" % ",".join("%d" % round(a) for a, _ in informative)
    if not any(a >= 40 for a, _ in informative):
        return False, "no_cdna_read"
    return True, ""


def main(argv):
    if argv[0] == "annotate":
        runs = [x.strip() for x in open(argv[1]) if x.strip()]
        with open(argv[2], "w") as f:
            for r in annotate(runs):
                f.write(json.dumps(r) + "\n")
        print("annotated", len(runs))
    elif argv[0] == "eval":
        recs = [json.loads(l) for l in open(argv[1])]
        labels = json.load(open(argv[2]))  # {run: "staged" | "parked_non10x" | "parked_no_gex"}
        rows = {"staged": [0, 0], "parked": [0, 0]}
        for m in recs:
            lab = labels.get(m["run"])
            if not lab:
                continue
            ok, why = accept(m)
            k = "staged" if lab == "staged" else "parked"
            rows[k][0 if ok else 1] += 1
            print("%-12s %-14s %s %s" % (m["run"], lab, "ACCEPT" if ok else "reject", why))
        print("staged: accepted %d rejected %d | parked: accepted %d rejected %d" % (*rows["staged"], *rows["parked"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
