#!/usr/bin/env python3
"""S15: does a Singlet route exist today for each corpus row? yes / no / planned. Reads corpus.md (section 6) and the engine's known protocol list
(include/singlet/fq/protocol.h on prod/campaign); never runs Singlet.
usage: singlet_routes.py CORPUS_MD OUT_TSV"""
import re, sys, csv

corpus = open(sys.argv[1], encoding="utf-8").read()
s0 = corpus.index("## 6. Final corpus table"); s1 = corpus.index("## 7. Tier check")
rows = []
for l in corpus[s0:s1].splitlines():
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    if len(c) >= 9 and c[0].isdigit(): rows.append(c)

# protocol -> (engine tag in known_protocols(), engine R1 length, note)
SPEC = [
    ("10x multiome", "10x-arc-gex", 28), ("10x 3p v3", "10x-3p-v3", 28), ("10x 3p v2", "10x-3p-v2", 26), ("10x 3p v1", "10x-3p-v1", 24),
    ("10x 5p v1", "10x-5p-v2", 26), ("10x 5p v2", "10x-5p-v2", 26), ("10x 5p", "10x-5p-v3", 28), ("10x 3p", "10x-3p-v3", 28),
    ("drop-seq", "dropseq", 20), ("seq-well", "seqwell", 20), ("split", "splitseq", 66), ("indrops", "indrop", 46), ("cel-seq2", "celseq2", 12),
    ("ddseq", "ddseq", 68), ("sci-rna", "sci-rna-seq3", 34), ("microwell", "microwell-seq", 54), ("visium", "10x-visium", 28), ("scnt", "dropseq", 20),
    ("sndrop", "dropseq", 20), ("scrb", None, 0), ("icell8", None, 0), ("smart-seq", None, 0), ("smartseq", None, 0), ("long", None, 0),
]
out = []
for c in rows:
    n, i, acc, sp, proto, outs, truth, status, fetch = c[:9]
    p = proto.lower(); f = fetch.lower()
    tag = None; r1 = 0
    for k, t, ln in SPEC:
        if k in p: tag, r1 = t, ln; break
    notes = []
    if "pbmc2" in i or "lr_" in i or "long" in p:
        route, note = "planned", "long-read comparison set; engine input is short-read"
    elif tag is None:
        if "smart-seq" in p or "smartseq" in p or "plate" in p or "per-cell" in f:
            route, note = "no", "plate-based per-cell FASTQ: no per-cell route in the engine (no barcode read)"
        elif "icell8" in p or "scrb" in p:
            route, note = "no", "no %s protocol spec in known_protocols()" % ("ICELL8" if "icell8" in p else "SCRB-seq")
        else:
            route, note = "no", "no protocol spec for '%s'" % proto
    else:
        sra = bool(re.search(r"sra|ena fastq", f)) and "10x url" not in f and "submitted bam" not in f and "bam" not in f.split("(")[0]
        if "10x url" in f or "bam" in f or "clipper" in f or "globus" in f or "tar" in f:
            route, note = "planned", "protocol %s exists, but the engine stages reads from SRA runs only (FASTQ and BAM inputs are an engine item)" % tag
        elif sra:
            route, note = "yes", "protocol %s in known_protocols() and reads come from an SRA run" % tag
            # read-length caveats from corpus 1.3
            if tag in ("seqwell",) and "32" in outs + proto + c[9]: note += "; R1 is 32 nt against 20 expected, untested"
        else:
            route, note = "planned", "protocol %s exists, input route unclear" % tag
        if i in ("seq-well", "gierahn2017_seqwell_mix"): note += "; R1 32 nt against 20 expected, detection untested"
        if tag == "dropseq" and "mereu" in i: route, note = "planned", "Drop-seq spec expects R1 of 20 nt, Mereu R1 is 76 nt: needs an R1-length rule"
        if tag in ("celseq2", "indrop", "ddseq") and "mereu" in i: route, note = "planned", "spec exists (%s) but expects R1 of %d nt; Mereu layout differs" % (tag, r1)
        if i in ("tian2019_celseq2_5cl_p1", "tian2019_rnamix_celseq2"): route, note = "planned", "celseq2 spec expects R1 of 12 nt, Tian R1 is 32 nt"
        if "sci-rna" in p: route, note = "no", "sci-RNA-seq3 spec exists, Cao 2017 is sci-RNA-seq v1 (R1 18 nt): different layout"
    if "gemcode" in i or "zheng2017" in i: route, note = "planned", "10x v1 (GemCode): spec 10x-3p-v1 exists, but the SRA runs lack R1; the 10x tar is a FASTQ input (engine item)"
    out.append((n, i, proto, route, note))
with open(sys.argv[2], "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n"); w.writerow(["corpus_row", "dataset_id", "protocol", "singlet_route", "singlet_route_note"]); w.writerows(out)
from collections import Counter
print(Counter(o[3] for o in out))
