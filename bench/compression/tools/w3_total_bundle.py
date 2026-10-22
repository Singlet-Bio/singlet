#!/usr/bin/env python
"""GSE145370 bundle table for w3_total_v1.

Reads the original archive's member sizes (zipfile, data only), the measured w3_total_v1 streams of the 28 exon / intron
matrices (results/w3_total_bundle*.tsv, results/w3_total_bundle_streams/) and the called-set streams
(results/w3_total_calls_bundle.tsv), and writes the new archive corpus_w3_total/GSE145370.w3.zip whose size is the measured
bundle number.  Prints the markdown tables.

    python -I tools/w3_total_bundle.py            (run from bench/compression)

The new archive holds, per sample: exon_counts.w3v1, intron_counts.w3v1, calls.w3 (called set), sj_counts.1pz,
vdj_gene_usage.1pz, mt_heteroplasmy.1pz (these three unchanged: not recoded), summary.json, pileup_stats.json,
provenance.json (unchanged); and at the study level feature_vocab.json, manifest.json, study_meta.json (unchanged).
Dropped as derivable (E4): splice_psi.1pz and cell_calls.tsv (replaced by the called-set stream).
"""
import csv, collections, os, re, sys, zipfile

SRC = "corpus/bundles/GSE145370.singlet"
OUT = "corpus_w3_total/GSE145370.w3.zip"
R = "results/"
PAT = re.compile(r"^(?:samples/(GSM[0-9]+)/)?([A-Za-z_0-9.]+)$")


def main():
    z = zipfile.ZipFile(SRC)
    src_size = os.path.getsize(SRC)
    infos = z.infolist()
    today = collections.defaultdict(lambda: [0, 0])      # member kind -> [files, bytes]
    for i in infos:
        kind = i.filename.split("/")[-1]
        today[kind][0] += 1
        today[kind][1] += i.compress_size
    members_total = sum(v[1] for v in today.values())
    overhead = src_size - members_total

    rows = list(csv.DictReader(open(R + "w3_total_bundle.tsv"), delimiter="\t"))
    v1 = {}
    for r in rows:
        if r["codec"] != "w3_total_v1":
            continue
        assert r["matrix_ok"] == "1", r
        v1[(r["sample"], r["member"])] = int(r["enc_bytes"])
    calls = {}
    for r in csv.DictReader(open(R + "w3_total_calls_bundle.tsv"), delimiter="\t"):
        assert r["ok"] == "1"
        gsm = r["file"].split("/")[-2]
        calls[gsm] = int(r["calls_bytes"])
    samples = sorted({k[0] for k in v1})
    assert len(samples) == 14 and len(v1) == 28, (len(samples), len(v1))

    new = collections.defaultdict(int)
    for s in samples:
        new["exon_counts"] += v1[(s, "exon_counts.1pz")]
        new["intron_counts"] += v1[(s, "intron_counts.1pz")]
        new["calls"] += calls[s]

    # real archive
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    keep_raw = {"sj_counts.1pz", "vdj_gene_usage.1pz", "mt_heteroplasmy.1pz"}
    keep_json = {"summary.json", "pileup_stats.json", "provenance.json"}
    study_keep = {"feature_vocab.json", "manifest.json", "study_meta.json"}
    with zipfile.ZipFile(OUT, "w") as w:
        for i in infos:
            m = PAT.match(i.filename)
            if not m:
                raise SystemExit("unexpected member name " + i.filename)
            gsm, name = m.group(1), m.group(2)
            if gsm is None:
                if name in study_keep:
                    w.writestr(name, z.read(i), compress_type=zipfile.ZIP_DEFLATED)
                continue
            base = f"samples/{gsm}/"
            if name in keep_raw:
                w.writestr(base + name, z.read(i), compress_type=zipfile.ZIP_STORED)
            elif name in keep_json:
                w.writestr(base + name, z.read(i), compress_type=zipfile.ZIP_DEFLATED)
            elif name == "exon_counts.1pz":
                w.writestr(base + "exon_counts.w3v1", open(f"{R}w3_total_bundle_streams/{gsm}__exon_counts.1pz__w3_total_v1.bin", "rb").read(), compress_type=zipfile.ZIP_STORED)
                w.writestr(base + "calls.w3", open(f"{R}w3_total_bundle_streams/corpus_w3_total_GSE145370_{gsm}_exon_counts.1pz.calls", "rb").read(), compress_type=zipfile.ZIP_STORED)
            elif name == "intron_counts.1pz":
                w.writestr(base + "intron_counts.w3v1", open(f"{R}w3_total_bundle_streams/{gsm}__intron_counts.1pz__w3_total_v1.bin", "rb").read(), compress_type=zipfile.ZIP_STORED)
            # splice_psi.1pz and cell_calls.tsv: derived, dropped
    out_size = os.path.getsize(OUT)
    with zipfile.ZipFile(OUT) as w:
        assert w.testzip() is None
        n_entries = len(w.infolist())
        new_members = sum(i.compress_size for i in w.infolist())
    new_overhead = out_size - new_members

    def row(label, a, b, note=""):
        print(f"| {label} | {a:,} | {b:,} | {note} |")

    print(f"## Table B1. GSE145370 bundle, today vs w3_total_v1 (bytes)\n")
    print("| member kind | today | w3_total (measured) | treatment |")
    print("|---|---|---|---|")
    row("exon_counts.1pz (14)", today["exon_counts.1pz"][1], new["exon_counts"], "w3_total_v1, bit-exact matrix on all 14")
    row("intron_counts.1pz (14)", today["intron_counts.1pz"][1], new["intron_counts"], "w3_total_v1, bit-exact matrix on all 14")
    row("sj_counts.1pz (14)", today["sj_counts.1pz"][1], today["sj_counts.1pz"][1], "kept as today: not recoded (E5 not run)")
    row("vdj_gene_usage.1pz (14)", today["vdj_gene_usage.1pz"][1], today["vdj_gene_usage.1pz"][1], "kept as today: not recoded")
    row("mt_heteroplasmy.1pz (14)", today["mt_heteroplasmy.1pz"][1], today["mt_heteroplasmy.1pz"][1], "kept as today (0x0 stubs)")
    row("splice_psi.1pz (14)", today["splice_psi.1pz"][1], 0, "derived from sj_counts (E4: 14/14 byte-identical); not rebuilt here")
    row("cell_calls.tsv (14)", today["cell_calls.tsv"][1], new["calls"], "called-set stream (round trip verified); the TSV regenerates from it + exon column sums (E4)")
    js_today = sum(today[k][1] for k in ("summary.json", "pileup_stats.json", "provenance.json", "manifest.json", "study_meta.json", "feature_vocab.json"))
    row("JSON members", js_today, js_today, "kept as today (E4 lists derivable numeric fields, not removed here)")
    row("zip framing (central directory + local headers)", overhead, new_overhead, f"measured: {n_entries} entries now, {len(infos)} before")
    tot_today = src_size
    tot_new = out_size
    print(f"| **total** | **{tot_today:,}** | **{tot_new:,}** | ratio {tot_new/tot_today:.4f} = {tot_today/tot_new:.2f}x smaller; file `{OUT}` |")
    assert abs(sum(v[1] for v in today.values()) + overhead - tot_today) == 0

    # shares
    print("\nShare of today's bundle: " + ", ".join(f"{k} {100*today[k][1]/tot_today:.1f}%" for k in ("exon_counts.1pz", "intron_counts.1pz", "sj_counts.1pz", "splice_psi.1pz", "vdj_gene_usage.1pz", "cell_calls.tsv")))
    other = tot_new - new["exon_counts"] - new["intron_counts"]
    print(f"\nNew bundle: exon+intron {new['exon_counts']+new['intron_counts']:,} ({100*(new['exon_counts']+new['intron_counts'])/tot_new:.1f}%), "
          f"sj+vdj+mt unchanged {today['sj_counts.1pz'][1]+today['vdj_gene_usage.1pz'][1]+today['mt_heteroplasmy.1pz'][1]:,} "
          f"({100*(today['sj_counts.1pz'][1]+today['vdj_gene_usage.1pz'][1]+today['mt_heteroplasmy.1pz'][1])/tot_new:.1f}%), everything else {other - (today['sj_counts.1pz'][1]+today['vdj_gene_usage.1pz'][1]+today['mt_heteroplasmy.1pz'][1]):,}")

    print("\n## Table B2. per sample (bytes)\n")
    print("| sample | called | exon today | exon w3 | intron today | intron w3 | calls stream |")
    print("|---|---|---|---|---|---|---|")
    ex_t = {}
    for i in infos:
        parts = i.filename.split("/")
        if len(parts) == 3:
            ex_t[(parts[1], parts[2])] = i.compress_size
    cr = {r["file"].split("/")[-2]: r for r in csv.DictReader(open(R + "w3_total_calls_bundle.tsv"), delimiter="\t")}
    for s in samples:
        print(f"| {s} | {int(cr[s]['called']):,} | {ex_t[(s,'exon_counts.1pz')]:,} | {v1[(s,'exon_counts.1pz')]:,} | {ex_t[(s,'intron_counts.1pz')]:,} | {v1[(s,'intron_counts.1pz')]:,} | {calls[s]:,} |")

    # secondary, not measured: Track A's accounting for the matrices that were not recoded
    print("\n## Table B3. Same bundle with Track A's container treatment applied to sj_counts and vdj (PROJECTION, not a measured codec)\n")
    sj_chunks, sj_other, sj_rc, vdj_chunks, vdj_other, vdj_rc, junction_subset = 13172766, 48920, 555130, 1749172, 25434, 6134, 570859
    proj_sj = sj_chunks + sj_other + sj_rc + junction_subset
    proj_vdj = vdj_chunks + vdj_other + vdj_rc
    print(f"sj_counts {today['sj_counts.1pz'][1]:,} -> about {proj_sj:,} (chunks {sj_chunks:,} + other {sj_other:,} + row counts {sj_rc:,} + study junction vocabulary subset coding {junction_subset:,}); "
          f"vdj_gene_usage {today['vdj_gene_usage.1pz'][1]:,} -> about {proj_vdj:,}. New bundle would be about "
          f"{tot_new - today['sj_counts.1pz'][1] - today['vdj_gene_usage.1pz'][1] + proj_sj + proj_vdj:,} B (these use Track A's results/track_a_bundle_whatif.md.txt numbers, payload unchanged).")


if __name__ == "__main__":
    main()
