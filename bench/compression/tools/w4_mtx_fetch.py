#!/usr/bin/env python
"""w4_mtx: fetch whole sample members (legacy STARsolo + 1pz) from public .singlet bundles by HTTP range requests.
    python -P tools/w4_mtx_fetch.py GSE309702:GSM9277757 [GSE..:GSM..] [--list]
Reuses HttpFile from tools/fetch_corpus.py. Output: corpus_w4_mtx/<GSE>.<GSM>/<member path with / -> __>, study files in corpus_w4_mtx/<GSE>/.
"""
import io, os, sys, json, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_corpus as F
OUT = os.path.join(os.path.dirname(HERE), "corpus_w4_mtx")
SAMPLE = ["star_Solo.out/Gene/raw/matrix.mtx", "star_Solo.out/Gene/raw/barcodes.tsv", "star_Solo.out/Gene/raw/features.tsv",
          "star_Solo.out/Gene/filtered/matrix.mtx", "star_Solo.out/Gene/filtered/barcodes.tsv", "star_Solo.out/Gene/filtered/features.tsv",
          "star_Solo.out/Gene/Summary.csv", "star_Solo.out/Gene/UMIperCellSorted.txt", "star_Solo.out/Gene/Features.stats", "star_Solo.out/Barcodes.stats",
          "ambient_profile.tsv", "gene_expression.tsv", "cell_qc_metrics.tsv", "star_SJ.out.tab", "cell_calls.tsv",
          "exon_counts.1pz", "intron_counts.1pz", "summary.json", "ambient_contamination.tsv", "ambient_contamination.json",
          "doublet_scores.tsv", "metrics_summary.csv", "provenance.json", "star_Log.final.out"]
STUDY = ["feature_vocab.json", "manifest.json"]

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    listonly = "--list" in sys.argv
    total = 0
    for a in args:
        gse, gsm = a.split(":")
        hf = F.HttpFile(F.BASE.format(gse=gse))
        zf = zipfile.ZipFile(io.BufferedReader(hf, buffer_size=1 << 20))
        names = set(zf.namelist())
        sd = os.path.join(OUT, gse); os.makedirs(sd, exist_ok=True)
        want = [(w, os.path.join(sd, w)) for w in STUDY]
        d = os.path.join(OUT, f"{gse}.{gsm}"); os.makedirs(d, exist_ok=True)
        if not listonly:
            want += [(f"samples/{gsm}/{m}", os.path.join(d, m.replace("/", "__"))) for m in SAMPLE]
        else:
            want = [(w, p) for w, p in want if w == "manifest.json"]
        for w, dst in want:
            if w not in names:
                print("  missing", w); continue
            zi = zf.getinfo(w)
            if os.path.exists(dst) and os.path.getsize(dst) == zi.file_size: continue
            with zf.open(w) as src, open(dst, "wb") as f:
                f.write(src.read())
        print(f"[{gse}:{gsm}] http bytes {hf.bytes_fetched/1e6:.1f} MB", flush=True)
        total += hf.bytes_fetched
    print(f"TOTAL http {total/1e6:.1f} MB")
main()
