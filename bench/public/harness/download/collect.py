"""Gather the raw outputs of dl_reference.sbatch and dl_bench.sbatch into the small artifact tables.

    python collect.py BP_DIR OUT_DIR

BP_DIR   $BP (…/bench-public): reads download/reference/{sra.tsv,reference.tsv} and download/rep<N>/<mode>.{streams,digest}.tsv
OUT_DIR  $ART/download/: writes streams.tsv, digests.tsv, reference.tsv and sra.tsv. Standard library only.
"""
import csv
import json
import shutil
import sys
from pathlib import Path


def read_tsv(p: Path):
    with open(p, newline="") as fh:
        return [r for r in csv.reader(fh, delimiter="\t") if r]


def main():
    bp, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ref = bp / "download" / "reference"
    shutil.copy(ref / "sra.tsv", out / "sra_files.tsv")
    with open(out / "reference.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["file", "spots", "read_len", "seq_digest", "qual_digest"])
        for r in read_tsv(ref / "reference.tsv"):
            w.writerow(r)
    sw = csv.writer(open(out / "streams.tsv", "w", newline=""), delimiter="\t")
    sw.writerow(["rep", "mode", "srr", "start_s", "end_s", "rc", "payload_bytes"])
    dw = csv.writer(open(out / "digests.tsv", "w", newline=""), delimiter="\t")
    dw.writerow(["rep", "mode", "srr", "file", "spots", "read_len", "seq_digest", "qual_digest"])
    mw = csv.writer(open(out / "memory.tsv", "w", newline=""), delimiter="\t")
    mw.writerow(["rep", "mode", "peak_rss_gb"])
    for rd in sorted((bp / "download").glob("rep*")):
        rep = rd.name[3:]
        for f in sorted(rd.glob("*.mem.json")):
            mw.writerow([rep, f.name[: -len(".mem.json")], round(json.loads(f.read_text())["peak_rss_gb"], 3)])
        for f in sorted(rd.glob("*.streams.tsv")):
            mode = f.name[: -len(".streams.tsv")]
            for r in read_tsv(f):
                sw.writerow([rep, mode] + r)
        for f in sorted(rd.glob("*.digest.tsv")):
            mode = f.name[: -len(".digest.tsv")]
            for r in read_tsv(f):
                dw.writerow([rep, mode] + r)
    print("download tables written to", out)


if __name__ == "__main__":
    main()
