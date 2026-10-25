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
    write_provenance(bp, out)
    print("download tables written to", out)


METHODS = """**Runs.** Four 10x 3' GEX runs of the atlas re-run universe (SRR24098804, SRR25032524, SRR20115176, SRR23322292; NovaSeq, 31.9 to 35.9 M read pairs, 1.2 to 3.4 GB as .sra, 132 M read pairs in all), drawn at random from runs with two ENA FASTQ files and checked to pass `singlet download`'s 10x gate. Every tool fetches the same four runs.

**Protocol.** Each tool fetches the four runs as four concurrent streams (one run each) from a Clipper compute node into node-local /tmp, from a fresh HOME so that no sra-tools cache is shared between tools. Tools run one after the other in a job; the order is reversed in the second of two repetitions and ENA runs in its own job. Clipper's link is shared with other users, so throughput varies between repetitions; the spread between repetitions is used as the tolerance. Per stream the clock runs from launch to the last byte decoded (scoring happens after the clock stops).

**Throughput.** Effective: bytes of the run as an .sra file (the same denominator for every tool, the size the archive stores) divided by seconds until the tool has delivered its output, and read pairs per second. AWS S3 and prefetch only move the file; the other tools also decode. Singlet's orchestrator stage also screens the reads twice.

**Completeness.** Truth is NCBI's own decode of the archive (`prefetch`, then `fasterq-dump --split-files`). A run is complete when the spot count of a tool's decoded output equals the archive's and the order-independent digest of its decoded sequences (a sum of per-read CRC32/Adler32 hashes) equals a decoded file of the truth. For AWS S3 the sha256 of the .sra must equal the prefetched file's. Singlet's default path trims poly(A) and adapters and drops base qualities by design, so identity is scored on the same code path with `--no-trim --quality full`.

**Singlet.** `singlet` is the I/O stage of the campaign orchestrator (feat/campaign-orchestrator, stages.io_stage) for one run: a 1 M-read head download, `screen-runs`, `prefetch`, a local `singlet download`, and a second `screen-runs`. `singlet-prefetch-local` is the same without the screens; `singlet-remote` is the 2026 campaign's VDB streaming."""


def write_provenance(bp: Path, out: Path):
    import subprocess
    src = Path(__file__).resolve().parents[4]
    commit = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    sw = "sra-tools 3.4.1"
    tools = [
        {"name": "singlet", "version": "1.0.0", "role": "singlet", "params": "I/O stage of feat/campaign-orchestrator for one run: singlet download SRR --max-reads 1000000 --vdb-threads 2 --no-profile; singlet screen-runs; prefetch --max-size u -O sra SRR; singlet download sra/SRR/SRR.sra --vdb-threads 1 --no-profile -o SRR.1fq; singlet screen-runs"},
        {"name": "singlet-prefetch-local", "version": "1.0.0", "role": "competitor", "params": "prefetch --max-size u -O sra SRR; singlet download sra/SRR/SRR.sra -o SRR.1fq --no-profile (Singlet's own path without the screens)"},
        {"name": "singlet-remote", "version": "1.0.0", "role": "competitor", "params": "singlet download SRR -o SRR.1fq --no-profile (VDB streaming, the 2026 campaign path)"},
        {"name": "fasterq-dump", "version": sw, "role": "competitor", "url": "https://github.com/ncbi/sra-tools", "params": "fasterq-dump --split-files -e 4 -t tmp -O fq SRR"},
        {"name": "prefetch+fasterq-dump", "version": sw, "role": "competitor", "url": "https://github.com/ncbi/sra-tools", "params": "prefetch --max-size u -O sra SRR; fasterq-dump --split-files -e 4 -t tmp -O fq sra/SRR/SRR.sra"},
        {"name": "aws-s3", "version": "curl", "role": "competitor", "url": "https://registry.opendata.aws/ncbi-sra/", "params": "curl -o SRR.sra https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR/SRR"},
        {"name": "ena-https", "version": "curl", "role": "competitor", "url": "https://www.ebi.ac.uk/ena/", "params": "curl -O https://<fastq_ftp of the ENA filereport> for every file of the run"},
        {"name": "sra-toolkit-reference", "version": sw, "role": "truth", "params": "prefetch --max-size u; fasterq-dump --split-files -e 16 of the archive"}]
    prov = {
        "bench_commit": commit, "status": "final",
        "generated_at": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%S+00:00"], text=True).strip(),
        "singlet": {"version": "1.0.0", "commit": "02aa9b938b54e0f92bba00c44c07b883a8660aa2", "branch": "prod/campaign", "binary": "singlet-campaign",
                    "binary_sha256": "261d3a50ed624572418d30489dfd58904323f6e0d8820bd5260dfe7fb5038e6c", "build_flags": "PGO, -march=x86-64-v3"},
        "methods": METHODS, "tools": tools,
        "datasets": [{"id": "sra_download_set", "title": "Four 10x 3' GEX runs from the atlas re-run universe", "accessions": {"srr": ["SRR24098804", "SRR25032524", "SRR20115176", "SRR23322292"]},
                      "truth": "NCBI's own decode of the archive (prefetch + fasterq-dump)"}],
        "caveats": [
            "Clipper's uplink is shared with every other job and user, so absolute throughput varies with the time of day; the repetitions alternate tool order and the spread is the tolerance.",
            "Four runs and two repetitions: a small sample of the network, not a bandwidth survey. The earlier 8 and 16 stream aggregate measurements (125 and 160 MB/s into Clipper) are in docs/atlas-rerun-plan.md and were not repeated.",
            "AWS S3 and prefetch only move the archive file; they are not decoders, so their effective throughput is an upper bound on a decode-inclusive path.",
            "Singlet's default path is lossy by design (poly(A) and adapter trimming, no base qualities); identity is scored on the lossless configuration of the same code path.",
            "The reference decode uses fasterq-dump without technical reads; Singlet's R1 and R2 are matched to its files by content."]}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")


if __name__ == "__main__":
    main()
