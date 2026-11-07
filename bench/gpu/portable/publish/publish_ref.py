#!/usr/bin/env python3
"""Manifest and upload of the portable bundle's reference tars (data.singlet.bio/reference/<version>/).

  publish_ref.py manifest --out DIR --version v1     writes DIR/manifest.json from DIR/SHA256SUMS and DIR/*.members.sha256
  publish_ref.py upload   --out DIR --version v1     uploads the tars, then manifest.json, to the public bucket; never overwrites
                                                     an object with other content; checks each object at its public URL

Credentials come from the environment (R2_ACCOUNT_ID or R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY) and are never printed.
"""
import argparse, json, os, sys, urllib.request

BUCKET = "singlet-data"            # served at https://data.singlet.bio/
PUBLIC = "https://data.singlet.bio"

SETS = {
    "star_2.7.11b-GRCh38-2024-A.tar": {
        "unpacks_to": "star_2.7.11b/ (place it as <ref>/GRCh38-2024-A/star_2.7.11b)",
        "contents": "STAR 2.7.11b genome index of the 10x Genomics GRCh38-2024-A reference, with singlet's minimizer and transcript-region indices",
        "sources": [
            {"name": "10x Genomics reference GRCh38-2024-A (cellranger mkref 8.0.0)",
             "url": "https://cf.10xgenomics.com/supp/cell-exp/refdata-gex-GRCh38-2024-A.tar.gz",
             "files": ["fasta/genome.fa (from Ensembl Homo_sapiens.GRCh38.dna.primary_assembly.fa, release 110, as modified by 10x)",
                       "genes/genes.gtf (from GENCODE v44 primary assembly annotation, filtered by 10x)"],
             "license": "Ensembl data carry no restrictions on use (https://www.ensembl.org/info/about/legal/disclaimer.html); "
                        "GENCODE data are freely available without restriction (https://www.gencodegenes.org/pages/data_access.html)"},
        ],
        "build": [
            "STAR 2.7.11b: STAR --runMode genomeGenerate --runThreadN 8 --genomeDir star_2.7.11b --genomeFastaFiles genome.fa "
            "--genomeSAsparseD 2 --sjdbGTFfile genes.gtf (defaults: sjdbOverhang 100, genomeSAindexNbases 14; see genomeParameters.txt)",
            "minimizerIndex.bin: singlet genome build-minimizers --genome-dir star_2.7.11b",
            "SA_tx, SAindex_tx, txFirstInfo.tab: singlet's suffix array of the annotated exonic regions, derived from the files above "
            "(the builder command is not recorded)",
        ],
    },
    "nonhost-refseq223.tar": {
        "unpacks_to": "nonhost/ (place it as <ref>/nonhost)",
        "contents": "singlet's non-host screening indices (viral, bacterial, fungal) and the human 21-mer Bloom filter",
        "sources": [
            {"name": "NCBI RefSeq release 223 (April 2024): viral genomes (about 19,149), complete bacterial reference genomes (about 6,478), "
                     "fungal release files (about 37)",
             "url": "https://ftp.ncbi.nlm.nih.gov/refseq/release/",
             "license": "NCBI places no restrictions on the use or distribution of GenBank and RefSeq data "
                        "(https://www.ncbi.nlm.nih.gov/home/about/policies/); submitters may claim rights in individual records"},
            {"name": "host_21mer.bloom: the GRCh38-2024-A genome.fa above", "url": "https://cf.10xgenomics.com/supp/cell-exp/refdata-gex-GRCh38-2024-A.tar.gz",
             "license": "as the STAR index sources"},
        ],
        "build": [
            "viral.snhskidx: singlet build-nonhost-db viral.snhskidx --kingdom VIRAL --fasta <RefSeq 223 viral genomes> (minimizers k=21, w=11)",
            "bacterial.snhskidx: singlet build-nonhost-db bacterial.snhskidx --kingdom BACTERIAL --fasta <RefSeq 223 complete bacterial genomes>",
            "fungal.snhskidx: singlet build-nonhost-db fungal.snhskidx --kingdom FUNGAL --fasta <RefSeq 223 fungal release files>",
            "host_21mer.bloom: singlet build-host-filter --genome-fasta genome.fa (k=21, w=11)",
            "The exact RefSeq input file lists are not recorded; the sha256 values match state/reference-manifest-v1.yaml",
        ],
    },
}


def read_sums(path):
    out = {}
    for line in open(path):
        h, name = line.split(None, 1)
        out[name.strip().lstrip("*")] = h
    return out


def manifest(a):
    tars = read_sums(os.path.join(a.out, "SHA256SUMS"))
    m = {"version": a.version, "base_url": f"{PUBLIC}/reference/{a.version}/", "created": a.created,
         "purpose": "inputs of the portable GPU timing bundle (bench/gpu/portable in Singlet-Bio/singlet)", "sets": []}
    for name, meta in SETS.items():
        members = read_sums(os.path.join(a.out, name.replace(".tar", ".members.sha256")))
        sizes = {p: int(b) for b, p in (l.split(None, 1) for l in open(os.path.join(a.out, name.replace(".tar", ".members.size"))))}
        sizes = {p.strip(): b for p, b in sizes.items()}
        m["sets"].append({"file": name, "url": f"{PUBLIC}/reference/{a.version}/{name}",
                          "bytes": os.path.getsize(os.path.join(a.out, name)), "sha256": tars[name], **meta,
                          "members": [{"path": p, "bytes": sizes[p], "sha256": h} for p, h in sorted(members.items())]})
    extra = os.path.join(a.out, "inputs.sha256")
    if os.path.exists(extra):
        m["source_file_sha256"] = read_sums(extra)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(m, f, indent=1)
        f.write("\n")
    print(json.dumps({s["file"]: [s["bytes"], s["sha256"]] for s in m["sets"]}, indent=1))


def client():
    import boto3
    from botocore.config import Config
    kw = dict(signature_version="s3v4", retries={"max_attempts": 12, "mode": "adaptive"}, max_pool_connections=64,
              connect_timeout=30, read_timeout=300, s3={"addressing_style": "path"})
    for opt in ("request_checksum_calculation", "response_checksum_validation"):
        try:
            Config(**{opt: "when_required"})
            kw[opt] = "when_required"
        except TypeError:
            pass
    ep = os.environ.get("R2_ENDPOINT") or f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    return boto3.client("s3", endpoint_url=ep, aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto", config=Config(**kw))


def upload(a):
    from boto3.s3.transfer import TransferConfig
    from botocore.exceptions import ClientError
    c = client()
    tars = read_sums(os.path.join(a.out, "SHA256SUMS"))
    import hashlib
    msha = hashlib.sha256(open(os.path.join(a.out, "manifest.json"), "rb").read()).hexdigest()
    files = [(n, tars[n], "application/x-tar") for n in SETS] + [("manifest.json", msha, "application/json")]
    cfg = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=256 << 20, max_concurrency=16)
    for name, sha, ctype in files:
        path, key = os.path.join(a.out, name), f"reference/{a.version}/{name}"
        size = os.path.getsize(path)
        try:
            h = c.head_object(Bucket=BUCKET, Key=key)
            if h["ContentLength"] == size and h.get("Metadata", {}).get("sha256") == sha:
                print(f"present {key} ({size} bytes, sha256 {sha})", flush=True)
                continue
            sys.exit(f"REFUSED: {key} exists with other content ({h['ContentLength']} bytes, sha256 {h.get('Metadata', {}).get('sha256')})")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                raise
        print(f"uploading {key} ({size} bytes)", flush=True)
        c.upload_file(path, BUCKET, key, Config=cfg, ExtraArgs={
            "ContentType": ctype, "Metadata": {"sha256": sha},
            "CacheControl": "public, max-age=31536000, immutable" if ctype != "application/json" else "public, max-age=300"})
        h = c.head_object(Bucket=BUCKET, Key=key)
        assert h["ContentLength"] == size, (key, h["ContentLength"], size)
        print(f"uploaded {key}", flush=True)
    for name, sha, _ in files:
        url = f"{PUBLIC}/reference/{a.version}/{name}"
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers={"User-Agent": "singlet-publish"}), timeout=60)
        print(f"public {url}: HTTP {r.status}, {r.headers.get('Content-Length')} bytes", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["manifest", "upload"])
    p.add_argument("--out", required=True)
    p.add_argument("--version", default="v1")
    p.add_argument("--created", default="2026-11-07")
    a = p.parse_args()
    manifest(a) if a.cmd == "manifest" else upload(a)
