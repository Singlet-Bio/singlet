#!/usr/bin/env python3
"""Keep every genotype output of a campaign sample on Clipper (D-52, 2026-11-06), before the worker deletes its scratch.

Copies from the sample's engine out dir into <genotype_dir>/<GSE>/<GSM>/ (directories 0700, files 0600, owner-only):
  private/genotype/      donors.vcf.gz (GT:GQ:AD:DP:PL per donor) and cellsnp/ (cellsnp-lite layout of the called cells)
  private/donor/         the engine's per-donor VCFs and coverage tables
  private/snp_ad.*, private/snp_dp.*, private/ase_counts.tsv   per-cell allele counts of every panel SNP and barcode
  donor_assignments.tsv, cell_calls.tsv, sex_call.json, ancestry_call.json, summary.json   (context for the genotypes)
and writes manifest.json (sha256 and bytes of each file). Never public: genotype_dir is under $WORK/genotypes/, outside every path that
reaches R2, D1 or singlet.bio, and the release tier is a separate owner decision (D-54). A sample without SNP outputs (no panel, e.g.
species without one) gets a manifest with status "no_snp_data".
"""
import hashlib
import json
import os
import shutil

DEFAULT_DIR = "/mnt/projects/debruinz_project/debruinz/singlet/genotypes/beta"
PRIVATE = ("genotype", "donor", "snp_ad.1pz", "snp_dp.1pz", "ase_counts.tsv", "snp_ad.mtx.gz", "snp_dp.mtx.gz",
           "snp_ad_features.tsv.gz", "snp_ad_barcodes.tsv.gz", "snp_dp_features.tsv.gz", "snp_dp_barcodes.tsv.gz")
CONTEXT = ("donor_assignments.tsv", "cell_calls.tsv", "sex_call.json", "ancestry_call.json", "summary.json")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _owner_only(top):
    """Directories 0700 (setgid cleared), files 0600: $WORK is group project space and the genotype store is owner-only."""
    os.chmod(top, 0o700)
    for dp, dns, fns in os.walk(top):
        for d in dns:
            os.chmod(os.path.join(dp, d), 0o700)
        for f in fns:
            os.chmod(os.path.join(dp, f), 0o600)


def keep(out_dir, gse, gsm, base=None):
    """-> record {path, files, bytes, status}. Raises when a copy fails or reads back differently (the reads are gone afterwards)."""
    base = base or os.environ.get("SINGLET_GENOTYPE_DIR")
    if not base:
        if not os.path.isdir(os.path.dirname(os.path.dirname(DEFAULT_DIR))):   # not on Clipper (tests, laptops): nothing configured
            return {"path": None, "files": 0, "bytes": 0, "status": "skipped_no_genotype_dir"}
        base = DEFAULT_DIR
    if "/genotypes/" not in os.path.abspath(base).replace(os.sep, "/") + "/":
        raise RuntimeError("genotype_dir must be under $WORK/genotypes/: %s" % base)
    dest = os.path.join(base, gse or "NA", gsm)
    old = os.umask(0o077)
    try:
        for d in (base, os.path.dirname(dest)):
            os.makedirs(d, exist_ok=True)
            os.chmod(d, 0o700)
        tmp = dest + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        priv = os.path.join(out_dir, "private")
        have_snp = any(os.path.exists(os.path.join(priv, f)) for f in ("snp_dp.1pz", "snp_dp.mtx.gz"))
        for rel in PRIVATE:
            src = os.path.join(priv, rel)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(tmp, rel))
            elif os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp, rel))
        for rel in CONTEXT:
            src = os.path.join(out_dir, rel)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp, rel))
        _owner_only(tmp)   # copy2 keeps the source modes (0644): the store is owner-only
        files = {}
        for dp, _d, fns in os.walk(tmp):
            for fn in fns:
                p = os.path.join(dp, fn)
                rel = os.path.relpath(p, tmp).replace(os.sep, "/")
                src = os.path.join(priv if rel.split("/")[0] in PRIVATE else out_dir, rel)
                sha = _sha256(p)
                if sha != _sha256(src):
                    raise RuntimeError("genotype copy differs from the source: %s" % rel)
                files[rel] = {"bytes": os.path.getsize(p), "sha256": sha}
        man = {"gse": gse, "gsm": gsm, "status": "ok" if have_snp else "no_snp_data", "files": dict(sorted(files.items()))}
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1)
        shutil.rmtree(dest, ignore_errors=True)
        os.rename(tmp, dest)
        os.chmod(os.path.join(dest, "manifest.json"), 0o600)
    finally:
        os.umask(old)
    return {"path": dest, "files": len(files), "bytes": sum(v["bytes"] for v in files.values()), "status": man["status"]}
