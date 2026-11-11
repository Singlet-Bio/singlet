"""Producer for `beta.producer`: a handed-off study (orchestrator outputs) -> `.s1` objects + manifest in the deployment layout.

    s1_producer.py GSE handoff.json WORK --catalog processing_catalog.parquet [--repo DIR] [--python python3]

No new transcoder. It reuses the Alpha v2 route, `scripts/s1_publish.py` (pack_gse -> singlet.s1.migrate -> singlet.s1.verify),
pointed at a local directory instead of R2 (`--local-root`), so nothing leaves the machine and verify.py must pass before
anything is copied out:

  1. results tree: `<tmp>/<GSM>/out -> <out_dir>` for every `done` sample of the hand-off (never `private_dir`);
  2. s1_publish.py --apply --local-root <tmp>/lr --public beta-local (no --private: restricted members are not packed);
  3. WORK/s1/{m,s,a}/... and WORK/<GSE>.manifest.json copied from <tmp>/lr/beta-local/s1, with `n_samples` set (the Beta
     publish endpoint requires it).

Per-cell labels (cell type, chromosomal sex, ancestry) are computed by the transcoder, as the Alpha campaign did (`--labels`, on by
default; `--af` for ancestry). Ancestry needs each donor's depth table, which the pipeline keeps out of the public output: the producer
copies only `donor/donor<k>_coverage.tsv` from each sample's private directory (`private_dir`, else the hand-off's
`private.root`/<GSM>, else `<out_dir>/private`) into a zip in its own scratch
directory, passes it to s1_publish.py as `--restricted-sidecar` (read in memory by the labeller, never packed, uploaded or copied to
WORK) and deletes it before returning, even with `--keep`.

The cost of the detour is one legacy-format bundle per study in scratch (deleted afterwards): the pipeline's .1pz members are
copied into it and then transcoded, so a study is read and written about twice. The orchestrator's hand-off packs a whole
study, so call it per study (or every k samples), not per sample.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

BUCKET = "beta-local"


def build_results(handoff, results):
    with open(handoff, encoding="utf-8") as fh:
        man = json.load(fh)
    n = 0
    for s in man.get("samples", []):
        if s.get("state") != "done" or not s.get("out_dir"):
            continue
        d = os.path.join(results, s["gsm"])
        os.makedirs(d)
        os.symlink(os.path.abspath(s["out_dir"]), os.path.join(d, "out"))
        n += 1
    if not n:
        raise SystemExit("no done sample in %s" % handoff)
    return n


_COVERAGE = re.compile(r"^donor\d+_coverage\.tsv$")


def depth_sidecar(handoff, path):
    """The per-donor depth tables of the hand-off's done samples -> a zip at `path` (`samples/<GSM>/donor/donor<k>_coverage.tsv`, the
    layout `singlet.celllabels.io.read_donor_sites` reads). Only coverage tables: no VCF, SNP matrix or other private file. Returns the
    number of tables, 0 (and no file) when there are none."""
    with open(handoff, encoding="utf-8") as fh:
        man = json.load(fh)
    found = []
    proot = (man.get("private") or {}).get("root")  # the orchestrator's <private_root>/<GSE>; <GSM>/donor/ below it
    for s in man.get("samples", []):
        if s.get("state") != "done":
            continue
        roots = (s.get("private_dir"), os.path.join(proot, s["gsm"]) if proot else None,
                 os.path.join(s["out_dir"], "private") if s.get("out_dir") else None)
        for root in roots:
            d = os.path.join(root, "donor") if root else None
            if d and os.path.isdir(d):
                found += [(s["gsm"], os.path.join(d, f), f) for f in sorted(os.listdir(d)) if _COVERAGE.match(f)]
                break
    if not found:
        return 0
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as raw, zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as z:
        for gsm, src, name in found:
            z.write(src, "samples/%s/donor/%s" % (gsm, name))
    return len(found)


def copy_store(store_root, gse, work):
    """The manifest and every object it names (and nothing else) -> WORK in the layout beta_ingest.py expects."""
    mpath = os.path.join(store_root, "m", gse + ".json")
    with open(mpath, encoding="utf-8") as fh:
        text = fh.read()
    manifest = json.loads(text)
    manifest["n_samples"] = len(manifest["samples"])
    hollow = [x.get("gsm") for x in manifest["samples"] if not x.get("n_columns") or not x.get("n_called") or not x.get("stats")]
    if hollow:
        raise SystemExit("hollow sample object(s), not published: %s" % hollow)
    import re

    keys = [s["key"] for s in manifest["samples"]] + sorted(set(re.findall(r"s1/a/GSM\d+/[0-9a-f]{32}\.parquet", text)))
    for k in keys:
        src = os.path.join(store_root, *k.split("/")[1:])  # store_root is <bucket>/s1; keys start with s1/
        dst = os.path.join(work, *k.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    with open(os.path.join(work, gse + ".manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return len(keys)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gse")
    ap.add_argument("handoff")
    ap.add_argument("work")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--repo", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--scratch")
    ap.add_argument("--keep", action="store_true", help="keep the scratch tree (study.log of the transcoder) for debugging")
    ap.add_argument("--policy", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "beta-policy.json"),
                    help="release policy; the default withholds exactly what Alpha's does (singlet.s1.policy.OMITTED_LAYERS)")
    ap.add_argument("--labels", action=argparse.BooleanOptionalAction, default=True,
                    help="per-cell labels computed by the transcoder (default on, as in the Alpha campaign)")
    ap.add_argument("--af", default=os.environ.get("SINGLET_LABELS_AF"),
                    help="1000 Genomes super-population allele frequencies (parquet) for the ancestry labels (default $SINGLET_LABELS_AF)")
    a = ap.parse_args(argv)
    tmp = tempfile.mkdtemp(prefix="s1prod-", dir=a.scratch)
    try:
        results = os.path.join(tmp, "results")
        os.makedirs(results)
        n = build_results(a.handoff, results)
        lr = os.path.join(tmp, "lr")
        cmd = [a.python, os.path.join(a.repo, "scripts", "s1_publish.py"), a.gse, "--results", results, "--catalog", a.catalog,
               "--apply", "--policy", a.policy, "--local-root", lr, "--public", BUCKET, "--scratch", os.path.join(tmp, "scratch")] + (["--keep-scratch", "--study-log-dir", os.path.join(tmp, "logs")] if a.keep else [])
        if a.labels:
            cmd += ["--labels"] + (["--af", a.af] if a.af else [])
            depth = os.path.join(tmp, "private-depth", a.gse + ".depth.zip")  # outside lr/: never part of the store
            nd = depth_sidecar(a.handoff, depth)
            if nd:
                cmd += ["--restricted-sidecar", depth]
            print("s1_producer: labels on, %d depth table(s) for ancestry%s" % (nd, "" if a.af else ", no --af: no ancestry"))
        env = dict(os.environ, PYTHONPATH=os.path.join(a.repo, "python") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        if subprocess.run([a.python, "-c", "import singlet._pz, singlet._pz2"], env=env).returncode != 0:
            # without the compiled reader the migrator treats every .1pz as undecodable and stores a hollow object
            print("%s has no compiled singlet._pz/_pz2: build it first (bench/atlas_rerun/build_s1_repo.sbatch)" % a.repo,
                  file=sys.stderr)
            return 5
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            print("s1_publish.py exited %d (verify must pass)" % rc, file=sys.stderr)
            return rc
        k = copy_store(os.path.join(lr, BUCKET, "s1"), a.gse, a.work)
        print("s1_producer: %d samples, %d objects -> %s" % (n, k, a.work))
        return 0
    finally:
        shutil.rmtree(os.path.join(tmp, "private-depth"), ignore_errors=True)  # the depth tables never outlive the run, even with --keep
        if a.keep:
            print("s1_producer: scratch kept at %s" % tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
