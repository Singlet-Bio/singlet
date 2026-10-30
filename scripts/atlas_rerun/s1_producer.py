"""Producer for `beta.producer`: a handed-off study (orchestrator outputs) -> `.s1` objects + manifest in the deployment layout.

    s1_producer.py GSE handoff.json WORK --catalog processing_catalog.parquet [--repo DIR] [--python python3]

No new transcoder. It reuses the Alpha v2 route, `scripts/s1_publish.py` (pack_gse -> singlet.s1.migrate -> singlet.s1.verify),
pointed at a local directory instead of R2 (`--local-root`), so nothing leaves the machine and verify.py must pass before
anything is copied out:

  1. results tree: `<tmp>/<GSM>/out -> <out_dir>` for every `done` sample of the hand-off (never `private_dir`);
  2. s1_publish.py --apply --local-root <tmp>/lr --public beta-local (no --private: restricted members are not packed);
  3. WORK/s1/{m,s,a}/... and WORK/<GSE>.manifest.json copied from <tmp>/lr/beta-local/s1, with `n_samples` set (the Beta
     publish endpoint requires it).

The cost of the detour is one legacy-format bundle per study in scratch (deleted afterwards): the pipeline's .1pz members are
copied into it and then transcoded, so a study is read and written about twice. The orchestrator's hand-off packs a whole
study, so call it per study (or every k samples), not per sample.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

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
    ap.add_argument("--policy", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "beta-policy.json"),
                    help="release policy; the default withholds exactly what Alpha's does (singlet.s1.policy.OMITTED_LAYERS)")
    a = ap.parse_args(argv)
    tmp = tempfile.mkdtemp(prefix="s1prod-", dir=a.scratch)
    try:
        results = os.path.join(tmp, "results")
        os.makedirs(results)
        n = build_results(a.handoff, results)
        lr = os.path.join(tmp, "lr")
        cmd = [a.python, os.path.join(a.repo, "scripts", "s1_publish.py"), a.gse, "--results", results, "--catalog", a.catalog,
               "--apply", "--policy", a.policy, "--local-root", lr, "--public", BUCKET, "--scratch", os.path.join(tmp, "scratch")]
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
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
