"""Per-sample Beta publish for the drain worker (S9): one finished sample -> `.s1` objects -> uploaded -> study manifest extended.

    publish_sample(cfg, ccfg, gse, gsm, public_dir, scratch, dry_run=True, ...)

Reuses #97 as it is: `s1_producer.py` (-> `s1_publish.py` pack/migrate/verify, Beta policy) builds the sample's objects from a one-sample
hand-off, and the Beta uploader's own primitives (`Client`, `upload`) move the bytes. What is new is only the manifest bookkeeping,
because the uploader publishes the manifest it is given as THE study manifest (`s1/m/<GSE>.json`) and verifies that every object it
names is under --root:

  1. producer -> `<work>/<GSE>.manifest.json` with this sample alone (+ its objects);
  2. objects: upload what the bucket lacks (idempotent by SHA-256; R2 re-checks the hash);
  3. under a per-study lock: base = the LIVE manifest (`<data_base>/s1/m/<GSE>.json`, public read; nothing is kept on Clipper), merged =
     base samples with this sample replaced/added, `rev` recomputed exactly as `singlet.s1.manifest.compute_rev` (same bytes:
     "SGMREV1", sorted sample keys, study files), study files and metadata from the newest manifest;
  4. POST `s1-publish` with the merged manifest; the server checks that every named object is in the bucket, writes the manifest to R2 last,
     then the D1 rows. Re-running a sample reproduces the same keys and the same rev: nothing is written (idempotent). A crash between 2 and 4
     leaves orphan objects, never a manifest that names a missing object.

Safety is beta.py's: the target must be a Beta host/bucket/database; keys must be public (`s1/`), never `private/`, `.1fq`, `.bam`,
genotype; the uploader refuses `private/*`. `dry_run` runs the producer and the local verification and, with a token, read-only presence
checks; it never writes to Cloudflare.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beta as B  # noqa: E402

LABEL_RE = re.compile(r"s1/a/GSM\d+/[0-9a-f]{32}\.parquet")
LOCK_STALE_S = 900


def compute_rev(sample_keys, study_files):
    """Identical to python/singlet/s1/manifest.compute_rev (tests assert it)."""
    h = hashlib.sha256(b"SGMREV1\n")
    for k in sorted(sample_keys):
        h.update(k.encode("utf-8") + b"\n")
    for name in sorted(study_files):
        h.update(name.encode("utf-8") + b"\0" + study_files[name]["sha256"].encode("ascii") + b"\n")
    return h.hexdigest()[:16]


def merge_manifests(base, new):
    """Extend the live study manifest `base` (or None) with the one-sample manifest `new`."""
    if not base:
        m = json.loads(json.dumps(new))
    else:
        m = json.loads(json.dumps(new))  # newest study files, pipeline, packs, created_at
        by_gsm = {s["gsm"]: s for s in base.get("samples", [])}
        for s in new["samples"]:
            by_gsm[s["gsm"]] = s
        m["samples"] = [by_gsm[g] for g in sorted(by_gsm)]
        for k in ("legacy",):
            if k in base and k not in m:
                m[k] = base[k]
    m["rev"] = compute_rev([s["key"] for s in m["samples"]], m["study_files"])
    m["n_samples"] = len(m["samples"])
    return m


def fetch_live_manifest(data_base, gse, timeout=60):
    url = "%s/s1/m/%s.json?rb=%d" % (data_base.rstrip("/"), gse, int(time.time()))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "s9-drain"})
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


class StudyLock:
    """mkdir lock on the shared file system, one per study, so two workers cannot publish merged manifests that overwrite each other."""

    def __init__(self, root, gse, wait_s=900):
        self.path = os.path.join(root, "beta", "locks", gse)
        self.wait_s = wait_s

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        t0 = time.time()
        while True:
            try:
                os.mkdir(self.path)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > LOCK_STALE_S:
                        shutil.rmtree(self.path, ignore_errors=True)  # holder died (preempted)
                        continue
                except OSError:
                    continue
                if time.time() - t0 > self.wait_s:
                    raise TimeoutError("study lock %s busy" % self.path)
                time.sleep(2)

    def __exit__(self, *a):
        shutil.rmtree(self.path, ignore_errors=True)


def producer_cmd(b, gse, handoff, work, s1_repo, scratch, runtime):
    cmd = [x.format(gse=gse, handoff=handoff, work=work, here=os.path.dirname(os.path.abspath(__file__)), catalog=b.get("catalog") or "")
           for x in b["producer"]]
    if s1_repo:
        cmd += ["--repo", s1_repo]
    cmd += ["--scratch", scratch]
    if runtime:  # Clipper: the producer needs numpy/pyarrow and the compiled codec, which live in the runtime image + venv
        # $0 must not be the sourced script: in_runtime.sh runs "$@" itself when BASH_SOURCE[0] == $0, which would run the producer twice
        cmd = ["bash", "-c", 'source "$1" && shift && in_runtime "$@"', "_", runtime] + cmd
    return cmd


def publish_sample(cfg, ccfg, gse, gsm, public_dir, scratch, dry_run=True, s1_repo=None, runtime=None, env=None, log=print):
    """-> dict(status = published | dry_run_ok | failed | refused, ...). Never raises for an upload problem (the caller retries)."""
    env = os.environ if env is None else env
    b = dict(cfg.get("beta", {}))
    try:
        base = B.check_target(b)
    except B.BetaRefused as e:
        return {"gse": gse, "gsm": gsm, "status": "refused", "detail": str(e)}
    work = os.path.join(scratch, "beta", gse)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    handoff = os.path.join(scratch, "beta", gse + ".handoff.json")
    json.dump({"schema": "singlet.atlas_rerun.handoff/1", "gse": gse, "samples": [
        {"gsm": gsm, "state": "done", "out_dir": os.path.abspath(public_dir)}]}, open(handoff, "w"))
    cmd = producer_cmd(b, gse, handoff, work, s1_repo, os.path.join(scratch, "s1prod"), runtime)
    os.makedirs(os.path.join(scratch, "s1prod"), exist_ok=True)
    log("[publish] producer: %s" % " ".join(cmd[:6] + ["..."]))
    rc = subprocess.run(cmd, env=env).returncode
    mpath = os.path.join(work, gse + ".manifest.json")
    if rc != 0 or not os.path.exists(mpath):
        return {"gse": gse, "gsm": gsm, "status": "failed", "detail": "producer rc=%s" % rc}
    try:
        up = B.load_uploader(os.path.expandvars(b["uploader"]))
        new = json.load(open(mpath, encoding="utf-8"))
        checked = B.verify_local(up, new, work)  # size, SHA-256 and key digest of this sample's objects
        base_m = fetch_live_manifest(b.get("data_base", "https://beta-data.singlet.bio"), gse)
        merged = merge_manifests(base_m, new)
        B.check_manifest_keys(merged)
    except B.BetaRefused as e:
        return {"gse": gse, "gsm": gsm, "status": "refused", "detail": str(e)}
    token = env.get(b.get("token_env", "SINGLET_INGEST_TOKEN"), "")
    res = {"gse": gse, "gsm": gsm, "objects": len(checked), "study_samples": merged["n_samples"], "rev": merged["rev"],
           "rev_was": (base_m or {}).get("rev"), "base": base}
    if dry_run:
        present = None
        if token:  # read-only presence checks
            c = up.Client(base, token)
            present = sum(int(c.call("GET", "s1-object", {"key": k})[0] == 200) for k, _ in checked)
        return {**res, "status": "dry_run_ok", "present": present}
    if not token:
        return {**res, "status": "failed", "detail": "%s is not set" % b.get("token_env", "SINGLET_INGEST_TOKEN")}
    c = up.Client(base, token)
    t0 = time.time()
    sent, counts = 0, {"present": 0, "unchanged": 0, "uploaded": 0}
    last = ""
    for i in range(int(b.get("retries", 3))):
        try:
            for key, sha in checked:
                n, word = up.upload(c, key, os.path.join(work, *key.split("/")), sha)
                sent += n
                counts[word] += 1
            with StudyLock(ccfg["root"], gse):
                # re-read the base under the lock: another worker may have published a sibling sample meanwhile
                merged = merge_manifests(fetch_live_manifest(b.get("data_base", "https://beta-data.singlet.bio"), gse), new)
                st, d = c.call("POST", "s1-publish", None, json.dumps({"manifest": merged, "force": False}).encode(),
                               {"Content-Type": "application/json"})
            if st != 200:
                raise RuntimeError("publish %s %s" % (st, d))
            res.update(counts, bytes_sent=sent, rev=d.get("rev"), rev_unchanged=d.get("unchanged"), d1_rows_written=d.get("rows_written"),
                       upload_s=round(time.time() - t0, 1), study_samples=merged["n_samples"])
            ledger = os.path.join(ccfg["root"], "beta", "ingest.jsonl")
            os.makedirs(os.path.dirname(ledger), exist_ok=True)
            with open(ledger, "a") as f:
                f.write(json.dumps({**res, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
            return {**res, "status": "published"}
        except (RuntimeError, OSError, TimeoutError) as e:
            last = repr(e)
            time.sleep(float(b.get("retry_wait_s", 30)) * (i + 1))
    return {**res, "status": "failed", "detail": last}
