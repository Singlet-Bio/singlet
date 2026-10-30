"""Publish a finished study of the atlas re-run to the Research Beta stack (beta.singlet.bio), through the Beta uploader.

    orchestrator.py beta-publish --cluster clipper --gse GSE123 [--dry-run]

Two steps, both configurable under `beta` in config.json:

  1. producer: turns the study's hand-off (`handoff/<GSE>.json`: the samples' `out_dir`s, never `private_dir`) into the
     deployment layout under a work directory: `<work>/<GSE>.manifest.json` and the objects it names
     (`<work>/s1/s/<GSM>/<d16>.s1`, `<work>/s1/a/<GSM>/<d16>.parquet`). The command is `beta.producer` with `{gse}`,
     `{handoff}`, `{work}` filled in. The default is `s1_producer.py`, which reuses scripts/s1_publish.py (pack_gse, migrate, verify);
     with no producer configured the study is reported as `no_producer` and nothing is uploaded.
  2. uploader: `beta_ingest.py` of the website repo (PR singlet-website#109): verifies the objects against the manifest,
     uploads what the bucket lacks, publishes the manifest and the D1 rows. It is the only writer of Beta's D1. It is
     idempotent, so a failed or repeated call is simply run again (`beta.retries` tries with back-off).

Safety. The target must name a Beta stack: the site host, the bucket and the database all contain "beta", and none is a
production name. Anything else is refused before a byte is read. Genotypes and reads never leave: the producer gets the
hand-off, the uploader refuses `private/*` keys, and a manifest that names one is refused here as well. A dry run does
everything except the writes: the producer, the local verification of every object against the manifest, and (with a
token) read-only presence checks against the bucket.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from urllib.parse import urlparse

import common as C

PRODUCTION_HOSTS = {"singlet.bio", "www.singlet.bio", "data.singlet.bio", "api.singlet.bio"}
PRODUCTION_NAMES = {"singlet-data", "singlet-catalog", "singlet-user-data", "singlet-hpc-dashboard", "singlet-restricted"}
PRIVATE_KEY = re.compile(r"(^|/)private/|restricted/|\.vcf(\.gz)?$|\.bam$|\.1fq$|genotype", re.I)


class BetaRefused(Exception):
    pass


def _is_beta_name(name):
    return bool(name) and "beta" in name.lower() and name.lower() not in PRODUCTION_NAMES


def check_target(b, base=None):
    """Refuse anything that is not the Beta stack. Returns the base URL to use."""
    base = (base or b.get("base") or "").rstrip("/")
    host = (urlparse(base).hostname or "").lower()
    if host in PRODUCTION_HOSTS or "beta" not in host:
        raise BetaRefused("site %r is not a Beta host (the host must contain 'beta' and not be production)" % host)
    for k in ("bucket", "database"):
        if not _is_beta_name(b.get(k)):
            raise BetaRefused("%s %r does not name a Beta resource (it must contain 'beta' and not be production)"
                              % (k, b.get(k)))
    return base


def load_uploader(path):
    spec = importlib.util.spec_from_file_location("beta_ingest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_manifest_keys(manifest):
    """Every key the manifest names must be a public sample or label object."""
    keys = [s.get("key", "") for s in manifest.get("samples", [])]
    keys += sorted(set(re.findall(r"s1/a/GSM\d+/[0-9a-f]{32}\.parquet", json.dumps(manifest))))
    bad = [k for k in keys if PRIVATE_KEY.search(k) or not k.startswith("s1/")]
    if bad:
        raise BetaRefused("manifest names non-public keys: %s" % bad[:5])
    return keys


def verify_local(up, manifest, work):
    """The uploader's step 1 (size, SHA-256, key digest), without a network. Returns [(key, sha256)]."""
    out = []
    for key in check_manifest_keys(manifest):
        s = next((x for x in manifest["samples"] if x.get("key") == key), {})
        path = os.path.join(work, *key.split("/"))
        if not os.path.isfile(path):
            raise BetaRefused("%s: not found under the work directory" % key)
        got = up.sha256_file(path)
        m = up.DIGEST_RE.search(key)
        if (s.get("bytes") is not None and os.path.getsize(path) != s["bytes"]) or \
                (s.get("sha256") and got != s["sha256"]) or (m and not got.startswith(m.group(1))):
            raise BetaRefused("%s: local file does not match the manifest or the key digest" % key)
        out.append((key, got))
    return out


def run_producer(b, gse, handoff, work, log=print):
    cmd = b.get("producer")
    if not cmd:
        return None
    cmd = [x.format(gse=gse, handoff=handoff, work=work, here=C.HERE, catalog=b.get("catalog") or "") for x in cmd]
    log("[beta] producer: %s" % " ".join(cmd))
    return subprocess.run(cmd).returncode


def publish_study(cfg, ccfg, gse, dry_run=False, log=print, env=None, sleep=time.sleep):
    """Publish one handed-off study. Returns a result dict: status is published | dry_run_ok | no_producer | failed |
    refused, with `detail`. Never raises for an upload problem (the caller records the event and retries later)."""
    b = dict(cfg.get("beta", {}))
    try:
        base = check_target(b)
    except BetaRefused as e:
        return {"gse": gse, "status": "refused", "detail": str(e)}
    root = ccfg["root"]
    handoff = os.path.join(root, "handoff", gse + ".json")
    if not os.path.exists(handoff):
        return {"gse": gse, "status": "failed", "detail": "no hand-off %s" % handoff}
    work = os.path.join(os.path.expandvars(b.get("work_root") or os.path.join(root, "beta")), gse)
    os.makedirs(work, exist_ok=True)
    rc = run_producer(b, gse, handoff, work, log)
    if rc is None:
        return {"gse": gse, "status": "no_producer", "detail": "beta.producer is not configured (the .s1 producer, W5)"}
    mpath = os.path.join(work, gse + ".manifest.json")
    if rc != 0 or not os.path.exists(mpath):
        return {"gse": gse, "status": "failed", "detail": "producer rc=%s, manifest %s" % (rc, os.path.exists(mpath))}
    try:
        up = load_uploader(os.path.expandvars(b["uploader"]))
        with open(mpath, encoding="utf-8") as fh:
            manifest = json.load(fh)
        checked = verify_local(up, manifest, work)
    except BetaRefused as e:
        return {"gse": gse, "status": "refused", "detail": str(e)}
    env = env if env is not None else os.environ
    token = env.get(b.get("token_env", "SINGLET_INGEST_TOKEN"), "")
    res = {"gse": gse, "samples": len(manifest.get("samples", [])), "objects": len(checked), "base": base}
    if dry_run:
        present = None
        if token:  # read-only: which objects does the bucket already hold
            c = up.Client(base, token)
            present = 0
            for key, sha in checked:
                st, d = c.call("GET", "s1-object", {"key": key})
                present += int(st == 200 and d.get("sha256") == sha)
        return {**res, "status": "dry_run_ok", "present": present, "to_upload": None if present is None
                else len(checked) - present}
    if not token:
        return {**res, "status": "failed", "detail": "%s is not set" % b.get("token_env", "SINGLET_INGEST_TOKEN")}
    tries, wait = int(b.get("retries", 3)), float(b.get("retry_wait_s", 30))
    ledger = os.path.join(root, "beta", "ingest.jsonl")
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    last = ""
    for i in range(tries):
        cmd = [sys.executable, os.path.expandvars(b["uploader"]), "--manifest", mpath, "--root", work, "--base", base,
               "--ledger", ledger]
        p = subprocess.run(cmd, env={**os.environ, "SINGLET_INGEST_TOKEN": token}, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True)
        last = p.stdout[-1500:]
        if p.returncode == 0:
            try:
                res.update(json.loads(p.stdout.strip().splitlines()[-1]))
            except ValueError:
                pass
            return {**res, "status": "published"}
        if p.returncode == 2:  # usage error: a retry cannot help
            break
        if i + 1 < tries:
            sleep(wait * (i + 1))
    return {**res, "status": "failed", "detail": last}
