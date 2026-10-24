#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Atlas-wide per-cell annotation: one Slurm array task annotates a shard of studies (see annotate_campaign.sbatch).

  plan    write the study list with sizes and a balanced shard number per study (default list: https://singlet.bio/api/manifest,
          human and mouse studies only)
          python scripts/annotate_campaign.py plan --shards 400 --out plan.tsv
  run     annotate the studies of one shard
          python scripts/annotate_campaign.py run --plan plan.tsv --shard $SLURM_ARRAY_TASK_ID --out $OUT --af af.parquet \
              --scratch /tmp/$USER-$SLURM_JOB_ID --ledger $OUT/ledger/shard-$SLURM_ARRAY_TASK_ID.jsonl
  status  summarise the ledgers
          python scripts/annotate_campaign.py status --out $OUT

``studies.tsv``: columns gse, bytes (legacy bundle size; 0 when unknown), v2 (1 if a v2 manifest exists). Build it from the catalog
(D1 ``gse`` / ``v2_study``) or from ``https://singlet.bio/api/manifest``; the planner balances shards by bytes.

Per study: a v2 study is read through the v2 client from data.singlet.bio (range requests, nothing staged); a legacy study is
downloaded to node-local scratch, annotated, and deleted. When R2 credentials are present (``R2_ACCOUNT_ID``,
``R2_ACCESS_KEY_ID``, ``R2_SECRET_ACCESS_KEY``) the study's restricted sidecar (``singlet-restricted`` bucket,
``restricted/<GSE>/<GSE>.restricted.singlet``) is downloaded to scratch too, read for the per-donor SNP depths, and deleted with
the bundle; nothing from it is written anywhere (the label files carry only labels, checked by
``singlet.celllabels.output.assert_public_safe``). This script never writes to R2 or D1: the cell index ingests ``$OUT``.

A study whose samples all have outputs at the current model version is skipped, so a failed or pre-empted task can simply be
re-run. Each finished study appends one JSON line to the shard's ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

DATA_BASE = os.environ.get("SINGLET_DATA_BASE", "https://data.singlet.bio")
RESTRICTED_BUCKET = os.environ.get("SINGLET_RESTRICTED_BUCKET", "singlet-restricted")
# data.singlet.bio answers 403 to the default Python-urllib User-Agent (as the v2 transport notes)
UA = {"User-Agent": "singlet-annotate/0.1 (+https://singlet.bio)"}


MANIFEST_URL = "https://singlet.bio/api/manifest"
SPECIES_WITH_MODELS = ("human", "homo sapiens", "mouse", "mus musculus")


def studies_from_manifest(url: str = MANIFEST_URL):
    """The catalogue's study list (``/api/manifest``, one TSV row per published study) as gse, bytes, v2, organism,
    n_cells; only studies with a human or mouse sample (the organisms the models cover)."""
    import io

    import pandas as pd

    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        m = pd.read_csv(io.BytesIO(r.read()), sep="\t", dtype=str)
    org = m["organism"].fillna("").str.lower()
    m = m[org.apply(lambda o: any(k in o for k in SPECIES_WITH_MODELS))]
    return pd.DataFrame(
        {
            "gse": m["gse_id"],
            "bytes": pd.to_numeric(m["bytes"], errors="coerce").fillna(0).astype("int64"),
            "v2": m["v2_manifest_url"].fillna("").str.len().gt(0).astype(int),
            "organism": m["organism"],
            "n_cells": pd.to_numeric(m["n_cells"], errors="coerce").fillna(0).astype("int64"),
        }
    )


def plan(a) -> int:
    import pandas as pd

    df = studies_from_manifest() if a.studies is None else pd.read_csv(a.studies, sep="\t")
    df["bytes"] = df["bytes"].fillna(0).astype("int64")
    df["v2"] = df.get("v2", 0)
    df = df.sort_values("bytes", ascending=False).reset_index(drop=True)
    load = [0] * a.shards
    shard = []
    for b in df["bytes"]:  # greedy: largest study to the lightest shard
        i = min(range(a.shards), key=load.__getitem__)
        shard.append(i)
        load[i] += max(int(b), 50 << 20)
    df["shard"] = shard
    df.to_csv(a.out, sep="\t", index=False)
    print(
        f"{len(df)} studies, {df['bytes'].sum() / 1e12:.2f} TB, {a.shards} shards, max shard {max(load) / 1e9:.1f} GB"
    )
    return 0


def _download(url: str, dst: Path, retries: int = 4) -> None:
    for i in range(retries):
        try:
            tmp = dst.with_name(dst.name + ".part")
            with (
                urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r,
                open(tmp, "wb") as f,
            ):
                want = int(r.headers.get("Content-Length") or -1)
                shutil.copyfileobj(r, f, 16 << 20)
            if (
                want >= 0 and tmp.stat().st_size != want
            ):  # a dropped connection can end the stream early without an error
                raise OSError(f"{url}: got {tmp.stat().st_size} of {want} bytes")
            tmp.replace(dst)
            return
        except OSError as e:
            if i == retries - 1 or getattr(e, "code", None) in (403, 404):
                raise
            time.sleep(10 * (i + 1))


def _sidecar(gse: str, dst: Path) -> bool:
    """Download the restricted sidecar of ``gse`` when credentials are present; False when there is none."""
    if not all(
        os.environ.get(k) for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    ):
        return False
    import boto3

    c = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    key = f"restricted/{gse}/{gse}.restricted.singlet"
    try:
        c.head_object(Bucket=RESTRICTED_BUCKET, Key=key)
    except Exception:  # noqa: BLE001  (404 or no access: no ancestry for this study)
        return False
    c.download_file(RESTRICTED_BUCKET, key, str(dst))
    return True


def _has_v2(gse: str) -> bool:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{DATA_BASE}/v2/m/{gse}.json", method="HEAD", headers=UA),
            timeout=30,
        ) as r:
            return r.status == 200
    except OSError:
        return False


def run(a) -> int:
    import pandas as pd

    df = pd.read_csv(a.plan, sep="\t")
    mine = df[df["shard"] == a.shard]
    a.out.mkdir(parents=True, exist_ok=True)
    a.ledger.parent.mkdir(parents=True, exist_ok=True)
    print(f"shard {a.shard}: {len(mine)} studies, {mine['bytes'].sum() / 1e9:.1f} GB", flush=True)
    bad = 0
    for _, row in mine.iterrows():
        gse = row["gse"]
        t0 = time.time()
        rec = dict(gse=gse, shard=a.shard, host=os.uname().nodename if hasattr(os, "uname") else "")
        work = a.scratch / gse
        work.mkdir(parents=True, exist_ok=True)
        try:
            cmd = [
                sys.executable,
                str(HERE / "annotate_sample.py"),
                "--gse",
                gse,
                "--out",
                str(a.out),
                "--method",
                a.method,
            ]
            if a.af:
                cmd += ["--af", str(a.af)]
            v2 = bool(row.get("v2", 0)) or _has_v2(gse)
            rec["source"] = "v2" if v2 else "legacy"
            if not v2:
                bundle = work / f"{gse}.singlet"
                _download(f"{DATA_BASE}/data/{gse}/{gse}.singlet", bundle)
                cmd += ["--bundle", str(bundle)]
            side = work / f"{gse}.restricted.singlet"
            if a.af and _sidecar(gse, side):
                cmd += ["--sidecar", str(side)]
                rec["sidecar"] = True
            p = subprocess.run(cmd, capture_output=True, text=True)
            (a.out / gse).mkdir(parents=True, exist_ok=True)
            (a.out / gse / "annotate.log").write_text(
                p.stdout[-200_000:] + "\n--- stderr ---\n" + p.stderr[-50_000:]
            )
            rec["status"] = "ok" if p.returncode == 0 else f"exit {p.returncode}"
            rec["n_written"] = len(list((a.out / gse).glob("*.labels.parquet")))
            rec["n_errors"] = len(list((a.out / gse).glob("*.error.txt")))
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"error {type(e).__name__}: {e}"
        finally:
            shutil.rmtree(
                work, ignore_errors=True
            )  # the bundle and the restricted sidecar never outlive the study
        rec["seconds"] = round(time.time() - t0, 1)
        bad += rec["status"] != "ok"
        with open(a.ledger, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    return 1 if bad else 0


def status(a) -> int:
    recs = []
    # oldest ledger first, so a retry (written later) is the record that counts for its study
    for p in sorted((a.out / "ledger").glob("*.jsonl"), key=lambda q: q.stat().st_mtime):
        recs += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    last = {r["gse"]: r for r in recs}
    ok = sum(r["status"] == "ok" for r in last.values())
    print(
        f"{len(last)} studies attempted, {ok} ok, {len(last) - ok} not ok; "
        f"{sum(r.get('n_written', 0) for r in last.values())} sample files; {sum(r['seconds'] for r in last.values()) / 3600:.1f} task-hours"
    )
    for r in list(last.values()):
        if r["status"] != "ok":
            print(" ", r["gse"], r["status"][:120])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument(
        "--studies", type=Path, help="TSV gse, bytes[, v2] (default: the catalogue's /api/manifest)"
    )
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--shard", type=int, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--af", type=Path)
    r.add_argument("--method", default="guide")
    r.add_argument(
        "--scratch", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "annotate"
    )
    r.add_argument("--ledger", type=Path, required=True)
    s = sub.add_parser("status")
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return {"plan": plan, "run": run, "status": status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
