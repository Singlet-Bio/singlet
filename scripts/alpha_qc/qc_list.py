# SPDX-License-Identifier: MIT
"""Research Alpha QC, step 0: list the public .s1 prefixes of bucket singlet-data (read-only S3 ListObjectsV2).

Writes one TSV per prefix (key, size, etag, last_modified), gzip-compressed, into OUT. Uses only list calls; never writes to R2.

    python qc_list.py OUT [prefix ...]          (default prefixes: s1/m/ s1/s/ s1/ref/ s1/a/ s1/cells/ s1/ledger/ s1/cidx/ v2/)

Credentials: R2_ACCOUNT_ID (or R2_ENDPOINT), R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY in the environment.
"""

import gzip
import os
import sys
import time

import boto3
from botocore.config import Config

BUCKET = "singlet-data"
DEFAULT = ["s1/m/", "s1/s/", "s1/ref/", "s1/a/", "s1/cells/", "s1/ledger/", "s1/cidx/", "v2/"]


def client():
    ep = os.environ.get("R2_ENDPOINT") or f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=ep,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )


def main():
    out = sys.argv[1]
    prefixes = sys.argv[2:] or DEFAULT
    os.makedirs(out, exist_ok=True)
    s3 = client()
    pg = s3.get_paginator("list_objects_v2")
    for pre in prefixes:
        t0 = time.time()
        n = b = 0
        fn = os.path.join(out, "list_" + pre.strip("/").replace("/", "_") + ".tsv.gz")
        with gzip.open(fn, "wt") as fh:
            fh.write("key\tsize\tetag\tlast_modified\n")
            for page in pg.paginate(Bucket=BUCKET, Prefix=pre, PaginationConfig={"PageSize": 1000}):
                for o in page.get("Contents", []):
                    fh.write(f"{o['Key']}\t{o['Size']}\t{o['ETag'].strip(chr(34))}\t{o['LastModified'].isoformat()}\n")
                    n += 1
                    b += o["Size"]
        print(f"{pre}\t{n} objects\t{b} bytes\t{time.time() - t0:.1f}s -> {fn}", flush=True)


if __name__ == "__main__":
    main()
