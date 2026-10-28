#!/bin/bash
# Install Toxiproxy into $S1/bin (from its GitHub releases) and smoke-test the network tier: `rclone serve s3` (the
# Clipper rclone module) behind Toxiproxy on loopback, a throttled GET, a range GET and an unsigned S3 listing (what
# TileDB-SOMA needs). MinIO no longer publishes community binaries (dl.min.io answers 410), so rclone is the store.
#SBATCH --job-name=s1-netem
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/logs/%x-%j.out
set -euo pipefail
source "/mnt/projects/debruinz_project/debruinz/singlet/s1paper/src/${S1_SHA:?set S1_SHA}/papers/manuscripts/s1-format/bench/slurm/clipper/common.sh"
s1_checkout
mkdir -p "$S1/bin"
cd "$S1/bin"
if [ ! -x toxiproxy-server ]; then
  tag=$(curl -fsSL https://api.github.com/repos/Shopify/toxiproxy/releases/latest | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])")
  curl -fsSL -o toxiproxy-server "https://github.com/Shopify/toxiproxy/releases/download/$tag/toxiproxy-server-linux-amd64"
  chmod +x toxiproxy-server
fi
./toxiproxy-server --version
s1_activate
s1_netem_tools
rclone version | head -1
cd "$BENCH"
python - <<'EOF'
import os, time, urllib.request
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig

from s1bench import netem
from s1bench.util import load_config

cfg = load_config()
work = Path(os.environ["TMPDIR"]) / "netem"
src = work / "blob.bin"
work.mkdir(parents=True, exist_ok=True)
src.write_bytes(os.urandom(256 << 20))
with netem.object_store(cfg, {"t/blob.bin": src}, work) as base:
    url = f"{base}/t/blob.bin"
    ceiling = netem.calibrate(cfg, url, 0)
    netem.set_tier(cfg, {"gbit": 1, "latency_ms": 1})
    t0 = time.perf_counter()
    n = len(urllib.request.urlopen(url).read())
    g = n * 8 / (time.perf_counter() - t0) / 1e9
    r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=100-199"}))
    endpoint = base.rsplit("/", 1)[0]
    s3 = boto3.client("s3", endpoint_url=endpoint, config=BotoConfig(signature_version=UNSIGNED), region_name="us-east-1")
    keys = [o["Key"] for o in s3.list_objects_v2(Bucket=cfg["netem"]["bucket"], Prefix="t/").get("Contents", [])]
    print(f"netem ok: ceiling {ceiling:.2f} Gbit/s; throttled to 1 Gbit/s -> {g:.2f} Gbit/s; range {r.status} "
          f"len {len(r.read())}; unsigned list {keys}")
EOF
