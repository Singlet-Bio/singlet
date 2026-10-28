#!/bin/bash
# Install the network-tier binaries into $S1/bin (MinIO server and client from min.io, Toxiproxy from its GitHub
# releases), then smoke-test them: start MinIO behind Toxiproxy on loopback, put an object, read it back throttled.
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
[ -x minio ] || curl -fsSL -o minio https://dl.min.io/server/minio/release/linux-amd64/minio
[ -x mc ] || curl -fsSL -o mc https://dl.min.io/client/mc/release/linux-amd64/mc
if [ ! -x toxiproxy-server ]; then
  tag=$(curl -fsSL https://api.github.com/repos/Shopify/toxiproxy/releases/latest | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])")
  curl -fsSL -o toxiproxy-server "https://github.com/Shopify/toxiproxy/releases/download/$tag/toxiproxy-server-linux-amd64"
fi
chmod +x minio mc toxiproxy-server
./minio --version | head -1; ./mc --version | head -1; ./toxiproxy-server --version
s1_activate
cd "$BENCH"
python - <<'EOF'
import os, time, urllib.request
from pathlib import Path
from s1bench import netem
from s1bench.util import load_config

cfg = load_config()
work = Path(os.environ["TMPDIR"]) / "netem"
src = work / "blob.bin"
work.mkdir(parents=True, exist_ok=True)
src.write_bytes(os.urandom(64 << 20))
with netem.object_store(cfg, {"t/blob.bin": src}, work) as base:
    url = f"{base}/t/blob.bin"
    ceiling = netem.calibrate(cfg, url, 0)
    netem.set_tier(cfg, {"gbit": 1, "latency_ms": 1})
    t0 = time.perf_counter()
    n = len(urllib.request.urlopen(url).read())
    g = n * 8 / (time.perf_counter() - t0) / 1e9
    req = urllib.request.Request(url, headers={"Range": "bytes=100-199"})
    r = urllib.request.urlopen(req)
    print(f"netem ok: ceiling {ceiling:.2f} Gbit/s, throttled to 1 Gbit/s -> {g:.2f} Gbit/s, range status {r.status} len {len(r.read())}")
EOF
