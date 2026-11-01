#!/bin/bash
#SBATCH --job-name=s14-multirun-proto
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Do the runs of one multi-run sample get different protocols from the decode gate's detector? A random sample of N multi-run GSMs (>= 2 runs, <= MAXRUNS runs
# probed each); for every run the first 5000 spots are read from NCBI (fastq-dump -X) and detect_protocol (bench/protocol_geometry/detect_fastq.cpp, real whitelists)
# is run on the two longest reads. A sample "disagrees" when its runs get different tags. Env: N (default 150), MAXRUNS (default 6), SEED, BRANCH.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/bpub/bin:$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
T=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
git clone -q --depth 1 -b ${BRANCH:-fix/r1-untrimmed-protocol} git@github.com:Singlet-Bio/singlet.git $T/repo 2>&1 | tail -1
$W/envs/pbench/bin/x86_64-conda-linux-gnu-g++ -O1 -std=c++17 -pthread -I$T/repo/include -I$W/envs/pbench/include $T/repo/bench/protocol_geometry/detect_fastq.cpp -o $T/detect \
  -L$W/envs/pbench/lib -Wl,-rpath,$W/envs/pbench/lib -lzstd -lz 2>&1 | tail -5
cat > $T/census.py <<'PY'
import glob, os, random, subprocess, sys, collections, concurrent.futures as cf
import pandas as pd
W = "/mnt/projects/debruinz_project/debruinz/singlet/"; T = sys.argv[1]
N, MAXR, SEED = int(os.environ.get("N", 150)), int(os.environ.get("MAXRUNS", 6)), int(os.environ.get("SEED", 11))
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(W + "data/atlas-rerun/runs*.parquet"))], ignore_index=True).drop_duplicates("run_accession")
d = d[d.read_count > 0].copy()
if "gsm" not in d: d["gsm"] = None
d["gsm"] = d["gsm"].fillna(d["sample_alias"].where(d["sample_alias"].astype(str).str.startswith("GSM"), d["sample_accession"]))
g = d.groupby("gsm").run_accession.apply(list)
multi = g[g.map(len) >= 2]
rng = random.Random(SEED)
pick = rng.sample(list(multi.index), min(N, len(multi)))
jobs = [(gsm, r) for gsm in pick for r in sorted(multi[gsm])[:MAXR]]
print("multi-run samples in the universe: %d; probing %d samples, %d runs" % (len(multi), len(pick), len(jobs)))
def probe(j):
    gsm, run = j
    dd = "%s/%s" % (T, run); os.makedirs(dd, exist_ok=True)
    try:
        subprocess.run(["fastq-dump", "-X", "5000", "--split-files", "-O", dd, run], capture_output=True, timeout=300)
    except Exception:
        return None
    fs = sorted(glob.glob(dd + "/*.fastq"))
    if len(fs) < 2: return None
    L = []
    for f in fs:
        with open(f) as fh:
            fh.readline(); L.append(len(fh.readline().strip()))
    top = sorted(sorted(range(len(fs)), key=lambda i: -L[i])[:2])
    os.makedirs(T + "/fq", exist_ok=True)
    a, b = "%s/fq/%s_a.fastq" % (T, run), "%s/fq/%s_b.fastq" % (T, run)
    os.replace(fs[top[0]], a); os.replace(fs[top[1]], b)
    subprocess.run(["rm", "-rf", dd])
    return (gsm, run, a, b)
with cf.ThreadPoolExecutor(8) as ex:
    got = [x for x in ex.map(probe, jobs) if x]
with open(T + "/list.tsv", "w") as fh:
    for gsm, run, a, b in got: fh.write("%s|%s\t%s\t%s\n" % (gsm, run, a, b))
out = subprocess.run([T + "/detect", W + "pipeline-benchmark/whitelists", T + "/list.tsv", "5000"], capture_output=True, text=True).stdout
rows = []
for l in out.splitlines():
    p = l.split("\t")
    if len(p) == 7:
        gsm, run = p[0].split("|"); rows.append(dict(gsm=gsm, run=run, tag=p[1], conf=int(p[2]), wl=float(p[3]), r1_length=p[4], bc_len=p[5]))
R = pd.DataFrame(rows)
R.to_csv(T + "/runs.tsv", sep="\t", index=False)
k = R.groupby("gsm").agg(n=("run", "count"), tags=("tag", lambda s: sorted(set(s))), minconf=("conf", "min"))
k = k[k.n >= 2]
dis = k[k.tags.map(len) > 1]
print("samples with >= 2 probed runs: %d; runs with different tags: %d (%.1f%%)" % (len(k), len(dis), 100 * len(dis) / max(len(k), 1)))
print("samples with a run below MEDIUM confidence: %d" % int((k.minconf < 2).sum()))
print("tag sets of the disagreeing samples:"); print(dis.tags.map(lambda t: " | ".join(t)).value_counts().to_string())
print(R[R.gsm.isin(dis.index)].sort_values(["gsm", "run"]).to_string())
print("all tags:", dict(R.tag.value_counts()))
PY
python3 $T/census.py $T
