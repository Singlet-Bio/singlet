#!/bin/bash
#SBATCH --job-name=s14-multirun-proto
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Do the runs of one multi-run sample get different protocols from the decode gate's detector? A random sample of N multi-run GSMs (>= 2 runs, <= MAXRUNS probed each),
# optionally only the campaign's (FILTER=human10x: organism contains Homo and the catalogue's protocol_inferred is 10x* / citeseq). For every run the first 5000 spots are read
# from NCBI (fastq-dump -X) and detect_protocol (bench/protocol_geometry/detect_fastq.cpp, real whitelists) is run on the two longest reads. Per sample: the tags of its runs;
# "flip" = the tags differ only in the 3p/5p flag (one family), "incompatible" = different families (whitelist / geometry). Env: N (150), MAXRUNS (6), SEED (11), FILTER, BRANCH.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/bpub/bin:$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
T=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
git clone -q --depth 1 -b ${BRANCH:-fix/protocol-vote} git@github.com:Singlet-Bio/singlet.git $T/repo 2>&1 | tail -1
$W/envs/pbench/bin/x86_64-conda-linux-gnu-g++ -O1 -std=c++17 -pthread -I$T/repo/include -I$W/envs/pbench/include $T/repo/bench/protocol_geometry/detect_fastq.cpp -o $T/detect \
  -L$W/envs/pbench/lib -Wl,-rpath,$W/envs/pbench/lib -lzstd -lz 2>&1 | tail -5
cat > $T/census.py <<'PY'
import glob, os, random, re, subprocess, sys, concurrent.futures as cf
import pandas as pd
W = "/mnt/projects/debruinz_project/debruinz/singlet/"; T = sys.argv[1]
N, MAXR, SEED, FILT = int(os.environ.get("N", 150)), int(os.environ.get("MAXRUNS", 6)), int(os.environ.get("SEED", 11)), os.environ.get("FILTER", "")
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(W + "data/atlas-rerun/runs*.parquet"))], ignore_index=True).drop_duplicates("run_accession")
d = d[d.read_count > 0].copy()
if "gsm" not in d: d["gsm"] = None
d["gsm"] = d["gsm"].fillna(d["sample_alias"].where(d["sample_alias"].astype(str).str.startswith("GSM"), d["sample_accession"]))
if FILT == "human10x":
    s = pd.read_parquet(W + "data/atlas-rerun/samples.parquet")
    pi = s.protocol_inferred.fillna("")
    keep = set(s[s.organism.fillna("").str.contains("Homo") & (pi.str.startswith("10x") | (pi == "citeseq"))].gsm_id)
    d = d[d.gsm.isin(keep)]
g = d.groupby("gsm").agg(runs=("run_accession", list), reads=("read_count", list))
multi = g[g.runs.map(len) >= 2]
print("multi-run samples in the universe%s: %d" % (" (%s)" % FILT if FILT else "", len(multi)))
rng = random.Random(SEED)
pick = rng.sample(list(multi.index), min(N, len(multi)))
jobs = [(gsm, r) for gsm in pick for r in sorted(multi.runs[gsm])[:MAXR]]
print("probing %d samples, %d runs" % (len(pick), len(jobs)), flush=True)
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
sys.path.insert(0, os.environ["SRC"])
import fetch_worker as FW
reads = {r: n for gsm in multi.index for r, n in zip(multi.runs[gsm], multi.reads[gsm])}
rows2 = []
for gsm, x in R.groupby("gsm"):
    if len(x) < 2: continue
    runs = list(x.run); calls = {r.run: (r.tag, r.conf) for r in x.itertuples()}
    main, vote, others = FW.vote_runs(runs, calls, [reads.get(r, 0) for r in runs])
    rows2.append(dict(gsm=gsm, n=len(runs), status=vote["status"], incompatible=vote["incompatible"], low=len(vote["low_confidence_runs"]), n_other_runs=len(vote.get("other_runs", [])),
                      raw_tags=sorted(set(x.tag)), conf_min=int(x.conf.min())))
V = pd.DataFrame(rows2)
print("samples with >= 2 probed runs: %d" % len(V))
print("  raw: runs with different tags: %d" % int((V.raw_tags.map(len) > 1).sum()))
print("  vote status: %s" % dict(V.status.value_counts()))
print("  3p/5p flip settled by the vote (status voted): %d (%.1f%%)" % (int((V.status == "voted").sum()), 100 * (V.status == "voted").mean()))
print("  with run(s) lacking a confident call that follow the main group: %d (%.1f%%)" % (int((V.low > 0).sum()), 100 * (V.low > 0).mean()))
print("  genuinely different families (confident calls) -> companion_candidates: %d (%.1f%%)" % (int(V.incompatible.sum()), 100 * V.incompatible.mean()))
print(V[V.incompatible].to_string())
print("all tags:", dict(R.tag.value_counts()))
PY
SRC=$T/repo/scripts/atlas_rerun python3 $T/census.py $T
