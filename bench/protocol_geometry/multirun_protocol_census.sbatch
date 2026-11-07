#!/bin/bash
#SBATCH --job-name=s14-multirun-proto
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# How does the per-sample protocol vote (fetch_worker.vote_runs) play out on the campaign's multi-run samples? A random sample of N multi-run GSMs (>= 2 runs, <= MAXRUNS probed
# each), optionally only the campaign's (FILTER=human10x: organism contains Homo and the catalogue's protocol_inferred is 10x* / citeseq). Every run is probed EXACTLY as the
# fetch worker probes it (fetch_worker.probe_run_protocol: `singlet download <run> --max-reads 5000`, the production gate and encoder on the run's real segments), then
# vote_runs decides. (An earlier version probed with fastq-dump, which drops technical reads and so mis-read runs whose barcode read is a technical segment.)
# Env: N (150), MAXRUNS (6), SEED (11), FILTER, BRANCH (default fix/protocol-vote), BIN (a singlet build).
set -uo pipefail
# the engine must be an immutable, identified build: <dir>-<commit>/bin/<binary> (or <dir>/<binary>) beside a `commit` file starting with that commit (never a shared mutable directory)
pin() { local r d c; r=$(readlink -f "$1"); d=$(dirname "$r"); [ "$(basename $d)" = bin ] && d=$(dirname $d); c=$(basename $d); c=${c##*[-_]}; [[ $c =~ ^[0-9a-f]{7,40}$ ]] && [ -x "$r" ] && case "$(cat $d/commit 2>/dev/null)" in $c*) return 0;; esac; echo "ABORT: $1 is not a pinned build (<dir>-<commit>/bin/<binary> with a commit file starting with the commit)"; exit 3; }
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/bpub/bin:$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
BIN=${BIN:?set BIN=$W/builds/<commit8>/bin/<binary> (an immutable build from scripts/engine_gpu/build_clipper.sbatch)}; pin "$BIN"
T=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
git clone -q --depth 1 -b ${BRANCH:-fix/protocol-vote} git@github.com:Singlet-Bio/singlet.git $T/repo 2>&1 | tail -1; (cd $T/repo && git log --oneline -1 | cut -c1-100)
cat > $T/census.py <<'PY'
import glob, os, random, sys, concurrent.futures as cf
import pandas as pd
W = "/mnt/projects/debruinz_project/debruinz/singlet/"; T, BIN = sys.argv[1], sys.argv[2]
sys.path.insert(0, T + "/repo/scripts/atlas_rerun")
import fetch_worker as FW
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
pick = random.Random(SEED).sample(list(multi.index), min(N, len(multi)))
reads = {r: n for gsm in multi.index for r, n in zip(multi.runs[gsm], multi.reads[gsm])}
jobs = [(gsm, r) for gsm in pick for r in sorted(multi.runs[gsm])[:MAXR]]
print("probing %d samples, %d runs" % (len(pick), len(jobs)), flush=True)
env = dict(os.environ, SINGLET_WHITELIST_DIR=W + "pipeline-benchmark/whitelists")
def probe(j):
    gsm, run = j
    scr = "%s/p_%s" % (T, run); os.makedirs(scr, exist_ok=True)
    try:
        return gsm, run, FW.probe_run_protocol(BIN, run, scr, env, whitelist_dir=W + "pipeline-benchmark/whitelists")
    finally:
        os.system("rm -rf %s" % scr)
with cf.ThreadPoolExecutor(8) as ex:
    got = list(ex.map(probe, jobs))
R = pd.DataFrame([dict(gsm=g_, run=r, tag=c[0], conf=c[1], wl=c[2]) for g_, r, c in got])
R.to_csv(T + "/runs.tsv", sep="\t", index=False)
print("runs probed: %d; without any call (probe failed or gate-rejected): %d" % (len(R), int(R.tag.isna().sum())))
rows = []
for gsm, x in R.groupby("gsm"):
    runs = list(x.run)
    if len(runs) < 2: continue
    calls = {r.run: (r.tag, r.conf, r.wl) for r in x.itertuples()}
    main, vote, others = FW.vote_runs(runs, calls, [reads.get(r, 0) for r in runs])
    ex_ = vote["excluded_runs"]
    rows.append(dict(gsm=gsm, n=len(runs), status=vote["status"], incompatible=vote["incompatible"], low=len(vote["low_confidence_runs"]), n_excluded=len(ex_),
                     excl_reads=sum(e["reads"] for e in ex_), sample_reads=sum(reads.get(r, 0) for r in runs), n_cand=len(others),
                     raw_tags=sorted({str(t) for t in x.tag}), confs=list(x.conf), wls=[None if w is None or w != w else round(w, 2) for w in x.wl]))
V = pd.DataFrame(rows)
print("samples with >= 2 probed runs: %d" % len(V))
print("  raw: runs with different tags: %d" % int((V.raw_tags.map(len) > 1).sum()))
print("  vote status: %s" % dict(V.status.value_counts()))
print("  3p/5p flip settled by the vote (status voted): %d (%.1f%%)" % (int((V.status == "voted").sum()), 100 * (V.status == "voted").mean()))
print("  samples with run(s) lacking a confident call that follow the main group: %d" % int((V.low > 0).sum()))
print("  samples with excluded runs (no confident call and a whitelist rate below half): %d (%.1f%%); excluded runs %d, their reads %.1f M of %.1f M (%.2f%%)" % (
    int((V.n_excluded > 0).sum()), 100 * (V.n_excluded > 0).mean(), int(V.n_excluded.sum()), V.excl_reads.sum() / 1e6, V.sample_reads.sum() / 1e6, 100 * V.excl_reads.sum() / max(V.sample_reads.sum(), 1)))
print("  samples with companion_candidates (a confident other family, or an excluded run with >= 10%% of the reads): %d (%.1f%%)" % (int((V.n_cand > 0).sum()), 100 * (V.n_cand > 0).mean()))
print("  samples where no run has a confident call (status none): %d" % int((V.status == "none").sum()))
print(V[(V.n_excluded > 0) | (V.n_cand > 0) | (V.status == "none")].to_string())
print("all tags:", dict(R.tag.fillna("-").value_counts()))
PY
python3 $T/census.py $T $BIN
