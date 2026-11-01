#!/bin/bash
#SBATCH --job-name=s14-spotlen
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# How many campaign samples carry an UNTRIMMED 10x barcode read (R1 longer than the protocol length: 28 nt v3, 26 nt v2) -- the samples a "R1 length" reject would drop?
# 1. ENA run table of the atlas re-run universe (data/atlas-rerun/runs*.parquet): per-sample mean spot length = bases / spots (read-weighted over runs).
#    A standard 10x GEX spot is R1 26-28 + R2 ~90-98 (+ index) = ~115-140; untrimmed deposits (R1 151 + R2 151 (+8)) are ~300. The length alone cannot separate 10x from
#    paired Smart-seq, so
# 2. a stratified random sample of samples (N_LONG with spot length >= 150, N_SHORT below) gets its first run's first 2000 spots read from NCBI (fastq-dump -X) and
#    its reads tested against the 10x v3 / v2 whitelists at offset 0 (exact 16-mer): a sample is "untrimmed 10x" when >= 50% of spots match and that read is
#    longer than the protocol R1 by >= 10 nt. Counts are extrapolated to the strata.
# Env: N_LONG (default 160), N_SHORT (default 40), SEED.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/bpub/bin:$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
T=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
cat > $T/census.py <<'PY'
import glob, os, random, subprocess, sys, collections, concurrent.futures as cf
import pandas as pd
W = "/mnt/projects/debruinz_project/debruinz/singlet/"; T = sys.argv[1]
NL, NS, SEED = int(os.environ.get("N_LONG", 160)), int(os.environ.get("N_SHORT", 40)), int(os.environ.get("SEED", 7))
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(W + "data/atlas-rerun/runs*.parquet"))], ignore_index=True).drop_duplicates("run_accession")
d = d[(d.read_count > 0) & (d.base_count > 0)].copy()
if "gsm" not in d: d["gsm"] = None
d["gsm"] = d["gsm"].fillna(d["sample_alias"].where(d["sample_alias"].astype(str).str.startswith("GSM"), d["sample_accession"]))
g = d.groupby("gsm").agg(spots=("read_count", "sum"), bases=("base_count", "sum"), run=("run_accession", "first"), layout=("library_layout", "first"),
                         n_runs=("run_accession", "count")).reset_index()
g["spotlen"] = g.bases / g.spots
print("samples with reads: %d (runs %d)" % (len(g), len(d)))
bins = [0, 100, 140, 150, 200, 250, 350, 600, 1e9]
print("mean spot length bins (samples, share of spots):")
for lo, hi in zip(bins[:-1], bins[1:]):
    s = g[(g.spotlen >= lo) & (g.spotlen < hi)]
    print("  %4d-%-6s %7d samples  %5.1f%% of samples  %5.1f%% of spots" % (lo, hi if hi < 1e8 else "inf", len(s), 100 * len(s) / len(g), 100 * s.spots.sum() / g.spots.sum()))
long_ = g[g.spotlen >= 150]; short = g[g.spotlen < 150]
rng = random.Random(SEED)
pick = lambda df, n: df.iloc[rng.sample(range(len(df)), min(n, len(df)))]
S = pd.concat([pick(long_, NL).assign(stratum="long"), pick(short, NS).assign(stratum="short")], ignore_index=True)
def load(name, k=16):
    return set(l.strip()[:k] for l in open(W + "pipeline-benchmark/whitelists/" + name) if len(l.strip()) >= k)
wl = {"v3": load("3M-february-2018.txt"), "v2": load("737K-august-2016.txt")}
exp = {"v3": 28, "v2": 26}
def probe(row):
    out = {"gsm": row.gsm, "run": row.run, "stratum": row.stratum, "spotlen": round(row.spotlen, 1), "lens": [], "wl": None, "wl_rate": 0.0, "bc_read_len": 0, "tenx": False, "untrimmed": False}
    dd = "%s/%s" % (T, row.run); os.makedirs(dd, exist_ok=True)
    try:
        subprocess.run(["fastq-dump", "-X", "2000", "--split-files", "-O", dd, row.run], capture_output=True, text=True, timeout=300)
    except Exception as e:
        out["err"] = repr(e)[:80]
    reads = []
    for f in sorted(glob.glob(dd + "/*.fastq")):
        seqs = [l.strip() for i, l in enumerate(open(f)) if i % 4 == 1]
        if seqs:
            reads.append((collections.Counter(len(s) for s in seqs).most_common(1)[0][0], seqs))
    subprocess.run(["rm", "-rf", dd])
    out["lens"] = [l for l, _ in reads]
    best = (0.0, None, 0)
    for L, seqs in reads:
        if L < 16: continue
        for v, w in wl.items():
            m = sum(1 for s in seqs if s[:16] in w) / len(seqs)
            if m > best[0]: best = (m, v, L)
    out["wl_rate"], out["wl"], out["bc_read_len"] = round(best[0], 3), best[1], best[2]
    out["tenx"] = bool(best[1] and best[0] >= 0.5)
    out["untrimmed"] = bool(out["tenx"] and best[2] >= exp[best[1]] + 10)
    return out
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(probe, list(S.itertuples())))
R = pd.DataFrame(res)
R.to_csv(T + "/probe.tsv", sep="\t", index=False)
print(R[["gsm", "stratum", "spotlen", "lens", "wl", "wl_rate", "bc_read_len", "tenx", "untrimmed"]][R.tenx].to_string(max_rows=400))
print()
tot = 0
for st, pop in (("long", len(long_)), ("short", len(short))):
    r = R[R.stratum == st]
    if not len(r): continue
    est = round(pop * r.untrimmed.mean()); tot += est
    print("%s stratum: %d samples in the universe, %d probed (%d without data): 10x (>=50%% WL at offset 0) %d (%.1f%%), untrimmed 10x %d (%.1f%%) -> ~%d samples" % (
        st, pop, len(r), int((r.lens.map(len) == 0).sum()), r.tenx.sum(), 100 * r.tenx.mean(), r.untrimmed.sum(), 100 * r.untrimmed.mean(), est))
print("ESTIMATE: ~%d of %d samples (%.2f%%) have an untrimmed 10x barcode read" % (tot, len(g), 100 * tot / len(g)))
PY
python3 $T/census.py $T
