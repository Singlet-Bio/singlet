#!/bin/bash
#SBATCH --job-name=dp-chem
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch chem_probe.sbatch <binary> <run> [<run> ...]
# Chemistry calls of `download` with and without whitelists (binary copied away from ../whitelists, no env: the
# smoke test's situation) and of `screen-runs`, on the first 200k reads of each run (remote stream).
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
BIN=$PB/bin/$1; shift
SCR=/tmp/$USER-chem-$SLURM_JOB_ID; mkdir -p $SCR/nowl/bin; trap 'rm -rf $SCR' EXIT; cd $SCR
cp $BIN $SCR/nowl/bin/singlet
for R in "$@"; do
  echo "===== $R"
  echo "--- download, no whitelists"
  env -u SINGLET_WHITELIST_DIR $SCR/nowl/bin/singlet download $R --max-reads 200000 --vdb-threads 2 --no-profile -o $SCR/$R.nowl.1fq 2>&1 \
    | tr '\r' '\n' | grep -E "gate|protocol_detect|Protocol|R1 length|swap|BC dict|whitelist|celseq|WARN" | head -20
  echo "--- download, SINGLET_WHITELIST_DIR"
  SINGLET_WHITELIST_DIR=$PB/whitelists $SCR/nowl/bin/singlet download $R --max-reads 200000 --vdb-threads 2 --no-profile -o $SCR/$R.wl.1fq 2>&1 \
    | tr '\r' '\n' | grep -E "gate|protocol_detect|Protocol|R1 length|swap|BC dict|whitelist|celseq|WARN" | head -20
  echo "--- screen-runs on both heads"
  SINGLET_WHITELIST_DIR=$PB/whitelists $SCR/nowl/bin/singlet screen-runs --json $SCR/$R.json $SCR/$R.nowl.1fq $SCR/$R.wl.1fq 2>&1 | head -6
  python3 -c "import json;d=json.load(open('$SCR/$R.json'));[print(r['run_id'],r['chemistry'],r['best_whitelist'],r['best_rate'],r['r1_len_mode'],r['r2_len_mode'],[(w['file'],round(w['one_mm'],3)) for w in r['whitelists']]) for r in d['runs']]" 2>&1
done
