#!/bin/bash
#SBATCH --job-name=s14-bloom
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Builds the species-detection Bloom filters (human_21mer.bloom, mouse_21mer.bloom: canonical 21-mers of all exons of GRCh38-2024-A / GRCm39-2024-A) into $OUT
# (default $WORK/reference-extra/species_filters) and checks them with species_detect::detect on the first 200 k read pairs of the six pilot samples whose detection came
# back 'unknown' (GSM5130155, GSM5171610, GSM7064234, GSM7884577, GSM8164511, GSM8183378), a human and a mouse control. Env: BRANCH (default fix/species-bloom), BIN.
set -uo pipefail
# the engine must be an immutable, identified build: <dir>-<commit>/bin/<binary> (or <dir>/<binary>) beside a `commit` file starting with that commit (never a shared mutable directory)
pin() { local r d c; r=$(readlink -f "$1"); d=$(dirname "$r"); [ "$(basename $d)" = bin ] && d=$(dirname $d); c=$(basename $d); c=${c##*[-_]}; [[ $c =~ ^[0-9a-f]{7,40}$ ]] && [ -x "$r" ] && case "$(cat $d/commit 2>/dev/null)" in $c*) return 0;; esac; echo "ABORT: $1 is not a pinned build (<dir>-<commit>/bin/<binary> with a commit file starting with the commit)"; exit 3; }
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; PB=$W/pipeline-benchmark
export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
OUT=${OUT:-$W/reference-extra/species_filters}; BIN=${BIN:?set BIN=$W/builds/<commit8>/bin/<binary> (an immutable build from scripts/engine_gpu/build_clipper.sbatch)}; pin "$BIN"; mkdir -p $OUT
T=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT; cd $T
git clone -q --depth 1 -b ${BRANCH:-fix/species-bloom} git@github.com:Singlet-Bio/singlet.git repo 2>&1 | tail -1; (cd repo && git log --oneline -1)
CXX=$W/envs/pbench/bin/x86_64-conda-linux-gnu-g++
for t in build_species_bloom species_check; do
  $CXX -O2 -std=c++17 -pthread -Irepo/include -I$W/envs/pbench/include repo/bench/species_detect/$t.cpp -o $t -L$W/envs/pbench/lib -Wl,-rpath,$W/envs/pbench/lib -lzstd -lz 2>&1 | tail -15
done
for sp in "human GRCh38-2024-A" "mouse GRCm39-2024-A"; do set -- $sp
  /usr/bin/time -v -o $T/$1.time ./build_species_bloom --genome $REF/$2/fasta/genome.fa --gtf $REF/$2/genes/genes.gtf --out $OUT/$1_21mer.bloom 2>&1 | tail -4
  grep -E "Elapsed|Maximum resident" $T/$1.time
done
ls -la $OUT; sha256sum $OUT/*.bloom | tee $OUT/SHA256SUMS
export SINGLET_SPECIES_FILTERS=$OUT SINGLET_WHITELIST_DIR=$PB/whitelists
: > list.txt
for g in GSM5130155 GSM5171610 GSM7064234 GSM7884577 GSM8164511 GSM8183378; do
  acc=$(python3 -c "import json;print(json.load(open('$W/pilot/$g/runs.json'))['runs'][0][0])" 2>/dev/null)
  [ -n "$acc" ] || { echo "$g: no runs.json"; continue; }
  $BIN download $acc --max-reads 200000 --vdb-threads 2 --no-profile --quality binned --whitelist-dir $PB/whitelists -o $T/$g.1fq > $T/$g.log 2>&1; echo "$g $acc rc=$?"
  echo $T/$g.1fq >> list.txt
done
ls $PB/data/pbmc1k_v3/pbmc1k_v3.1fq $PB/data/neuron1k_v3/*.1fq >> list.txt
echo "== species_detect::detect with the Bloom filters (expected: human for the six pilot samples and pbmc1k_v3; mouse for neuron1k_v3)"
./species_check $(cat list.txt) 2>&1 | sed "s|$T/||; s|$PB/data/||" | cut -c1-230
echo "== the same without the filters (the diagnostic fallback the campaign has been using)"
SINGLET_SPECIES_FILTERS=/nonexistent ./species_check $(cat list.txt) 2>&1 | sed "s|$T/||; s|$PB/data/||" | cut -c1-230
