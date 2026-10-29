#!/bin/bash
#SBATCH --job-name=w11-ing-cpu
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU-only (no GPU gres): zstd level ladder for the transform, host decode vs the .1fq library decode (CPU-engine compatibility),
# the upload dry run. Env: SAMPLE_LABEL, SAMPLE_1FQ. .1fg files live only in job /tmp.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
D=$W/jobs/w11/ingest; M8=$W/jobs/w11/m8; BIN=${BIN:-$D/bin2}
SAMPLE_LABEL=${SAMPLE_LABEL:?}; Q=${SAMPLE_1FQ:?}
OUT=$D/results/cpu_${SAMPLE_LABEL}_$SLURM_JOB_ID; mkdir -p $OUT; R=$OUT/results.jsonl
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
ctr() { apptainer exec -B $W --env LD_LIBRARY_PATH=$M8/lib $IMG "$@"; }
echo "== node $(hostname) job $SLURM_JOB_ID $(date -Is) sample $SAMPLE_LABEL"; lscpu | grep -E "Model name"
run() { echo "[$(date +%T)] $*" >&2; "$@" >> $R 2> $OUT/stderr.last || { echo "FAILED rc=$? $*" >&2; cat $OUT/stderr.last >&2; }; }
tag() { sed -i "\$s/^{/{\"sample\":\"$SAMPLE_LABEL\",\"run\":\"$1\",/" $R; }
tag2() { n=$(wc -l < $R); sed -i "$((n-1)),\$s/^{/{\"sample\":\"$SAMPLE_LABEL\",\"run\":\"$1\",/" $R; }
ENC="ctr $BIN/enc_1fg --in $Q --threads 4 --crc 1 --chunk-pairs 32768"
for L in -3 -1 1 3 5; do run $ENC --out $S/l.1fg --codec zstd --level $L; tag enc_level_$L; rm -f $S/l.1fg; done
run $ENC --out $S/z1.1fg --codec zstd --level 1; tag enc_level_1_keep
echo "== CPU-engine compatibility (byte-numeric reads, hash of every read)"
run ctr $BIN/dec_1fg --fg $S/z1.1fg --fq $Q --threads 4; tag2 cpu_decode_compare_t4_hash
run ctr $BIN/dec_1fg --fg $S/z1.1fg --fq $Q --threads 1 --nohash; tag2 cpu_decode_compare_t1_nohash
echo "== upload dry run"
python3 $D/src/upload_dryrun.py $S/z1.1fg --accession $SAMPLE_LABEL --bucket DRYRUN-NO-BUCKET --part-mib 64 --out $OUT/manifest_dryrun.json > $OUT/dryrun.stdout 2> $OUT/dryrun.stderr; head -n 60 $OUT/dryrun.stdout; cat $OUT/dryrun.stderr
echo "== done $(date -Is)"; cut -c1-400 $R
