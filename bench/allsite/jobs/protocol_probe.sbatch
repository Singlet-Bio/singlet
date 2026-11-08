#!/bin/bash
#SBATCH --job-name=proto-probe
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Why encode auto-detected GSE230239 (10x 3' v3, GEO) as split-seq: protocol_probe.py replays detect_protocol's scores on the first 5000
# pairs, and the engine's own encoder runs on the first 400k pairs with auto-detection (its log lines) and with --protocol 10x-3p-v3.
# Needs BINDIR, R1, R2, NAME. Results: $W/jobs/allsite/results/protocol/$NAME/.
set -uo pipefail
: "${BINDIR:?}" "${R1:?}" "${R2:?}" "${NAME:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; PY=$W/envs/pbench/bin/python3; WL=$W/pipeline-benchmark/whitelists
R=$W/jobs/allsite/results/protocol/$NAME; mkdir -p $R
SCR=/tmp/$USER-pp-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
BARE=$W/repos/singlet-py.git; BR=${BR:-feat/allele-outputs}
( flock 9; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/pp/$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/pp/$SLURM_JOB_ID bench/allsite/protocol_probe.py | tar -x -C $SCR
$PY -I $SCR/bench/allsite/protocol_probe.py $R1 $R2 $WL $R/probe.json 5000
zcat $R1 | head -1600000 | gzip -1 > $SCR/R1.fastq.gz; zcat $R2 | head -1600000 | gzip -1 > $SCR/R2.fastq.gz
export SINGLET_WHITELIST_DIR=$WL
for mode in auto 10x-3p-v3; do
  ( cd $SCR && $BINDIR/singlet-gpu encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz $([ $mode = auto ] || echo --protocol $mode) -o $SCR/$mode.1fq ) 2>&1 \
    | tr '\r' '\n' | grep -v -E "^\[singlet-encode\] [0-9]+ / " | tee $R/encode_$mode.log
done
echo done $(date -Is)
