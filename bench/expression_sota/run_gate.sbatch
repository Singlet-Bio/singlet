#!/bin/bash
#SBATCH --job-name=ec-run
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# sbatch --export=ALL,SHA8=<8 chars of the built commit>,ENGINE=cpu|gpu run_gate.sbatch DATASET   (the .1fq in pipeline-benchmark/data)
# Same command line as the release-gate runs of the pin (--pipeline --output-format mtx --raw-matrix, 16 threads); output $RG/runs/ec/<SHA8>-<engine>/<ds>/
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; RG=$W/release-gate; REF=/mnt/projects/debruinz_project/cellarium/reference
DS=$1; ENGINE=${ENGINE:-cpu}; BIN=$RG/bin/$SHA8/singlet-$ENGINE; ARGS=(); [ $ENGINE = gpu ] && ARGS=(--engine gpu)
case $DS in hgmm*) SP=GRCh38-GRCm39-barnyard;; neuron*|nuclei*) SP=GRCm39-2024-A;; *) SP=GRCh38-2024-A;; esac
G=$REF/$SP; O=$RG/runs/ec/$SHA8-$ENGINE/$DS; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; export TMPDIR=$SCR SINGLET_STALL_TIMEOUT=10800
cp $PB/data/$DS/$DS.1fq $SCR/s.1fq; cd $SCR
$BIN $SCR/s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline --output-format mtx --raw-matrix "${ARGS[@]}" > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -c "BAM: unsorted STAR stream" $O/run.log > $O/stream_ok.txt || true
tail -2 $O/run.log
