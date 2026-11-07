#!/bin/bash
#SBATCH --job-name=amb-run
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# ambient-rna A/B on one barnyard library with a given binary. Env: BIN (singlet binary, whitelists at BIN/../../whitelists), ID (hgmm1k_v3 | hgmm1k_v2), TAG.
# Runs the pipeline as the gate does (--pipeline --output-format mtx --raw-matrix, 16 threads) into release-gate/ambient/runs/$ID/singlet-$TAG,
# rebuilds the per-cell artifact with the competitors' existing outputs (work/$ID) into release-gate/ambient/artifacts-$TAG, and prints
# the ambient-rna metrics of the Singlet column. hgmm1k_v2 is the out-of-sample library: never tuned on.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; A=$W/release-gate/ambient; PB=$W/pipeline-benchmark
ID=${ID:?}; TAG=${TAG:?}; BIN=${BIN:?}
REFB=/mnt/projects/debruinz_project/cellarium/reference; REF=$REFB/GRCh38-GRCm39-barnyard
O=$A/runs/$ID/singlet-$TAG; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export PATH=$W/envs/pbench/bin:$PATH SINGLET_STALL_TIMEOUT=7200
if [ -f $PB/data/$ID/$ID.1fq ]; then FQ=$PB/data/$ID/$ID.1fq
else
  L=$A/data/$ID/gex_fastqs.txt
  cat $(grep "_R1_" $L) > $SCR/R1.fastq.gz; cat $(grep "_R2_" $L) > $SCR/R2.fastq.gz
  $BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/$ID.1fq > $O/encode.log 2>&1; rm -f $SCR/R1.fastq.gz $SCR/R2.fastq.gz; FQ=$SCR/$ID.1fq
fi
cd $SCR
$BIN $FQ --ref-base $REFB --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1
grep -c "BAM: unsorted STAR stream into the pileup" $O/run.log | sed 's/^/stream-path-lines: /'
grep -E "\[ambient\]" $O/run.log | head -5
export PATH=$W/envs/ambient/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl
python -I $A/harness/barnyard_artifacts.py $ID $O $A/work/$ID $A/artifacts-$TAG > $SCR/art.log 2>&1 || { tail -20 $SCR/art.log; exit 1; }
python -I $W/jobs/gate2/amb_score.py $A/artifacts-$TAG/ambient-rna $ID
