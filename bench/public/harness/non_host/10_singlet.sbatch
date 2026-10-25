#!/bin/bash
#SBATCH --job-name=bp-nh-singlet
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch 10_singlet.sbatch LABEL        LABEL = gse166766_hbec_mock | gse166766_hbec_dpi2 | pbmc1k_v3
# Singlet 1.0.0 (pinned campaign binary) with the non-host screen on, as the campaign runs it: SINGLET_REF_BASE set, so the
# viral, bacterial and fungal screens and the host Bloom filter run on the reads STAR could not map. 16 threads.
# Output: $BP/nonhost/LABEL/singlet/ (nonhost/nonhost_per_cell.tsv, nonhost_summary.json, nonhost_em_abundance.tsv,
# star_Unmapped.out.mate1/2 for the Kraken2 and minimap2 runs), time.txt and mem.json of the full run.
set -eu
source ${BP_SRC:-/mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet}/bench/public/harness/perf_common.sh
L=$1; verify_bin
OLD=$WORK/modality-bench/sj_nonhost
O=$BP/nonhost/$L/singlet; rm -rf $O; mkdir -p $O/nonhost $BP/nonhost/cells
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
case $L in
  gse166766_hbec_mock) R1=$OLD/data/mock/R1.fastq.gz; R2=$OLD/data/mock/R2.fastq.gz; CELLS=$OLD/cells/mock.txt ;;
  gse166766_hbec_dpi2) R1=$OLD/data/dpi2/R1.fastq.gz; R2=$OLD/data/dpi2/R2.fastq.gz; CELLS=$OLD/cells/dpi2.txt ;;
  pbmc1k_v3)
    D=$PB/data/pbmc1k_v3; cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz; cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
    R1=$SCR/R1.fastq.gz; R2=$SCR/R2.fastq.gz; CELLS=$OLD/cells/pbmc1k_v3.txt ;;
  *) echo "unknown label $L"; exit 2 ;;
esac
cp $CELLS $BP/nonhost/cells/$L.txt
REF=$REFBASE/GRCh38-2024-A
echo "[bp] $L node=$(hostname) cpu=$(cpu_model) $(date -Is)" | tee $O/node.txt
timed $O/encode $BIN encode --reads $R1 $R2 -o $SCR/s.1fq > $O/encode.log 2>&1
export SINGLET_REF_BASE=$REFBASE SINGLET_STALL_TIMEOUT=10800
cd $SCR
timed $O/run $BIN $SCR/s.1fq --ref-base $REFBASE --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf \
  --out-prefix $O/ --threads 16 --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -hE "Elapsed|Maximum resident" $O/encode/time.txt $O/run/time.txt; cat $O/run/mem.json; echo
grep -E "nonhost|non-host|Non-host" $O/run.log | head -20
ls $O $O/nonhost
