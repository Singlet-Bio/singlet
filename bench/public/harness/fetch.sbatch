#!/bin/bash
#SBATCH --job-name=bpub-fetch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Download one cohort dataset into $BP/data/<id>/ and classify its FASTQs by library (libs.tsv: library, read, path).
#   sbatch --array=1-N%4 fetch.sbatch            (N = lines of $BP/work/datasets.txt, written by submit.sh)
#   sbatch fetch.sbatch <dataset_id>
# Sources: 10x tar (curl); GEO/SRA runs (prefetch + fasterq-dump, or fastq-dump -X when the cohort caps the reads; NCBI is
# ~100x faster than ENA HTTPS from Clipper); Kang 2018 is a BAM only (reads rebuilt from CR/UR tags, as the genotype-pool
# benchmark did; that copy is reused when present). Data already fetched by the pipeline-benchmark session is linked, not
# downloaded again.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__bench-public-cohort-485cf7/bench/public/harness/common.sh
ID=${1:-$(sed -n "${SLURM_ARRAY_TASK_ID}p" $BP/work/datasets.txt)}
cohort_row $ID
D=$BP/data/$ID; mkdir -p $D; cd $D
[ -e .done ] && { echo "[fetch] $ID already fetched"; exit 0; }
echo "[fetch] $ID source=${C[source]} $(date -Is)"
CAP=${C[read_pairs_used_m]%.*}; TOT=${C[read_pairs_m]%.*}

classify() {   # libs.tsv from file names (10x naming: <sample>_S1_L001_R1_001.fastq.gz)
  find -L $D -name "*.fastq.gz" | sort | while read f; do
    b=$(basename $f); lb=$(echo "$b" | tr A-Z a-z); lib=gex
    case "$lb" in
      *atac*) lib=atac ;; *crispr*|*_cr_*) lib=crispr ;; *vdj_t*|*_t_s[0-9]*|*tcr*) lib=vdj_t ;; *vdj_b*|*_b_s[0-9]*|*bcr*) lib=vdj_b ;;
      *antibody*|*_adt*|*_ab_*|*_f_s[0-9]*|*protein*|*_fb_*) lib=adt ;; *hto*|*hashtag*|*cmo*|*multiplexing*) lib=hto ;;
    esac
    rd=$(echo "$b" | grep -oE "_(R1|R2|R3|I1|I2)_" | tr -d _ || true); [ -n "$rd" ] || rd=$(echo "$b" | grep -oE "_[123]\.fastq" | tr -d _. | sed 's/fastq//;s/^/R/')
    printf "%s\t%s\t%s\n" "$lib" "$rd" "$f"
  done > libs.tsv
  cut -f1,2 libs.tsv | sort | uniq -c
}

case ${C[source]} in
10x)
  PREV=$PB/data/$ID
  if [ -d $PREV ] && ls $PREV/*/ >/dev/null 2>&1; then ln -sfn $PREV prev; echo "[fetch] linked $PREV"
  else
    curl -sSL --retry 5 -C - -o fastqs.tar "${C[url]}"
    tar -xf fastqs.tar && rm -f fastqs.tar
  fi ;;
geo|sra)
  if [ "$ID" = kang2018_ctrl ] && [ -s $WORK/modality-bench/genotype_pool/fastq/R1.fastq.gz ]; then
    mkdir -p kang; ln -sf $WORK/modality-bench/genotype_pool/fastq/R1.fastq.gz kang/kang_S1_L001_R1_001.fastq.gz
    ln -sf $WORK/modality-bench/genotype_pool/fastq/R2.fastq.gz kang/kang_S1_L001_R2_001.fastq.gz
  else
    IFS=';' read -r -a RUNS <<< "${C[srr]}"
    if [ ${#RUNS[@]} -eq 0 ] || [[ ${RUNS[0]} != SRR* ]]; then
      # GSM list or ranges (GSM3836772-GSM3837155) -> runs, via ENA (a GEO sample's ENA sample_alias is its GSM)
      mapfile -t RUNS < <(python3 $SRC/bench/public/harness/gsm_runs.py "${C[gsm]}")
      echo "[fetch] resolved ${#RUNS[@]} runs from ${C[gsm]}"
      [ ${#RUNS[@]} -gt 0 ] || { echo "no runs for $ID"; exit 2; }
    fi
    SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
    for r in "${RUNS[@]}"; do
      mkdir -p $r
      if [ -n "$CAP" ] && [ -n "$TOT" ] && [ "$CAP" -lt "$TOT" ]; then
        # capped: the first N spots of each run, split proportionally (fastq-dump streams; no full download)
        N=$(( CAP * 1000000 / ${#RUNS[@]} ))
        fastq-dump -X $N --split-files --gzip -O $r $r
      else
        prefetch -q -O $SCR $r && fasterq-dump -q -e 8 -t $SCR -O $r $SCR/$r && pigz -p 8 $r/*.fastq; rm -rf $SCR/$r
      fi
      # SRA names reads <run>_1/_2/_3; map to 10x-style names so classify() and the runners treat them alike
      for f in $r/${r}_*.fastq.gz; do i=$(basename $f .fastq.gz); i=${i##*_}; mv $f $r/${r}_S1_L001_R${i}_001.fastq.gz; done
    done
  fi ;;
*) echo "source ${C[source]} is fetched by its owner's scripts (see cohort.md)"; exit 0 ;;
esac
classify
du -shL $D; touch .done
