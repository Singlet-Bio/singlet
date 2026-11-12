#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:quadro_rtx_8000:1 -c 18 --mem=170G --time=01:30:00 --exclude=g050 -J hla-presift -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# hla-presift: the candidate reads of the HLA prefilter, byte for byte, from three routes (the identity gate sets the read files aside):
#   A  the reference build's `singlet hla --defer` (the prefilter's own pass over the .1fq, as released)
#   B  the new build's `singlet hla --defer` (the same pass after scan_block was factored out)
#   C  the new build's GPU pipeline with SINGLET_HLA=defer (the prefilter inside the read ingest's pass: Presift)
# Per sample: the three files' sha256 (the gzip file and its decompressed stream) and read counts. Prints hashes and counts only.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
OLD=${OLD:-$W/jobs/gpu-pr2/bin2/rel138-e0824b21/singlet-gpu}; NEW=${NEW:?NEW: the new build}; TK=${TK:-$W/jobs/gpu-campaign/agents/hla-speed/t1k-p3b}
SAMPLES=${SAMPLES:-"pbmc2m pbmc1k_v3 pbmc5p_v1 nuclei2k_v2"}; T=${SLURM_CPUS_PER_TASK:-18}; GD=$REF/GRCh38-2024-A
SCR=/tmp/$USER-hlapresift-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; umask 077
export PATH=$W/envs/pbench/bin:$PATH SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_REF_BASE=$REF SINGLET_STALL_TIMEOUT=3600 TMPDIR=$SCR
echo "node=$(hostname) old=$OLD new=$NEW ($(cat $(dirname $NEW)/BUILD.txt 2>/dev/null | grep patch_sha | cut -c1-30)) $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
h() { [ -s "$1" ] && echo "gz $(sha256sum < $1 | cut -c1-16) stream $(zcat $1 | sha256sum | cut -c1-16) reads $(( $(zcat $1 | wc -l) / 4 ))" || echo "MISSING"; }
bad=0
for s in $SAMPLES; do
  F=$PB/data/$s/$s.1fq; [ -s $F ] || { echo "$s: no input"; continue; }
  D=$SCR/$s; mkdir -p $D/a $D/b $D/c
  $OLD hla --defer --t1k $TK --out $D/a --threads $T $F > $D/a.log 2>&1
  $NEW hla --defer --t1k $TK --out $D/b --threads $T $F > $D/b.log 2>&1
  ( cd $D && SINGLET_HLA=defer SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70 CUDA_VISIBLE_DEVICES=0 \
      $NEW $F --ref-base $REF --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf --out-prefix $D/c/ --threads $T --pipeline --output-format 1pz --engine gpu \
      --hla-t1k $TK --no-editing > $D/c.log 2>&1 ); rc=$?
  A=$(h $D/a/private/hla/pending/candidate_reads.fq.gz); B=$(h $D/b/private/hla/pending/candidate_reads.fq.gz); C=$(h $D/c/private/hla/pending/candidate_reads.fq.gz)
  echo "== $s (pipeline rc=$rc)"; echo "  A released pass:      $A"; echo "  B new build's pass:   $B"; echo "  C inside the ingest:  $C"
  grep -a -h "hla\] prefilter" $D/a.log $D/b.log $D/c.log | cut -c1-200 | sed 's/^/  /'
  if [ "$A" = "$B" ] && [ "$A" = "$C" ] && [ "$A" != MISSING ]; then echo "  IDENTICAL"; else echo "  DIFFERENT"; bad=1; grep -a -E "1fq-ingest\]|\[hla\]" $D/c.log | head -8 | cut -c1-200; fi
  rm -rf $D
done
[ $bad = 0 ] && echo "HLA-PRESIFT-CHECK PASS" || echo "HLA-PRESIFT-CHECK FAIL"
