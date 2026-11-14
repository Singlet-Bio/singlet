#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 32 --mem=300G --time=02:30:00 --exclude=g050 -J rt-ident -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# rt-ident: end-to-end byte identity of the export on the Kang 8-donor pool (.1fq, 182 M reads): bin/vbfix, bin/tailgraph with SINGLET_EXPORT_GRAPH=0,
# bin/tailgraph with the graph on. Every file of each output dir is hashed; the lists are compared (only counts and file names are printed).
set -uo pipefail
umask 077
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/cellarium/reference; REF=/mnt/projects/debruinz_project/cellarium/reference
GD=$REF/GRCh38-2024-A; J=$W/jobs/gpu-runtime; O=$J/results/ident-$SLURM_JOB_ID; mkdir -p $O; chmod go-rwx $O
SCR=/tmp/$USER-ident-$SLURM_JOB_ID; mkdir -p $SCR
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_ALLSITE_TIMING=1
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70
export SINGLET_GPU_SHARED_INDEX=1 SINGLET_ENGINE_GPU_SLOTS=2 SINGLET_ENGINE_GPU_MEM_GB=60 SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16
export CUDA_VISIBLE_DEVICES=0
cp $W/modality-bench/genotype_pool/kang21.1fq $SCR/kang.1fq
T1K=$W/jobs/gpu-campaign/agents/hla-speed/t1k-p3b
run() { lab=$1; bin=$2; shift 2; mkdir -p $SCR/$lab
  s=$(date +%s.%N)
  env "$@" numactl --cpunodebind=0 --preferred=0 $bin $SCR/kang.1fq --out-prefix $SCR/$lab --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf --ref-base $REF --pipeline --output-format 1pz --hla-t1k $T1K --no-editing --no-hla --n-donors 8 --engine gpu --threads 16 > $O/$lab.out 2> $O/$lab.err
  echo "RUN $lab rc=$? wall=$(echo "$(date +%s.%N) - $s" | bc) s total=$(grep -a '\[export\] Total' $O/$lab.err | tail -1)"
  (cd $SCR/$lab && find . -type f | sort | while read -r f; do echo "$(md5sum < "$f" | cut -c1-32) $f"; done) > $O/$lab.md5
  echo "  $lab: $(wc -l < $O/$lab.md5) files"
  mkdir -p $O/$lab.small; for f in ambient_contamination.json summary.json provenance.json read_stats.tsv pileup_stats.json; do cp $SCR/$lab/$f $O/$lab.small/ 2>/dev/null; done ; }
run serial $J/bin/tailgraph/singlet-gpu SINGLET_EXPORT_GRAPH=0
run graph  $J/bin/tailgraph/singlet-gpu X=1
run vbfix  $J/bin/vbfix/singlet-gpu X=1
cmpl() { a=$1; b=$2
  echo "CMP $a vs $b: $(comm -3 <(sort $O/$a.md5) <(sort $O/$b.md5) | wc -l) differing lines"
  join -j2 <(sort -k2 $O/$a.md5) <(sort -k2 $O/$b.md5) -a1 -a2 -e MISSING -o 0,1.1,2.1 | awk '$2!=$3{print "  DIFF " $1}' ; }
cmpl serial graph; cmpl vbfix graph; cmpl vbfix serial
dk() { a=$1; b=$2; for f in ambient_contamination.json summary.json provenance.json read_stats.tsv pileup_stats.json; do echo "  $a vs $b $f: $(diff $O/$a.small/$f $O/$b.small/$f | grep -c '^[<>]') differing lines; keys: $(diff $O/$a.small/$f $O/$b.small/$f | grep '^[<>]' | sed -E 's/^[<>] *"?([A-Za-z_0-9]+).*//' | sort -u | head -12 | tr '
' ' ')"; done; }
dk serial graph; dk vbfix graph
for l in serial graph vbfix; do echo "== $l"; grep -a -E '\[export\] (Total|CSC)|\[ambient\] method|\[doublet\] n_cells|\[multimap\]|\[te\] |\[cell_calling\] n_cells|\[mt\] ref' $O/$l.err | sed -E 's/(allele|genotype)[^ ]*//' | cut -c1-200; done
chmod -R go-rwx $O
rm -rf $SCR
