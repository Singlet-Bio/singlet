#!/bin/bash
#SBATCH --job-name=rg-remeasure
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Task 3: re-measure on the pinned commit what was only covered by ctest after merging.
#   #94 SPLiT-seq (SRR31302021) and inDrop (SRR15409581): whitelist rate of 200,000 real reads with the pinned whitelists (shipped
#       next to the binary) and the pinned geometry (bench/protocol_geometry/wl_scan.py), plus the pinned binary's encode of the same reads
#   #91 CRISPR guides: pinned crispr_guide.h on the A375 10k CRISPR library reads and 10x's feature reference, before (match) and after
#       (match_bam) the orientation fix with half the reads stored reverse
#   (#87 HTO is rg-hto.) Output: $RG/results/remeasure_*.txt
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/release-gate/jobs/rg_common.sh
COMMIT=$(cat $RG/bin/$PIN8/commit); BARE=$W/repos/singlet-py.git; WL=$RG/bin/whitelists; OUT=$RG/results; mkdir -p $OUT
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $COMMIT include bench/protocol_geometry bench/crispr_orient | tar -x -C $SCR/src
export PATH=$W/envs/pbench/bin:$W/envs/sratools/bin:$PATH
# ---- #91 CRISPR
A=$W/data/vdjc/a375; CF=$(ls -d $A/fastqs/*/*crispr_fastqs 2>/dev/null | head -1 || true); [ -n "$CF" ] || CF=$(ls -d $A/fastqs/*/*crispr* | head -1)
R1=$(ls $CF/*_R1_*.fastq.gz | head -1); R2=$(ls $CF/*_R2_*.fastq.gz | head -1)
awk -F, 'NR>1 {print $1","$5}' $A/crispr_analysis/crispr_analysis/feature_reference.csv > $SCR/guides.csv; G=$SCR/guides.csv   # 10x feature reference -> name,sequence (the guide library, $(wc -l < $SCR/guides.csv) guides)
for v in OLD NEW; do
  f=; [ $v = NEW ] && f=-DNEWMATCH
  g++ -O2 -std=c++17 $f -I$SCR/src/include -o $SCR/guide_$v $SCR/src/bench/crispr_orient/guide_orient.cpp
done
{ echo "# #91 guide matching, pin $COMMIT, A375 10k CRISPR library, 2,000,000 read pairs, 50% stored reverse"
  echo "before (match):";   $SCR/guide_OLD $R1 $R2 $G 2000000 50
  echo "after (match_bam):"; $SCR/guide_NEW $R1 $R2 $G 2000000 50
  echo "no reverse (reference orientation) after:"; $SCR/guide_NEW $R1 $R2 $G 2000000 0; } | tee $OUT/remeasure_crispr.txt
# ---- #94 protocol geometry (SRA: first 200,000 spots)
for P in splitseq:SRR31302021 indrop:SRR15409581; do
  N=${P%%:*}; S=${P##*:}; D=$SCR/$N; mkdir -p $D; cd $D
  fastq-dump -X 200000 --split-files $S > $D/fd.log 2>&1 || { echo "fastq-dump $S failed"; tail -3 $D/fd.log; continue; }
  ls -la $D | head; head -2 ${S}_2.fastq | cut -c1-120; cd $SCR
done
SP=$SCR/splitseq/SRR31302021_2.fastq; ID=$SCR/indrop/SRR15409581_2.fastq
{ echo "# #94 whitelist rates, pin $COMMIT, 200,000 real reads (R2), whitelists from the pinned release dir"
  if [ -s $SP ]; then echo "SPLiT-seq SRR31302021:"; $W/envs/pbench/bin/python -P $SCR/src/bench/protocol_geometry/wl_scan.py --fq $SP --n 200000 \
     --wl rd3=$WL/splitseq_bc_rd3.txt --wl rd2=$WL/splitseq_bc_rd2.txt --wl rd1=$WL/splitseq_bc_rd1.txt --geom rd3@10,rd2@48,rd1@78 | tail -12; fi
  if [ -s $ID ]; then echo "inDrop SRR15409581:"; $W/envs/pbench/bin/python -P $SCR/src/bench/protocol_geometry/wl_scan.py --fq $ID --n 200000 \
     --wl bc1=$WL/indrop_bc1.txt --maxoff 40 | tail -12; fi; } | tee $OUT/remeasure_protocol_geometry.txt
# the pinned binary's own view of the same reads (barcode read first): protocol auto-detection and with the protocol named
for P in splitseq:SRR31302021 indrop:SRR15409581; do
  N=${P%%:*}; S=${P##*:}; D=$SCR/$N; [ -s $D/${S}_1.fastq ] || continue
  for PR in auto $N; do
    A=(); [ $PR = auto ] || A=(--protocol $PR)
    echo "== $N encode ($PR), barcode read = _2"; $RG/bin/$PIN8/singlet-cpu encode --reads $D/${S}_2.fastq $D/${S}_1.fastq "${A[@]}" -o $D/x.1fq 2>&1 | tr '' '
' | grep -E "Protocol|protocol_detect|ERROR|Done" | tail -5
  done
done | tee -a $OUT/remeasure_protocol_geometry.txt
