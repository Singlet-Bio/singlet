#!/bin/bash
#SBATCH --job-name=gsota-qcmp
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
R=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/runs
for p in kang2018_b1_c kang2018_b1_a kang2018_b2_stim; do
  echo == $p; ls $R/$p/singlet_mc99cpu_binned/private/donor 2>&1 | head -3
  for f in $R/$p/singlet_mc99cpu_binned/private/donor/donor*.vcf; do b=$(basename $f); cmp -s $f ${f/binned/full} && echo "$b identical" || echo "$b DIFF"; done | sort | uniq -c | head
  tail -4 $R/$p/singlet_mc99cpu_binned/run.log | cut -c1-200
  ls -l $R/$p/singlet_mc99cpu_binned/possorted.bam $R/$p/singlet_mc99cpu_full/possorted.bam
done
grep -E "qual|ILLUMINA|min_base|Q10" $R/kang2018_b1_c/singlet_mc99cpu_binned/run.log | head -5
