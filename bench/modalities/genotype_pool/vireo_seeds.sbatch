#!/bin/bash
#SBATCH --job-name=gp-vireo2
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch vireo_seeds.sbatch <singlet-tag>
# vireo is sensitive to initialisation: rerun with more restarts and several seeds on the same cellsnp pileup,
# then pick by the final ELBO (no truth used for the pick).
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/modality-bench/genotype_pool; E=$W/envs/gpool; O=$D/ref_$1
cd $O
for s in 1 2 3 4 5; do
  $E/venv/bin/vireo -c $O/cellsnp -o $O/vireo_s$s -N 8 -p 8 --randSeed $s --nInit 200 > vireo_s$s.log 2>&1
  echo "seed $s: $(grep -i 'ELBO\|LB' vireo_s$s.log | tail -1)"
done
