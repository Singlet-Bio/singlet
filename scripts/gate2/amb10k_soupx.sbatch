#!/bin/bash
#SBATCH --job-name=amb-10k-soupx
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# SoupX and DecontX on hgmm10k_v3 with the harness' own call (scoring only; same inputs as the 1k rows).
W=/mnt/projects/debruinz_project/debruinz/singlet; A=$W/release-gate/ambient; ID=hgmm10k_v3; D=$A/work/$ID; H=$A/harness
export PATH=$W/envs/ambient/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $D/tools
for t in soupx decontx; do
  [ -f $D/tools/$t.csv ] || /usr/bin/time -v -o $D/tools/$t.time Rscript $H/barnyard_tools.R $t $D 2>&1 | tail -15
  grep -E "Elapsed \(wall|Maximum resident" $D/tools/$t.time
done
ls -la $D/tools
