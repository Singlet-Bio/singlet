#!/bin/bash
#SBATCH --job-name=s8-ugap
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: per-gene U gap (Singlet vs STARsolo-Velocyto and alevin-fry) on the gate-2 spliced-unspliced artifact of build 0cd56886, pbmc1k_cite_v3 and pbmc1k_v3.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PY=/mnt/projects/debruinz_project/debruinz/envs/gt/bin/python; REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
mkdir -p $W/s8/ugap
for ds in pbmc1k_cite_v3 pbmc1k_v3; do
  echo "=== $ds"; $PY -P $W/s8/musc/code/s8_ugap.py $W/release-gate/fixcounts/0cd56886/$ds/art/spliced-unspliced/$ds.genes.tsv.gz $REF/genes/genes.gtf $W/s8/ugap/$ds.json
done
