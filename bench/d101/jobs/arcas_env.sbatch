#!/bin/bash
#SBATCH --job-name=d101-arcas-env
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# arcasHLA (Orenbuch et al. 2020), the second Geuvadis comparator for B5: bioconda arcas-hla in its own env, reference built from
# IPD-IMGT/HLA 3.66.0, the release of T1K's index, so both tools type against the same allele set.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/arcashla; MM=$W/tools/mm/bin/micromamba
if [ ! -f $E/.ok ]; then
  rm -rf $E
  $MM create -y -p $E -c conda-forge -c bioconda arcas-hla git git-lfs 2>&1 | tail -3
  export PATH=$E/bin:$PATH
  git lfs install --skip-repo
  arcasHLA reference --version 3.66.0 -v 2>&1 | tail -5
  touch $E/.ok
fi
export PATH=$E/bin:$PATH
arcasHLA --version 2>&1 | head -2 || true; kallisto version
ls $(dirname $(readlink -f $(which arcasHLA)))/../share/arcas-hla*/dat/ref 2>/dev/null | head || true
