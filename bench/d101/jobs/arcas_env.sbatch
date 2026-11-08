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
# arcasHLA 0.6.0's version table stops long before 3.66.0, so the release is checked out by its commit (tag v3.66.0-alpha of
# ANHIG/IMGTHLA, the release T1K's index was built from).
IMGT_3660=d972ae287948dc2e5468a40ff3b3d27227541189
if [ ! -f $E/.ok ]; then
  [ -x $E/bin/arcasHLA ] || { rm -rf $E; $MM create -y -p $E -c conda-forge -c bioconda arcas-hla git git-lfs 2>&1 | tail -3; }
  export PATH=$E/bin:$PATH
  git lfs install --skip-repo
  arcasHLA reference --commit $IMGT_3660 -v 2>&1 | tail -8 || true
  ls $E/share/arcas-hla*/dat/ref/ | head
  if ls $E/share/arcas-hla*/dat/ref/*.idx >/dev/null 2>&1; then touch $E/.ok; else echo "arcasHLA reference not built; Geuvadis runs go ahead without it"; fi
fi
export PATH=$E/bin:$PATH
arcasHLA --version 2>&1 | head -2 || true; kallisto version
ls $(dirname $(readlink -f $(which arcasHLA)))/../share/arcas-hla*/dat/ref 2>/dev/null | head || true
