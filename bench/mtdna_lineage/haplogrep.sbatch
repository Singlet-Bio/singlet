#!/bin/bash
#SBATCH --job-name=mtl-hgrep
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Haplogrep 3 (PhyloTree 17 FU1) on the pseudobulk homoplasmic VCFs that analyze.py writes (private outputs).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; MM=$W/tools/mm/bin/micromamba; export MAMBA_ROOT_PREFIX=$W/tools/mm/root
E=$W/envs/mtl; H=$W/tools/haplogrep3; R=$W/out/mtdna_lineage
[ -x $E/bin/java ] || $MM install -y -p $E -c conda-forge openjdk=17
if [ ! -e $H/haplogrep3 ]; then
  mkdir -p $H && cd $H
  curl -sL -o hg.zip https://github.com/genepi/haplogrep3/releases/download/v3.2.2/haplogrep3-3.2.2-linux.zip
  unzip -o -q hg.zip && rm hg.zip
fi
export PATH=$E/bin:$PATH
cd $R/private_vcf
for v in *.vcf; do
  $H/haplogrep3 classify --input $v --output ${v%.vcf}.hg.txt --tree phylotree-fu-rcrs@1.2 --extend-report > /dev/null 2>&1 \
    || $H/haplogrep3 classify --input $v --output ${v%.vcf}.hg.txt --tree phylotree-rcrs@17.2 > /dev/null 2>&1 || echo "failed $v"
  echo "== $v $(grep -vc '^#' $v) variants"; cut -f1-4 ${v%.vcf}.hg.txt 2>/dev/null | head -3
done
