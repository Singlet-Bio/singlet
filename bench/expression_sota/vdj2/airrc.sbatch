#!/bin/bash
#SBATCH --job-name=vdj2-airrc
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# AIRR-C human IG germline reference sets (OGRDB, CC0 1.0) -> V genes the GRCh38 annotation lacks, added to both references.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
BRANCH=${BRANCH:-bench/vdj-sota}
source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome
A=$WORK/data/vdjc/airrc; mkdir -p $A
O=$WORK/expression-sota/vdj2/ref; mkdir -p $O
B=https://ogrdb.airr-community.org
for s in IGH_VDJ IGKappa_VJ IGLambda_VJ; do
  curl -fsSL --retry 5 -o $A/$s.fasta "$B/download_germline_set/Homo%20sapiens/$s/published/ungapped_ex"
  echo "$s $(grep -c '>' $A/$s.fasta) records sha256 $(sha256sum $A/$s.fasta | cut -c1-16)"
done
curl -fsSL --retry 5 "$B/render_page/licensing_statement.html" -o $A/licensing_statement.html
echo "licence page:"; sed 's/<[^>]*>//g' $A/licensing_statement.html | grep -i -m3 "licen"
curl -fsSL --retry 5 "$B/api/germline/sets/Homo%20sapiens" -o $A/sets.json; head -c 1500 $A/sets.json; echo
grep -h '>IGHV1-8\*\|>IGHV4-30-4\*' $A/*.fasta
V="AIRR-C Homo sapiens IGH_VDJ / IGKappa_VJ / IGLambda_VJ published sets downloaded $(date -u +%F) (OGRDB ungapped_ex)"
python -P $SRC/bench/expression_sota/vdj2/add_airrc.py $SRC/python/singlet/vdj/data/vdj_human.tsv.gz $O/vdj_human.tsv.gz \
  $A/IGH_VDJ.fasta $A/IGKappa_VJ.fasta $A/IGLambda_VJ.fasta --version "$V"
python -P $SRC/bench/expression_sota/vdj2/add_airrc.py $SRC/python/singlet/vdj/data/vdj_human_imgtcys.tsv.gz $O/vdj_human_imgtcys.tsv.gz \
  $A/IGH_VDJ.fasta $A/IGKappa_VJ.fasta $A/IGLambda_VJ.fasta --version "$V" --first-cys
ls -la $O
