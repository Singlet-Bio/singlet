#!/bin/bash
#SBATCH --job-name=hto-geo
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:28:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Authors' deposited tables of Stoeckius 2018 (GSE108313), by Range requests.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/adt-bench/bin:$PATH
G=https://ftp.ncbi.nlm.nih.gov/geo/samples
python -P $W/expression-sota/hto/code/bench/expression_sota/hto/geo_fetch.py $W/expression-sota/hto/data/stoeckius2018/geo \
  $G/GSM2895nnn/GSM2895283/suppl/GSM2895283_Hashtag-HTO-count.csv.gz \
  $G/GSM3501nnn/GSM3501447/suppl/GSM3501447_MixCellLines-HTO-counts.csv.gz \
  $G/GSM3501nnn/GSM3501446/suppl/GSM3501446_MixCellLines-RNA.umi.txt.gz
cd $W/expression-sota/hto/data/stoeckius2018/geo; ls -l
for f in *.gz; do echo "-- $f"; zcat $f | head -3 | cut -c1-300; zcat $f | wc -l; done
