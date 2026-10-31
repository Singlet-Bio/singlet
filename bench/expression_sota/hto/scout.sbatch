#!/bin/bash
#SBATCH --job-name=hto-scout
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Scout 2: GEO sample supplementary download routes, ENA run listing for the Stoeckius 2018 libraries.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
D=$W/expression-sota/hto/data/stoeckius2018; mkdir -p $D; cd $D
G=GSM2895283; N=${G}_Hashtag-HTO-count.csv.gz
S=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2895nnn/$G/suppl/$N
echo "== curl --http1.1";    curl -fsS --http1.1 -A "Mozilla/5.0" -o t1.gz $S; echo rc=$?; ls -l t1.gz
echo "== wget";              wget -q -O t2.gz $S; echo rc=$?; ls -l t2.gz
echo "== geo download?acc";  curl -fsSL -A "Mozilla/5.0" -o t3.gz "https://www.ncbi.nlm.nih.gov/geo/download/?acc=$G&format=file&file=$N"; echo rc=$?; ls -l t3.gz
echo "== ENA filereport"
for r in SRR8281306 SRR8281307 SRR8281308; do
  curl -fsSL "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$r&result=read_run&fields=run_accession,experiment_title,library_layout,read_count,fastq_bytes,fastq_ftp,sample_accession&format=tsv"
done
echo "== 10x CellPlex"
curl -fsSI "https://cf.10xgenomics.com/samples/cell-exp/6.0.0/SC3_v3_NextGem_DI_CellPlex_Human_PBMC_10K_Multiplex/SC3_v3_NextGem_DI_CellPlex_Human_PBMC_10K_Multiplex_fastqs.tar" | head -5
for f in t1.gz t2.gz t3.gz; do [ -s $f ] && { echo "-- $f"; zcat $f 2>/dev/null | head -3 | cut -c1-200; }; done
