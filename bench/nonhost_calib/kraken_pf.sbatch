#!/bin/bash
#SBATCH --job-name=nh-kraken-pf
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# The registered comparator of the non-host output (claims-to-specs T31): Kraken2 with PlusPF k2_pluspf_20261022, confidence 0.1, the first
# 50 M reads of the cDNA mate, on the same development libraries as run_dev.sbatch. The database stays memory-mapped, so it is read once.
# Writes $J/kraken/<id>/report.txt. Also records the database file sizes and md5 of the small files in $J/kraken/database.manifest.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/nhcal; PB=$W/pipeline-benchmark; OLD=$W/modality-bench/sj_nonhost
KDB=$W/data/kraken_db/k2_pluspf_20261022; K2=$W/envs/kraken2/bin/kraken2
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR $J/kraken; trap 'rm -rf $SCR' EXIT
[ -s $KDB/hash.k2d ] || { echo "no database at $KDB"; exit 6; }
{ echo "database $KDB (k2_pluspf_20261022, downloaded $(cat $KDB/DOWNLOADED))"; ls -l $KDB | awk '{print $5, $9}'; (cd $KDB && md5sum opts.k2d taxo.k2d ktaxonomy.tsv seqid2taxid.map library_report.tsv names.dmp nodes.dmp 2>/dev/null); $K2 --version | head -1; } > $J/kraken/database.manifest
( md5sum $KDB/hash.k2d > $J/kraken/hash.k2d.md5 ) &   # 125 GB: runs while the libraries are classified
cat > $SCR/libs.txt <<EOF
ravindra2021_dpi2 pair:$OLD/data/dpi2/R2.fastq.gz
ravindra2021_mock pair:$OLD/data/mock/R2.fastq.gz
ravindra2021_dpi1 pair:$J/data/ravindra2021_dpi1/R2.fastq.gz
lcl_gse230239_HG01277 pair:$J/data/lcl_gse230239_HG01277/R2.fastq.gz
lcl_gse230239_HG00362 pair:$J/data/lcl_gse230239_HG00362/R2.fastq.gz
pbmc1k_v3 fq:$PB/data/pbmc1k_v3/gex_fastqs.txt
pbmc1k_v2 fq:$PB/data/pbmc1k_v2/gex_fastqs.txt
pbmc5p_v1 fq:$PB/data/pbmc5p_v1/gex_fastqs.txt
hgmm1k_v3 fq:$PB/data/hgmm1k_v3/gex_fastqs.txt
neuron1k_v3 fq:$PB/data/neuron1k_v3/gex_fastqs.txt
nuclei2k_v2 fq:$PB/data/nuclei2k_v2/gex_fastqs.txt
kang2018_ctrl pair:$W/modality-bench/genotype_pool/fastq/R2.fastq.gz
EOF
while read -r ID SRC; do
  O=$J/kraken/$ID; [ -s $O/report.txt ] && continue; mkdir -p $O
  case $SRC in
    fq:*) L=${SRC#fq:}; cat $(grep "_R2_" $L) | zcat 2>/dev/null | head -n 200000000 > $SCR/r2.fq ;;
    pair:*) zcat ${SRC#pair:} 2>/dev/null | head -n 200000000 > $SCR/r2.fq ;;
  esac
  n=$(awk 'END{print NR/4}' $SCR/r2.fq)
  /usr/bin/time -v -o $O/time.txt $K2 --db $KDB --threads $(nproc) --confidence 0.1 --memory-mapping --report $O/report.txt --output /dev/null $SCR/r2.fq > $O/kraken.log 2>&1
  echo "$n" > $O/reads.txt; echo "[kraken] $ID reads=$n $(grep -E 'sequences classified' $O/kraken.log | head -1) $(date -Is)"
  rm -f $SCR/r2.fq
done < $SCR/libs.txt
wait; cat $J/kraken/hash.k2d.md5
