#!/bin/bash
#SBATCH --job-name=nht-viralref
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Viral reference: NC_045512.2 FASTA + GFF3 downloaded from NCBI in this job; viral GTF + TRS coordinates; viral-only STAR genome.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/nonhost; V=$X/ref/viral
export PATH=$W/envs/pbench/bin:$PATH
mkdir -p $V; cd $V
FTP=https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/858/895/GCF_009858895.2_ASM985889v3/GCF_009858895.2_ASM985889v3_genomic
if curl -fsSL --retry 3 -o g.fna.gz $FTP.fna.gz && curl -fsSL --retry 3 -o g.gff.gz $FTP.gff.gz; then
  zcat g.fna.gz > NC_045512.2.fna; zcat g.gff.gz > NC_045512.2.gff3; echo "source: NCBI FTP GCF_009858895.2"
else
  E=https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
  curl -fsSL "$E?db=nuccore&id=NC_045512.2&rettype=fasta&retmode=text" -o NC_045512.2.fna
  curl -fsSL "$E?db=nuccore&id=NC_045512.2&rettype=gff3&retmode=text" -o NC_045512.2.gff3; echo "source: NCBI efetch"
fi
head -1 NC_045512.2.fna; grep -v '^#' NC_045512.2.gff3 | cut -f3 | sort | uniq -c | sort -rn | head -12; md5sum NC_045512.2.fna
python -I $X/code/viral_ref.py NC_045512.2.fna NC_045512.2.gff3 $V/viral
awk '$3=="gene"' viral.gtf | cut -f1,4,5,7,9
mkdir -p star_viral
STAR --runMode genomeGenerate --runThreadN 4 --genomeDir star_viral --genomeFastaFiles NC_045512.2.fna --sjdbGTFfile viral.gtf \
  --sjdbOverhang 100 --genomeSAindexNbases 4 --outFileNamePrefix star_viral_ > star_viral.log 2>&1; tail -2 star_viral.log
