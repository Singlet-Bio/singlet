#!/bin/bash
#SBATCH --job-name=hto-fetch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Independent HTO set: Stoeckius et al. 2018 (GEO GSE108313, public data, cite the paper).
#  (a) the authors' deposited HTO count tables and the mixed-cell-line RNA / HTO tables (small; resume loop because the GEO
#      mirror closes connections early);
#  (b) PBMC 8-donor hashing RNA reads, first 200 M read pairs of SRR8281306 (ENA), and the HTO library reads SRR8281307.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
D=$W/expression-sota/hto/data/stoeckius2018; mkdir -p $D/geo $D/fastq; cd $D
getr() { for i in $(seq 40); do curl -fsS --http1.1 -C - -A "Mozilla/5.0" -o "$2" "$1" && return 0; sleep 3; done; return 1; }
G=https://ftp.ncbi.nlm.nih.gov/geo/samples
getr $G/GSM2895nnn/GSM2895283/suppl/GSM2895283_Hashtag-HTO-count.csv.gz geo/GSM2895283_Hashtag-HTO-count.csv.gz; echo "hto rc=$?"
getr $G/GSM3501nnn/GSM3501447/suppl/GSM3501447_MixCellLines-HTO-counts.csv.gz geo/GSM3501447_MixCellLines-HTO-counts.csv.gz; echo "clhto rc=$?"
getr $G/GSM3501nnn/GSM3501446/suppl/GSM3501446_MixCellLines-RNA.umi.txt.gz geo/GSM3501446_MixCellLines-RNA.umi.txt.gz; echo "clrna rc=$?"
ls -l geo; for f in geo/*; do echo "-- $f"; zcat $f | head -3 | cut -c1-250; done
E=https://ftp.sra.ebi.ac.uk/vol1/fastq
N=200000000
if [ ! -e fastq/rna.done ]; then
  for r in 1 2; do
    ( curl -fsS --retry 5 --retry-all-errors $E/SRR828/006/SRR8281306/SRR8281306_$r.fastq.gz | pigz -dc | head -n $((4*N)) | pigz -p 3 > fastq/rna_R$r.fastq.gz ) &
  done; wait; touch fastq/rna.done
fi
if [ ! -e fastq/hto.done ]; then
  for r in 1 2; do ( curl -fsS --retry 5 --retry-all-errors $E/SRR828/007/SRR8281307/SRR8281307_$r.fastq.gz -o fastq/hto_R$r.fastq.gz ) & done; wait; touch fastq/hto.done
fi
ls -l fastq; for f in fastq/rna_R1.fastq.gz fastq/rna_R2.fastq.gz fastq/hto_R1.fastq.gz fastq/hto_R2.fastq.gz; do echo "-- $f"; pigz -dc $f 2>/dev/null | head -8; done
for f in fastq/rna_R1.fastq.gz fastq/rna_R2.fastq.gz; do echo "$f reads: $(pigz -dc $f | wc -l)"; done
