#!/bin/bash
#SBATCH --job-name=bl-fetch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Barcode-layer phase 1c prep, step 0: real read subsets for the STAR comparison. The first N spots of each run, read names
# rewritten to r<i>, barcode mate and cDNA mate split by name. Nothing is generated: 10x pbmc1k_v3 from the local copy of the
# public Cell Ranger FASTQs, the other classes streamed from ENA (public) inside this job.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; D=$W/data/barcode-layer; mkdir -p $D
export PATH=$W/envs/pbench/bin:$PATH
trap 'kill $(jobs -p) 2>/dev/null' EXIT INT TERM
echo "node=$(hostname) $(date -Is)"
python -c "import numpy, numba; print('numpy', numpy.__version__, 'numba', numba.__version__)" 2>&1 | tail -1
which samtools STAR curl awk zcat 2>&1
rename() { awk 'NR%4==1{print "@r" (NR-1)/4; next} {print}'; }   # keeps +, seq, qual
# --- 10x pbmc1k_v3, lane 1, first 12 M reads
N=12000000; mkdir -p $D/pbmc1k_v3
FQ=$PB/data/pbmc1k_v3/pbmc_1k_v3_fastqs
for m in R1 R2; do
  gzip -dc $FQ/pbmc_1k_v3_S1_L001_${m}_001.fastq.gz | head -n $((4*N)) | rename > $D/pbmc1k_v3/$m.fq.tmp; mv $D/pbmc1k_v3/$m.fq.tmp $D/pbmc1k_v3/$m.fq
done
echo "pbmc1k_v3: $(( $(wc -l < $D/pbmc1k_v3/R1.fq) / 4 )) reads"; sed -n 1,4p $D/pbmc1k_v3/R1.fq; sed -n 1,4p $D/pbmc1k_v3/R2.fq
# --- ENA subsets
ena() { # name srr nreads
  local name=$1 srr=$2 n=$3 num=${2#SRR} dir
  case ${#num} in 6) dir="";; 7) dir="/00${num: -1}";; 8) dir="/0${num: -2}";; *) dir="/${num: -3}";; esac
  local base=ftp.sra.ebi.ac.uk/vol1/fastq/SRR${num:0:3}$dir/$srr
  mkdir -p $D/$name
  for m in 1 2; do
    set +o pipefail
    curl -sf --retry 3 --max-time 7000 https://$base/${srr}_$m.fastq.gz | gzip -dc 2>/dev/null | head -n $((4*n)) | rename > $D/$name/M$m.fq.tmp
    set -o pipefail
    [ -s $D/$name/M$m.fq.tmp ] && mv $D/$name/M$m.fq.tmp $D/$name/M$m.fq || echo "FAILED $name mate $m ($base)"
  done
  echo "== $name $srr: $(( $(wc -l < $D/$name/M1.fq) / 4 )) / $(( $(wc -l < $D/$name/M2.fq) / 4 )) reads"
  sed -n 2p $D/$name/M1.fq; sed -n 2p $D/$name/M2.fq; awk 'NR%4==2{print length($0)}' $D/$name/M1.fq | head -2000 | sort -n | uniq -c | sort -k1nr | head -3
  awk 'NR%4==2{print length($0)}' $D/$name/M2.fq | head -2000 | sort -n | uniq -c | sort -k1nr | head -3
}
ena dropseq   SRR5747459  6000000
ena bd        SRR12364726 6000000
ena splitseq  SRR31302021 6000000
ena indrop    SRR15409581 900000
ls -la $D/*; echo "done $(date -Is)"
