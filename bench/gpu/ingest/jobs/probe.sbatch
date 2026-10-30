#!/bin/bash
#SBATCH --job-name=w11-ing-probe
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU-only probe (no GPU gres): public-source sizes (ENA, NCBI SDL), timed ranged downloads, nvCOMP wheel contents.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
OUT=$W/jobs/w11/ingest/probe_$SLURM_JOB_ID; mkdir -p $OUT
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
which curl wget python3 || true
echo "== ENA filereport for known runs"
F="run_accession,experiment_accession,library_layout,read_count,base_count,fastq_bytes,fastq_ftp,submitted_bytes,submitted_format,submitted_ftp,sra_bytes,sra_ftp"
for acc in SRR21102187 SRR21102188; do
  curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$acc&result=read_run&fields=$F&format=tsv" | tee -a $OUT/ena_runs.tsv
done
echo "== GSM -> runs via ENA search"
for g in GSM4006843 GSM5411066 GSM7901264 GSM7036403 GSM6470208 GSM7656149; do
  echo "# $g"
  curl -s "https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&query=experiment_alias%3D%22${g}%22%20OR%20sample_alias%3D%22${g}%22&fields=$F&format=tsv" | tee -a $OUT/ena_gsm.tsv
done
echo "== NCBI SDL"
for acc in SRR21102187 SRR21102188; do
  curl -s "https://locate.ncbi.nlm.nih.gov/sdl/2/retrieve?acc=$acc&filetype=run" > $OUT/sdl_$acc.json; head -c 3500 $OUT/sdl_$acc.json; echo
done
echo "== nvCOMP wheel"
IMG=$W/images/python-3.12-bookworm.sif
apptainer exec -B $W $IMG bash -c "python -m venv $S/v && $S/v/bin/pip install -q nvidia-nvcomp-cu12 2>&1 | tail -3; $S/v/bin/pip list 2>/dev/null | grep -i nvcomp; find $S/v -ipath '*nvcomp*' \( -name '*.h' -o -name '*.hpp' -o -name '*.so*' \) | head -50; du -sh $S/v/lib/python3.12/site-packages/nvidia/nvcomp 2>/dev/null"
echo "== timed ranged downloads"
dl() { # url bytes label
  local t0=$(date +%s.%N)
  curl -s -r 0-$(( $2 - 1 )) -o $S/dl.bin "$1"; local rc=$?
  local t1=$(date +%s.%N); local sz=$(stat -c %s $S/dl.bin)
  echo "DL label=$3 rc=$rc bytes=$sz secs=$(echo "$t1 - $t0" | bc) MBps=$(echo "$sz / ($t1 - $t0) / 1000000" | bc -l)"
  rm -f $S/dl.bin
}
# ENA fastq.gz for the 5' sample's first run (R2/_2 file) and NCBI S3 .sra
FQ2=$(awk -F'\t' 'NR==2{n=split($7,a,";"); print a[n]}' $OUT/ena_runs.tsv)
echo "FQ2=$FQ2"
[ -n "$FQ2" ] && dl "https://$FQ2" 1000000000 ena_fastqgz_http_1GB
SRA_URL="https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR21102187/SRR21102187"
curl -sI "$SRA_URL" | head -8
dl "$SRA_URL" 1000000000 ncbi_s3_sra_1GB
# parallel scaling: 4 ranged streams of 250 MB
t0=$(date +%s.%N)
for i in 0 1 2 3; do curl -s -r $((i*250000000+1000000000))-$((i*250000000+1000000000+249999999)) -o $S/p$i.bin "$SRA_URL" & done; wait
t1=$(date +%s.%N); sz=$(cat $S/p?.bin | wc -c)
echo "DL label=ncbi_s3_sra_4streams bytes=$sz secs=$(echo "$t1 - $t0" | bc) MBps=$(echo "$sz / ($t1 - $t0) / 1000000" | bc -l)"
echo DONE
