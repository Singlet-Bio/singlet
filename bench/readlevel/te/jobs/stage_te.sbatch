#!/bin/bash
#SBATCH --job-name=te-stage
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Stage the TE annotations that --pipeline uses by default: TEtranscripts rmsk GTFs (Tam and Hammell, Zenodo record 18927862) into
# <ref-base>/pileup_indices/te/. The md5 sums come from the Zenodo record's API.
set -eu
R=/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/te; mkdir -p $R; cd $R
curl -sSfL https://zenodo.org/api/records/18927862 -o record.json
for f in GRCh38_GENCODE_rmsk_TE.gtf.gz GRCm39_GENCODE_rmsk_TE.gtf.gz; do
  [ -s $f ] || curl -sSfL -o $f.part "https://zenodo.org/records/18927862/files/$f?download=1" && mv -f $f.part $f 2>/dev/null || true
  want=$(python3 -c "import json,sys; r=json.load(open('record.json')); print([x['checksum'].split(':')[1] for x in r['files'] if x['key']=='$f'][0])")
  got=$(md5sum $f | cut -d' ' -f1); echo "$f want $want got $got"; [ "$want" = "$got" ] || { echo "MD5 MISMATCH $f"; exit 1; }
  zcat $f | head -2 || true; echo "lines $(zcat $f | wc -l) subfamilies $(zcat $f | grep -o 'gene_id "[^"]*"' | sort -u | wc -l)"
done
md5sum *.gtf.gz > MD5SUMS; ls -l
