#!/bin/bash
#SBATCH --job-name=bp-nh-krakendb
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch 00_setup_kraken_db.sbatch
# Kraken2 standard database for the non-host benchmark: Ben Langmead's prebuilt PlusPF index capped at 8 GB
# (k2_pluspf_08gb: RefSeq archaea, bacteria, viral, plasmid, human, UniVec_Core, protozoa, fungi). Unlike the viral-only
# database used in the first benchmark, it contains human sequence, so host-derived reads do not classify as viral.
# The newest k2_pluspf_08gb archive in the bucket is taken and its date recorded in $DB/VERSION.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; BP=$W/bench-public
DB=$BP/db/k2_pluspf08; mkdir -p $DB $BP/logs; cd $DB
key=$(curl -s "https://genome-idx.s3.amazonaws.com/?list-type=2&prefix=kraken/k2_pluspf_08gb_" |
      grep -o "kraken/k2_pluspf_08gb_[0-9]*\.tar\.gz" | sort -u | tail -1)
echo "archive: $key"
[ -n "$key" ] || { echo "no archive found"; exit 1; }
curl -fL --retry 5 -o db.tar.gz "https://genome-idx.s3.amazonaws.com/$key"
ls -la db.tar.gz
curl -fsL "https://genome-idx.s3.amazonaws.com/$key.md5" -o db.tar.gz.md5 2>/dev/null || curl -fsL "https://genome-idx.s3.amazonaws.com/${key%.tar.gz}.md5" -o db.tar.gz.md5 || true
if [ -s db.tar.gz.md5 ]; then echo "published md5: $(cat db.tar.gz.md5)"; fi
echo "md5: $(md5sum db.tar.gz | cut -d' ' -f1)"
tar -xzf db.tar.gz
rm -f db.tar.gz
echo "$key" > VERSION
ls -la $DB
