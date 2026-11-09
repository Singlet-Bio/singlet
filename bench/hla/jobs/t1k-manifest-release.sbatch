#!/bin/bash
#SBATCH --job-name=t1k-rel
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# One-off repair of $REF/hla/t1k/manifest.json: the first setup run recorded the oldest "Rel." line of hla.dat. Re-download hla.dat.zip,
# require the recorded sha256, and record the newest release named in the file (the release the index was built from).
set -euo pipefail
REF=/mnt/projects/debruinz_project/cellarium/reference; M=$REF/hla/t1k/manifest.json; SCR=/tmp/$USER-t1krel-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
curl -fsS --retry 5 -o $SCR/hla.dat.zip https://ftp.ebi.ac.uk/pub/databases/ipd/imgt/hla/hla.dat.zip
SHA=$(sha256sum $SCR/hla.dat.zip | cut -d' ' -f1); WANT=$(grep -o '"hla_dat_zip_sha256": "[0-9a-f]*"' $M | grep -o '[0-9a-f]\{64\}')
[ "$SHA" = "$WANT" ] || { echo "sha differs: $SHA vs $WANT (the release moved on; rerun t1k-setup instead)"; exit 2; }
REL=$(unzip -p $SCR/hla.dat.zip hla.dat | grep -o 'Rel\. [0-9][0-9.]*' | awk '{print $2}' | sed 's/\.$//' | sort -V | tail -1)
echo "release $REL"
sed -i "s|\"reference_release\": \"[^\"]*\"|\"reference_release\": \"IPD-IMGT/HLA $REL\"|" $M
grep -o '"reference_release": "[^"]*"' $M
