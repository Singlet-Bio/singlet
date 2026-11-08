#!/bin/bash
# Pack, checksum and publish the portable bundle's reference tars on data.singlet.bio/reference/$VER/ (publish_ref.py).
#   sbatch --export=ALL,STAGE=pack publish_ref.sbatch      tars, sha256 of every member and tar, manifest.json (no upload)
#   sbatch --export=ALL,STAGE=upload publish_ref.sbatch    uploads the tars and manifest.json (refuses to overwrite other content)
# CPU only. R2 credentials come from a 600-mode file into the environment, never onto a command line.
#SBATCH --job-name=pub-ref
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; R=/mnt/projects/debruinz_project/cellarium/reference
VER=${VER:-v1}; OUT=${OUT:-$W/data/publish/reference-$VER}; STAGE=${STAGE:?pack or upload}
HERE=${HERE:-$W/jobs/gpu-campaign/publish}
mkdir -p $OUT; cd $OUT
TARF=(--sort=name --owner=0 --group=0 --numeric-owner --mtime=2026-11-07T00:00:00Z --format=gnu)
NH="nonhost/viral.snhskidx nonhost/bacterial.snhskidx nonhost/fungal.snhskidx nonhost/host_21mer.bloom"
if [ $STAGE = pack ]; then
  echo "pack $(date -Is) on $(hostname)"
  [ -e star_2.7.11b-GRCh38-2024-A.tar ] || { tar "${TARF[@]}" -C $R/GRCh38-2024-A -cf star_2.7.11b-GRCh38-2024-A.tar.part star_2.7.11b && mv star_2.7.11b-GRCh38-2024-A.tar.part star_2.7.11b-GRCh38-2024-A.tar; } &
  [ -e nonhost-refseq223.tar ] || { tar "${TARF[@]}" -C $R -cf nonhost-refseq223.tar.part $NH && mv nonhost-refseq223.tar.part nonhost-refseq223.tar; } &
  ( cd $R/GRCh38-2024-A && ls star_2.7.11b/* | xargs -P 8 -n 1 sha256sum | sort -k2 ) > star_2.7.11b-GRCh38-2024-A.members.sha256 &
  ( cd $R && echo $NH | tr ' ' '\n' | xargs -P 4 -n 1 sha256sum | sort -k2 ) > nonhost-refseq223.members.sha256 &
  ( cd $R/GRCh38-2024-A && sha256sum fasta/genome.fa genes/genes.gtf genes/genes.gtf.gz ) > inputs.sha256 &
  wait
  ( cd $R/GRCh38-2024-A && stat -c '%s %n' star_2.7.11b/* ) > star_2.7.11b-GRCh38-2024-A.members.size
  ( cd $R && stat -c '%s %n' $NH ) > nonhost-refseq223.members.size
  sha256sum star_2.7.11b-GRCh38-2024-A.tar nonhost-refseq223.tar > SHA256SUMS
  for t in star_2.7.11b-GRCh38-2024-A.tar nonhost-refseq223.tar; do echo "$t members: $(tar -tf $t | grep -vc '/$')"; done
  echo "## recorded sha256 (state/reference-manifest-v1.yaml) against the files"
  for p in "nonhost/viral.snhskidx ab42272227d49f974ce2e0ba47fa23aa813b7fb607b6a15999c60c318f2c30ae" \
           "nonhost/bacterial.snhskidx b32534e489a07408b6cb22a0c66ce718d1b2da8825de91976b6ce69070dc13ff" \
           "nonhost/fungal.snhskidx 36e2e9a31b93cb581aa2ff35b12c44d2b4385d410e2e752f55e2912ad45c0dac" \
           "nonhost/host_21mer.bloom 6f7f7895ed26e4d5149838a30e0ad0af39d45a45c6f9a625b45f2938d1980128" \
           "star_2.7.11b/Genome 9d97df99c23bc0d84323ece5bac1c72200343eab459a767786c14eff7ecd9d4c" \
           "star_2.7.11b/SA b6b8f6d68c5f879dbf84ac045deda8a4ccc970ee1de65ea7abcff3afbe3448a0" \
           "star_2.7.11b/SAindex 7afc61a12592d995f15431b1d0eef0caed6043933563dc53aa4cdd236b26e10a" \
           "fasta/genome.fa fb7421217e7058120cd60a5277445198e8deef2dff1edc46cd1e98b31fe64cbb"; do
    set -- $p; got=$(cat *.members.sha256 inputs.sha256 | awk -v f="$1" '$2==f{print $1}')
    [ "$got" = "$2" ] && echo "OK $1" || echo "MISMATCH $1 recorded $2 file ${got:-missing}"
  done
fi
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
export PYTHONPATH=$W/venvs/annotate-extra
if [ $STAGE = pack ]; then in_runtime python3 $HERE/publish_ref.py manifest --out $OUT --version $VER; cat SHA256SUMS; ls -la $OUT; fi
if [ $STAGE = upload ]; then
  [ -f "$HOME/.config/singlet-r2.env" ] && source "$HOME/.config/singlet-r2.env"
  export R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY ${R2_ENDPOINT:+R2_ENDPOINT}
  echo "upload $(date -Is)"; in_runtime python3 -u $HERE/publish_ref.py upload --out $OUT --version $VER; echo "done $(date -Is)"
fi
