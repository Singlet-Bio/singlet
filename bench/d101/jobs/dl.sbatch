#!/bin/bash
#SBATCH --job-name=d101-dl
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# D-101 validation data, Oetjen et al. 2018 human bone marrow (GSE120221 10x 3' v2, GSE120446 bulk RNA-seq of the same donors).
# Not in the NM corpus (sprint/intel/corpus-engine-touch.md). Row SLURM_ARRAY_TASK_ID of bench/d101/libs.tsv (1-based, header excluded).
#   KIND=sc   : the run was deposited as a Cell Ranger BAM, so `singlet download --bam <ENA submitted BAM>` -> $W/data/d101/sc/<sc_run>.1fq
#               (barcode read from CR+UR, cDNA from the read; SRA's own FASTQ has the cDNA read only)
#   KIND=bulk : fasterq-dump <bulk_run> (paired, 2 x 150) -> $W/data/d101/bulk/<bulk_run>_{1,2}.fastq.gz
# BIN: any singlet binary with `download` (default: the gpu/frontier build 654109ad). LIBS: path of libs.tsv on Clipper.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench
KIND=${KIND:?sc or bulk}; LIBS=${LIBS:?}; BIN=${BIN:-$W/jobs/gpu-pr2/bin2/654109ad/singlet-gpu}
ROW=$(awk -v n=$SLURM_ARRAY_TASK_ID 'NR==n+1' $LIBS); [ -n "$ROW" ] || { echo "no row $SLURM_ARRAY_TASK_ID"; exit 0; }
read -r DONOR SC SCG BULK BULKG <<< "$ROW"
SCR=/tmp/$USER-d101dl-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
if [ $KIND = sc ]; then
  D=$W/data/d101/sc; mkdir -p $D; O=$D/$SC.1fq
  [ -s $O ] && { echo "have $O"; exit 0; }
  echo "donor $DONOR $SC ($SCG)"
  URL=$(curl -fsS "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$SC&result=read_run&fields=submitted_ftp" | tail -1 | cut -f2 | tr ';' '\n' | grep '\.bam$')
  [ -n "$URL" ] || { echo "no submitted BAM for $SC"; exit 4; }
  # The original BAM (30-40 GB) comes from NCBI's S3 copy of the submitted file when it is there (56 MB/s per stream measured on
  # b002, against 6 MB/s per stream from EBI, which throttles each connection), else from EBI. Either way it is fetched as 8 byte
  # ranges in parallel, each resumable, into project space, so a job killed at the time limit picks up where it stopped.
  ST=$W/data/d101/stage/$SC; P=$ST/parts; mkdir -p $P; BAMN=$(basename $URL); BAM=$ST/$BAMN
  size() { curl -fsSIL "$1" 2>/dev/null | awk '{gsub(/[^[:print:]]/,"")} tolower($1)=="content-length:"{n=$2} END{print n+0}'; }
  NE=$(size https://$URL); SRC=https://$URL
  for k in 1 2; do u=https://sra-pub-src-$k.s3.amazonaws.com/$SC/$BAMN.1; n=$(size $u); [ "$n" -gt 0 ] && [ "$n" = "$NE" ] && { SRC=$u; break; }; done
  [ "$NE" -gt 0 ] || { echo "cannot size $URL"; exit 4; }
  echo "source $SRC ($NE bytes)"
  if [ "$(stat -c %s $BAM 2>/dev/null || echo 0)" != "$NE" ]; then
    rm -f $BAM; N=8; L=$(( (NE + N - 1) / N ))
    part() { local i=$1 lo=$(( $1 * L )) hi=$(( ($1 + 1) * L - 1 )) f=$P/$1 t have
      [ $hi -ge $NE ] && hi=$(( NE - 1 )); t=$(( hi - lo + 1 ))
      for try in $(seq 1 30); do have=$(stat -c %s $f 2>/dev/null || echo 0)
        [ $have -ge $t ] && return 0
        curl -fsSL --retry 3 --retry-all-errors --retry-delay 5 -r $(( lo + have ))-$hi $SRC >> $f || sleep 10
      done; return 1; }
    t0=$(date +%s); pids=""
    for i in $(seq 0 $(( N - 1 ))); do part $i & pids="$pids $!"; done
    ok=1; for p in $pids; do wait $p || ok=0; done
    [ $ok = 1 ] || { echo "range download failed for $SC"; exit 5; }
    cat $(for i in $(seq 0 $(( N - 1 ))); do echo $P/$i; done) > $BAM.part && mv $BAM.part $BAM && rm -rf $P
    echo "fetched $NE bytes in $(( $(date +%s) - t0 )) s"
  fi
  [ "$(stat -c %s $BAM)" = "$NE" ] || { echo "size mismatch for $BAM"; exit 5; }
  ( cd $SCR && $BIN download --bam $BAM --staging-dir $SCR/stage -o $SCR/$SC.1fq ) 2>&1 | tail -25
  [ -s $SCR/$SC.1fq ] || { echo "conversion failed for $SC"; exit 5; }
  mv $SCR/$SC.1fq $O.part && mv $O.part $O && rm -rf $ST
  ls -l $O
else
  [ "$BULK" = "-" ] && { echo "donor $DONOR has no bulk run"; exit 0; }
  D=$W/data/d101/bulk; mkdir -p $D
  [ -s $D/${BULK}_2.fastq.gz ] && { echo "have $BULK"; exit 0; }
  echo "donor $DONOR $BULK ($BULKG)"
  ( cd $SCR && prefetch -q -O $SCR $BULK && fasterq-dump -q -e 6 -t $SCR --split-files -O $SCR $SCR/$BULK/$BULK.sra )
  for m in 1 2; do gzip -1 -c $SCR/${BULK}_$m.fastq > $D/${BULK}_$m.fastq.gz.part && mv $D/${BULK}_$m.fastq.gz.part $D/${BULK}_$m.fastq.gz; done
  ls -l $D/${BULK}_*
fi
