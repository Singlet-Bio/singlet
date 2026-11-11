#!/bin/bash
#SBATCH --job-name=b3-stage
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# B3 donor-assignment benchmark data (oracle B3, P3-FRZ-04), staged to $W/data/b3 on one node. Resubmit until it prints "all staged":
# every finished file is skipped and every partial range resumes.
#   souporcell_hipsci  ERP119738 (Heaton et al. 2020): the three HipSci iPSC pools iPSpool_1-3 and the five single-line libraries
#                      babz_3 euts_1 ieki_3 nufh_3 oaqd_2 (each line's own reads: per-donor genotype truth). ENA-origin runs keep
#                      no original file on NCBI, so the source is the SRA Normalized object on NCBI S3 (sra-pub-run-odp), fetched in
#                      8 byte ranges, then fasterq-dump --include-technical. Fallback: the EBI FASTQ pair.
#   neavin2023_village GSE225278 / PRJNA934891 (Neavin et al. 2023), the iPSC_Village GEX runs. ENA serves only the 91 bp cDNA read,
#                      so the source is the submitter's original R1/R2 FASTQ on NCBI S3 (sra-pub-src-K), fetched in 8 byte ranges
#                      and checked against SRA's size and md5. Fallback: the SRA Normalized object with the technical reads.
#   hipsci_arrays      GSE224950 supplementary files (HipSci genotyping arrays of the village lines).
# Genotype data stays owner-only (chmod go-rwx).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; export PATH=$ENV/bin:$PATH
D=$W/data/b3; mkdir -p $D; chmod 700 $D; M=$D/manifest; mkdir -p $M
P=${PAR:-6}; T=${SLURM_CPUS_PER_TASK:-28}
SCR=/tmp/$USER-b3-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "node $(hostname) $(date -Is)"
size() { { curl -fsSIL "$1" 2>/dev/null || true; } | awk '{gsub(/[^[:print:]]/,"")} tolower($1)=="content-length:"{n=$2} END{print n+0}'; }
# fetch URL DEST SIZE: 8 resumable byte ranges in project space, then concatenated; exit 0 only when DEST has SIZE bytes
fetch() {
  local url=$1 dst=$2 n=$3 pd=$2.parts L i pids="" ok=1
  [ "$(stat -c %s $dst 2>/dev/null || echo 0)" = "$n" ] && return 0
  rm -f $dst; mkdir -p $pd; L=$(( (n + 7) / 8 ))
  for i in 0 1 2 3 4 5 6 7; do
    ( lo=$(( i * L )); hi=$(( (i + 1) * L - 1 )); [ $hi -ge $n ] && hi=$(( n - 1 )); t=$(( hi - lo + 1 )); f=$pd/$i
      [ $lo -ge $n ] && exit 0
      for try in $(seq 1 30); do have=$(stat -c %s $f 2>/dev/null || echo 0); [ $have -ge $t ] && exit 0
        curl -fsSL --retry 3 --retry-all-errors --retry-delay 5 -r $(( lo + have ))-$hi "$url" >> $f || sleep 10; done; exit 1 ) &
    pids="$pids $!"
  done
  for i in $pids; do wait $i || ok=0; done
  [ $ok = 1 ] || return 1
  cat $(for i in 0 1 2 3 4 5 6 7; do [ -f $pd/$i ] && echo $pd/$i; done) > $dst.tmp && mv $dst.tmp $dst && rm -rf $pd
  [ "$(stat -c %s $dst)" = "$n" ]
}
export -f fetch size

# ---- manifests (built once on the node; SRA XML is the source of sizes, md5 and original names) ----
if [ ! -s $M/soup.tsv ]; then
  curl -fsS "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=ERP119738&result=read_run&fields=run_accession,sample_title,scientific_name,read_count,fastq_ftp,fastq_bytes&format=tsv&limit=0" \
    | awk -F'\t' 'NR>1 && $3=="Homo sapiens" && $2 ~ /^(iPSpool_[123]|babz_3|euts_1|ieki_3|nufh_3|oaqd_2)$/' > $M/soup.ena.tsv
  : > $M/soup.tsv.tmp
  while IFS=$'\t' read -r run samp org nr fq fb; do
    x=$(curl -fsS "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=$run&rettype=xml")
    u=$(echo "$x" | grep -o "<SRAFile [^>]*semantic_name=\"SRA Normalized\"[^>]*>" | grep "filename=\"$run\"" | grep -o 'url="[^"]*"' | cut -d'"' -f2 | head -1)
    s=$(echo "$x" | grep -o "<SRAFile [^>]*semantic_name=\"SRA Normalized\"[^>]*>" | grep "filename=\"$run\"" | grep -o ' size="[0-9]*"' | grep -o '[0-9]*' | head -1)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$run" "$samp" "${u:--}" "${s:-0}" "$fq" "$fb" >> $M/soup.tsv.tmp; sleep 0.4
  done < $M/soup.ena.tsv
  mv $M/soup.tsv.tmp $M/soup.tsv
fi
if [ ! -s $M/village.tsv ]; then
  curl -fsS "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA934891&result=read_run&fields=run_accession,sample_title&format=tsv&limit=0" \
    | awk -F'\t' 'NR>1 && $2 ~ /^iPSC_Village/' > $M/village.ena.tsv
  : > $M/village.tsv.tmp
  while IFS=$'\t' read -r run samp; do
    x=$(curl -fsS "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=$run&rettype=xml" | tr '\n' ' ' | sed 's#<RUN #\n<RUN #g' | grep "accession=\"$run\"")
    nu=$(echo "$x" | grep -o "<SRAFile [^>]*filename=\"$run\"[^>]*>" | grep -o 'url="[^"]*"' | cut -d'"' -f2 | head -1)
    ns=$(echo "$x" | grep -o "<SRAFile [^>]*filename=\"$run\"[^>]*>" | grep -o ' size="[0-9]*"' | grep -o '[0-9]*' | head -1)
    echo "$x" | grep -o '<SRAFile [^>]*supertype="Original"[^>]*>' | grep -E '_R[12]_[0-9]+\.fastq\.gz"' | while read -r f; do
      fn=$(echo "$f" | grep -o 'filename="[^"]*"' | cut -d'"' -f2); fs=$(echo "$f" | grep -o ' size="[0-9]*"' | grep -o '[0-9]*'); md=$(echo "$f" | grep -o 'md5="[^"]*"' | cut -d'"' -f2)
      src=$(echo "$x" | grep -o "s3://sra-pub-src-[0-9]*/$run/$fn\.1" | head -1)
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$run" "$samp" "$fn" "$fs" "$md" "${src:--}" "${nu:--}" "${ns:-0}"
    done >> $M/village.tsv.tmp; sleep 0.4
  done < $M/village.ena.tsv
  mv $M/village.tsv.tmp $M/village.tsv
fi
echo "souporcell runs $(wc -l < $M/soup.tsv) ($(awk -F'\t' '{s+=$4} END{printf "%.1f GB", s/1e9}' $M/soup.tsv) SRA Normalized)"
echo "village files $(wc -l < $M/village.tsv) ($(awk -F'\t' '{s+=$4} END{printf "%.1f GB", s/1e9}' $M/village.tsv) original R1/R2)"

# ---- souporcell: SRA Normalized from S3 -> fasterq-dump; EBI FASTQ pair when S3 does not match ----
soup_one() {
  local run=$1 samp=$2 url=$3 n=$4 fq=$5 fb=$6 o=$D/souporcell_hipsci/$2 s
  mkdir -p $o; [ -s $o/${run}_2.fastq.gz ] && return 0
  local sra=$o/$run.sra src=none
  if [ "$url" != "-" ] && [ "$n" -gt 0 ] && [ "$(size $url)" = "$n" ] && fetch $url $sra $n; then src=s3
    local tmp=/tmp/$USER-b3-$SLURM_JOB_ID/$run; mkdir -p $tmp
    if fasterq-dump -q -e 4 -t $tmp --split-files --include-technical -O $tmp $sra; then
      local k=1; for f in $(ls $tmp/${run}_*.fastq | sort -V); do gzip -1 -c $f > $o/${run}_$k.fastq.gz.tmp && mv $o/${run}_$k.fastq.gz.tmp $o/${run}_$k.fastq.gz; k=$((k+1)); done
      rm -rf $tmp $sra
    else rm -rf $tmp; src=none; fi
  fi
  if [ "$src" = none ]; then src=ebi; local k=1
    for u in $(echo $fq | tr ';' ' '); do s=$(echo $fb | cut -d';' -f$k); fetch https://$u $o/${run}_$k.fastq.gz $s || { echo "FAIL $run"; return 1; }; k=$((k+1)); done
  fi
  echo "$run $samp $src $(du -cb $o/${run}_*.fastq.gz | tail -1 | cut -f1)"
}
export -f soup_one; export D SLURM_JOB_ID USER
# ---- village: original R1/R2 from sra-pub-src-K, md5 checked; SRA Normalized + fasterq-dump when it does not match ----
vil_one() {
  local run=$1 samp=$2 fn=$3 n=$4 md=$5 src=$6 nu=$7 ns=$8 o=$D/neavin2023_village/$2 u
  mkdir -p $o; [ -s $o/$fn ] && [ -f $o/$fn.ok ] && return 0
  u=$(echo "$src" | sed -E 's#^s3://([^/]+)/#https://\1.s3.amazonaws.com/#')
  if [ "$src" != "-" ] && [ "$(size $u)" = "$n" ] && fetch $u $o/$fn $n && [ "$(md5sum < $o/$fn | cut -c1-32)" = "$md" ]; then
    touch $o/$fn.ok; echo "$run $fn s3 $n"; return 0
  fi
  rm -f $o/$fn; echo "$run $fn S3 mismatch: falling back to SRA Normalized"
  [ -s $o/${run}_2.fastq.gz ] && return 0
  local tmp=/tmp/$USER-b3-$SLURM_JOB_ID/$run; mkdir -p $tmp
  fetch $nu $tmp/$run.sra $ns && fasterq-dump -q -e 4 -t $tmp --split-files --include-technical -O $tmp $tmp/$run.sra \
    && { k=1; for f in $(ls $tmp/${run}_*.fastq | sort -V); do gzip -1 -c $f > $o/${run}_$k.fastq.gz; k=$((k+1)); done; } || echo "FAIL $run $fn"
  rm -rf $tmp
}
export -f vil_one
t0=$(date +%s)
tr '\t' ' ' < $M/soup.tsv | xargs -P $P -L 1 bash -c 'soup_one "$@"' _ > $D/souporcell.log 2>&1
echo "souporcell pass $(( $(date +%s) - t0 )) s, failures $(grep -c FAIL $D/souporcell.log)"; t0=$(date +%s)
tr '\t' ' ' < $M/village.tsv | xargs -P $P -L 1 bash -c 'vil_one "$@"' _ > $D/village.log 2>&1
echo "village pass $(( $(date +%s) - t0 )) s, failures $(grep -c FAIL $D/village.log), S3 mismatches $(grep -c mismatch $D/village.log)"

# ---- HipSci arrays of the village lines (GEO supplementary) ----
G=$D/hipsci_arrays; mkdir -p $G
if [ ! -f $G/.done ]; then
  base=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE224nnn/GSE224950/suppl/
  for f in $(curl -fsS $base | grep -o 'href="[^"]*"' | cut -d'"' -f2 | grep -v -E '^\?|^/|/$'); do
    curl -fsSL --retry 3 -C - -o $G/$f $base$f && echo "array file $f $(stat -c %s $G/$f)"
  done; touch $G/.done
fi

# ---- completeness ----
ns=$(awk -F'\t' '{print $2"/"$1"_2.fastq.gz"}' $M/soup.tsv | while read f; do [ -s $D/souporcell_hipsci/$f ] || echo x; done | wc -l)
nv=$(awk -F'\t' '{print $2"/"$3}' $M/village.tsv | while read f; do [ -f $D/neavin2023_village/$f.ok ] || [ -n "$(ls $D/neavin2023_village/$(dirname $f)/*_2.fastq.gz 2>/dev/null)" ] || echo x; done | wc -l)
du -sh $D/souporcell_hipsci $D/neavin2023_village $G 2>/dev/null
chmod -R go-rwx $D
[ $ns = 0 ] && [ $nv = 0 ] && echo "all staged $(date -Is)" || echo "missing: souporcell $ns runs, village $nv files: resubmit"
