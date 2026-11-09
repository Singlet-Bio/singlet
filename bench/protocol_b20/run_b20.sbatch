#!/bin/bash
#SBATCH --job-name=b20-proto
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Launch oracle B20: protocol auto-detection against the GEO chemistry, one library per 10x chemistry of the corpus that the engine has
# already touched (dev, bench, gate and pilot libraries only; libs.tsv), through every input path:
#   enc   singlet encode --reads R1 R2              (FastqEncoder), on the first 400k pairs of the staged FASTQ
#   dlfq  singlet download --r1 R1 --r2 R2           (spot source + SRA encoder), same 400k pairs
#   sra   singlet download <SRR> --max-reads 400000  (VDB + SRA encoder)
# each with $SINGLET_WHITELIST_DIR set and unset (the binaries sit where <binary>/../whitelists does not exist, so "unset" means no
# whitelist at all), for each binary in BINS (name=path, comma-separated). Output: $W/jobs/protocol_b20/<job>/b20.tsv, one row per
# library x path x env x binary: detected tag, confidence, exit code, and PASS when the tag matches the GEO chemistry, ERROR-OK when
# auto-detection stopped for want of whitelists (the intended behaviour without them), FAIL otherwise.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; WL=$W/pipeline-benchmark/whitelists
: "${BINS:?name=path[,name=path]}" "${LIBS:?libs.tsv}"
N=${N_READS:-400000}; O=$W/jobs/protocol_b20/$SLURM_JOB_ID; mkdir -p $O/logs
SCR=/tmp/$USER-b20-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo -e "lib\tgeo_chemistry\tpath\tenv\tbinary\ttag\tconfidence\texit\tresult" > $O/b20.tsv
run() {   # lib geo expect path env binname binpath cmd...
  local lib=$1 geo=$2 exp=$3 path=$4 env=$5 bn=$6 bp=$7; shift 7
  local log=$O/logs/$lib.$path.$env.$bn.log out=$SCR/$lib.$path.$env.$bn.1fq rc tag conf res
  rm -f $out
  ( cd $SCR; if [ $env = set ]; then export SINGLET_WHITELIST_DIR=$WL; else unset SINGLET_WHITELIST_DIR; fi
    timeout 1800 $bp "$@" -o $out ) > $log 2>&1; rc=$?
  tag=$(tr '\r' '\n' < $log | grep -a -o -E "Protocol: [A-Za-z0-9._-]+|protocol[=:] ?[A-Za-z0-9._-]+" | tail -1 | sed -E 's/.*[:=] ?//')
  conf=$(tr '\r' '\n' < $log | grep -a -o -E "confidence: [0-9]+" | tail -1 | grep -o -E "[0-9]+")
  if [ $rc = 0 ] && echo "$tag" | grep -q -E "$exp"; then res=PASS
  elif [ $rc != 0 ] && grep -a -q "auto-detection needs the 10x whitelists" $log; then res=ERROR-OK
  else res=FAIL; fi
  echo -e "$lib\t$geo\t$path\t$env\t$bn\t${tag:--}\t${conf:--}\t$rc\t$res" | tee -a $O/b20.tsv
  rm -f $out
}
tail -n +2 $LIBS | while IFS=$'\t' read -r lib touch geo exp r1 r2 srr; do
  if [ "$r1" != "-" ]; then
    zcat $W/$r1 | head -$((4 * N)) > $SCR/$lib.R1.fastq; zcat $W/$r2 | head -$((4 * N)) > $SCR/$lib.R2.fastq
  fi
  for b in $(echo $BINS | tr , ' '); do bn=${b%%=*}; bp=${b#*=}
    for env in set unset; do
      if [ "$r1" != "-" ]; then
        run $lib "$geo" "$exp" enc $env $bn $bp encode --reads $SCR/$lib.R1.fastq $SCR/$lib.R2.fastq
        run $lib "$geo" "$exp" dlfq $env $bn $bp download --r1 $SCR/$lib.R1.fastq --r2 $SCR/$lib.R2.fastq --staging-dir $SCR/stage-$lib
      fi
      [ "$srr" != "-" ] && run $lib "$geo" "$exp" sra $env $bn $bp download $srr --max-reads $N
    done
  done
  rm -f $SCR/$lib.R1.fastq $SCR/$lib.R2.fastq
done
echo; echo "summary"; awk -F'\t' 'NR>1{n[$5" "$4" "$9]++} END{for(k in n) print k, n[k]}' $O/b20.tsv | sort
echo done $(date -Is)
