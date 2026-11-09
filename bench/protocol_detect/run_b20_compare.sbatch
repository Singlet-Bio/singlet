#!/bin/bash
#SBATCH --job-name=detect-b20
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Regression of the B20 probe on the libraries of an earlier run: the picks file of that run (PICKS), minus every accession in EXCLUDE, goes through
# each binary in BINS and the calls are compared (compare_b20.py). Same head fetch and detection as bench/protocol_b20/scale/run_scale.sbatch.
# Original header: B20 at scale, detection only (no engine run on these samples): ~300 Beta-corpus samples stratified by stated assay (pick.py), the first
# 20,000 read pairs of one gene-expression run each (HTTP range request on the ENA FASTQ: only the head of each file is read), then
# `singlet download --r1 --r2` (the SRA encoder's detector, as the campaign runs it) with the whitelists, for each binary in BINS.
# Output $W/jobs/protocol_b20/scale-<job>/: picks.tsv, calls.tsv (one row per run x binary, with the evidence), logs/.
# Needs BINS (name=path,...) and SRC (the bench/protocol_b20/scale directory on Clipper).
set -uo pipefail
: "${BINS:?}" "${SRC:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; PY=$W/envs/pbench/bin/python3
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
O=$W/jobs/protocol_b20/scale-$SLURM_JOB_ID; mkdir -p $O/logs; SCR=/tmp/$USER-b20s-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
: "${PICKS:?}"; EXCLUDE=${EXCLUDE:-$W/jobs/detect/heldout_exclude.txt}
head -1 $PICKS > $O/picks.tsv; tail -n +2 $PICKS | grep -v -F -w -f $EXCLUDE >> $O/picks.tsv; echo "picks: $(( $(wc -l < $O/picks.tsv) - 1 )) of $(( $(wc -l < $PICKS) - 1 ))"
export O SCR BINS
one() {   # sample study assay run instrument urls odp
  local sample=$1 study=$2 assay=$3 run=$4 urls=$6 d=$SCR/$4 u f n i
  mkdir -p $d; i=0
  for u in $(echo $urls | tr ';' ' '); do i=$((i + 1))
    curl -sS --retry 3 --max-time 120 -r 0-12582911 "$u" 2>/dev/null | gzip -dc 2>/dev/null | head -80000 > $d/f$i.fastq
  done
  # read lengths per file; index reads (<= 12 nt) and empty files are dropped; the first two others are R1, R2 (the detector swaps if needed)
  local files=() lens=()
  for f in $d/f*.fastq; do n=$(awk 'NR % 4 == 2 { s += length($0); c++ } END { if (c) printf "%d", s / c; else print 0 }' $f)
    [ "$n" -gt 12 ] && [ $(wc -l < $f) -ge 4000 ] && { files+=($f); lens+=($n); }; done
  if [ ${#files[@]} -lt 2 ]; then
    echo -e "$run\t$sample\t$study\t$assay\t-\tno-pair\t-\t-\t-\t-\t-\t-" >> $O/calls.part.$run; rm -rf $d; return
  fi
  local np=$(( $(wc -l < ${files[0]}) / 4 )) b bn bp log tag conf wl r1l r2l ev
  for b in $(echo $BINS | tr , ' '); do bn=${b%%=*}; bp=${b#*=}; log=$O/logs/$run.$bn.log
    ( cd $d && timeout 600 $bp download --r1 ${files[0]} --r2 ${files[1]} --staging-dir $d/st-$bn -o $d/$bn.1fq ) > $log 2>&1
    tag=$(tr '\r' '\n' < $log | grep -a -o -E "\[singlet-download\] Protocol: [A-Za-z0-9._-]+" | tail -1 | sed 's/.*: //')
    conf=$(tr '\r' '\n' < $log | grep -a -o -E "Protocol: [A-Za-z0-9._-]+ \(confidence: [0-9]+" | tail -1 | grep -o -E "[0-9]+$")
    # the gate rejects non-10x libraries before the Protocol line; its REJECT line carries the detector's tag
    [ -z "$tag" ] && { g=$(grep -a -o -E "\[singlet-gate\] REJECT .* tag=[A-Za-z0-9._-]+ confidence=[0-9]+" $log | tail -1)
      [ -n "$g" ] && { tag=gate:$(echo "$g" | sed 's/.* tag=//; s/ confidence=.*//'); conf=${g##*=}; }; }
    [ -z "$tag" ] && grep -a -q "data_incomplete" $log && tag=data_incomplete
    wl=$(grep -a -o -E "whitelist validation: [0-9]+/[0-9]+ match \([0-9]+" $log | tail -1 | grep -o -E "[0-9]+$")
    r1l=$(grep -a -o -E "R1 length: [0-9]+, R2 length: [0-9]+" $log | tail -1 | sed 's/R1 length: //; s/, R2 length: /\//')
    ev=$(grep -a "3./5. check" $log | tail -1 | sed 's/.*check: //; s/[;,] */ /g')
    echo -e "$run\t$sample\t$study\t$assay\t$bn\t${tag:--}\t${conf:--}\t${wl:--}\t${r1l:--}\t${lens[0]}/${lens[1]}\t$np\t${ev:--}" >> $O/calls.part.$run
    rm -rf $d/$bn.1fq $d/st-$bn
  done
  rm -rf $d
}
tail -n +2 $O/picks.tsv | while IFS=$'\t' read -r a b c d e f g; do
  while [ $(jobs -rp | wc -l) -ge ${PAR:-6} ]; do wait -n; done
  one "$a" "$b" "$c" "$d" "${e// /_}" "$f" "${g:--}" &
done; wait
echo -e "run\tsample\tstudy\tassay\tbinary\ttag\tconfidence\twl_pct\tencoded_r1/r2\tfile_r1/r2\tpairs\tevidence_r1tso_r1polyt_r2tso_r2polya_r2polyt" > $O/calls.tsv
cat $O/calls.part.* | sort >> $O/calls.tsv; rm -f $O/calls.part.*
$PY $SRC/compare_b20.py $O/calls.tsv | tee $O/score.txt
chmod -R go-rwx $O
echo done $(date -Is)
