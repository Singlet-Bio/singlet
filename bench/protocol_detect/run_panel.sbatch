#!/bin/bash
#SBATCH --job-name=detect-panel
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Protocol-detection truth panel: the head of each library (HTTP range on the ENA FASTQ, or a local FASTQ) goes through
# `singlet download --r1 --r2` in auto mode, once per binary in BINS (name=path,...). Output $W/jobs/detect/panel-<job>/calls.tsv:
# one row per library x binary with the gate verdict, the protocol call, its confidence, the assay type and the read lengths.
# Needs BINS and SRC (this directory on Clipper). EXCLUDE names the accessions that must never be scored (kept outside the repository).
set -uo pipefail
: "${BINS:?name=path[,name=path]}" "${SRC:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; PY=$W/envs/bpub/bin/python3
export SINGLET_WHITELIST_DIR=${WLDIR:-$W/pipeline-benchmark/whitelists}
EXCLUDE=${EXCLUDE:-$W/jobs/detect/heldout_exclude.txt}
O=$W/jobs/detect/panel-$SLURM_JOB_ID; mkdir -p $O/logs; SCR=/tmp/$USER-detp-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
cp $SRC/panel_design.tsv $O/panel.tsv; echo -e "id	r1_len	r2_len	r1_fixed	r2_fixed	r1_12mer_reuse	r2_12mer_reuse	r2_top100_15mer_share" > $O/geometry.tsv
if [ "${CORPUS:-1}" = 1 ]; then
  $PY $SRC/panel_pick.py $W/data/corpus-beta/out $EXCLUDE $O/panel_corpus.tsv && tail -n +2 $O/panel_corpus.tsv >> $O/panel.tsv
fi
[ -n "${ONLY:-}" ] && { head -1 $O/panel.tsv > $O/p.tmp; grep -E "$ONLY" $O/panel.tsv >> $O/p.tmp; mv $O/p.tmp $O/panel.tsv; }
HEADB=${HEADB:-12582911}
one() {   # id group truth run r1 r2
  local id=$1 truth=$3 d=$SCR/$1 i=0 u f n
  mkdir -p $d
  for u in $5 $6; do i=$((i + 1))
    case $u in
      local:*) zcat $W/${u#local:} 2>/dev/null | head -80000 > $d/f$i.fastq ;;
      *) curl -sS --retry 3 --max-time 180 -r 0-$HEADB "$u" 2>/dev/null | gzip -dc 2>/dev/null | head -80000 > $d/f$i.fastq ;;
    esac
  done
  local files=()
  for f in $d/f1.fastq $d/f2.fastq; do [ $(wc -l < $f) -ge 4000 ] && files+=($f); done
  if [ ${#files[@]} -lt 2 ]; then echo -e "$id\t$2\t$truth\t$4\t-\tno-reads\t-\t-\t-\t-\t-\t-\t-" > $O/calls.part.$id; rm -rf $d; return; fi
  $PY $SRC/read_geometry.py $id ${files[0]} ${files[1]} >> $O/geometry.tsv 2>/dev/null
  local r1l=$(awk 'NR%4==2{s+=length($0);c++}END{printf "%d", s/c}' ${files[0]}) r2l=$(awk 'NR%4==2{s+=length($0);c++}END{printf "%d", s/c}' ${files[1]})
  for b in $(echo $BINS | tr , ' '); do local bn=${b%%=*}; local bp=${b#*=}; local log=$O/logs/$id.$bn.log
    ( cd $d && timeout 900 $bp download --r1 ${files[0]} --r2 ${files[1]} --staging-dir $d/st-$bn -o $d/$bn.1fq ) > $log 2>&1; local rc=$?
    echo -e "$id\t$2\t$truth\t$4\t$bn\t$rc\t$r1l/$r2l" > $O/calls.part.$id.$bn
    rm -rf $d/$bn.1fq $d/st-$bn
  done
  rm -rf $d
}
tail -n +2 $O/panel.tsv | while IFS=$'\t' read -r id group truth run r1 r2 src note; do
  while [ $(jobs -rp | wc -l) -ge ${PAR:-6} ]; do wait -n; done
  one "$id" "$group" "$truth" "$run" "$r1" "$r2" &
done; wait
$PY $SRC/score_panel.py $O > $O/score.txt; cat $O/score.txt
chmod -R go-rwx $O; echo done $(date -Is)
