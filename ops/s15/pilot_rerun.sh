#!/bin/bash
# S15 engine rerun on the fixed tip (prod/campaign 2a3b0d61): Singlet CPU and GPU only, from the .sra (refetched if the first clean-up removed it).
#   pilot_rerun.sh STAGE GSM     STAGE: rfetch | rcpu | rgpu | rclean        (Slurm options come from the submitter: submit_rerun.sh)
# Outputs: $P/s1cpu_$COMMIT, $P/s1gpu_$COMMIT, reads in $P/stage_$COMMIT. The first-pass runs ($P/s1cpu, $P/s1gpu, binary 337b405b) stay as the before column.
set -uo pipefail
ST=$1; GSM=$2
W=/mnt/projects/debruinz_project/debruinz/singlet; PL=$W/pilot; P=$PL/$GSM; D=$W/jobs/s15
SINGLET_THREADS=16; COMMIT=0ea48783; BINDIR=$D/bin/$COMMIT; REFB=/mnt/projects/debruinz_project/cellarium/reference; G=$REFB/GRCh38-2024-A
WL=$W/pipeline-benchmark/whitelists; ENV=$W/envs/pbench
row=$(awk -F'\t' -v g=$GSM '$1==g' $D/samples.tsv); TIER=$(echo "$row" | cut -f5)
export SINGLET_WHITELIST_DIR=$WL PATH=$ENV/bin:$PATH SINGLET_STALL_TIMEOUT=21600   # deep samples sort the tagged BAM for more than the default 1800 s
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR $P; trap 'rm -rf $SCR' EXIT
echo "== s15 rerun $ST $GSM node=$(hostname) $(date -Is)"
for i in $(seq 120); do [ -s $BINDIR/commit ] && break; sleep 30; done   # the build job may still be running when a lane job starts
[ "$(cut -c1-8 $BINDIR/commit)" = "$COMMIT" ] || { echo "WRONG BINARY COMMIT"; exit 3; }
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=no "$@"; }
# engine outdir runs...   Pooled runs must share a protocol: when the engine names a run that differs, drop it (recorded in runs_dropped.txt) and retry.
singlet_run() {
  eng=$1; out=$2; shift 2; rm -rf $out; mkdir -p $out; GD=; ST1=${STG:-$P/stage_$COMMIT}
  used=$(ls $ST1/*.1fq | xargs -n1 basename | sed s/.1fq$// | sort | paste -sd,); : > $out/runs_dropped.txt
  while :; do
    if [ "$(echo $used | tr , '\n' | wc -l)" = 1 ]; then ra="$ST1/$used.1fq"; else ra="--runs-dir $ST1 --runs-used $used"; fi
    /usr/bin/time -v -o $out/time.txt $BINDIR/singlet-$eng $ra $GD --ref-base $REFB --out-prefix $out/out/ --threads $SINGLET_THREADS --pipeline \
       --output-format 1pz --engine $eng --memory-tier $TIER > $out/run.log 2>&1
    rc=$?; grep -h '^\[species-detect\]' $out/run.log | head -1 >> $out/species.txt
    if [ $rc != 0 ] && [ -z "$GD" ] && grep -q 'requires --genome-dir' $out/run.log; then   # species detection failed: record it, then give the genome explicitly (GTF too: it is only auto-resolved from a detected genome)
      echo "AUTODETECT FAILED -> explicit genome-dir" >> $out/species.txt; cp $out/run.log $out/run_autodetect.log; GD="--genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf"; rm -rf $out/out; continue; fi
    bad=$(grep -oE 'stage_[0-9a-f]+/[A-Z0-9]+\.1fq differs from the first run in its protocol' $out/run.log | head -1 | sed -E 's#.*stage_[0-9a-f]+/##;s#\.1fq.*##')
    [ -n "$bad" ] || break
    echo "$bad" >> $out/runs_dropped.txt; used=$(echo $used | tr , '\n' | grep -vx $bad | paste -sd,); echo "dropped $bad (protocol differs), retry with $used"; rm -rf $out/out
  done
  echo "$used" > $out/runs_used.txt; echo "$SINGLET_THREADS" > $out/threads.txt
  pt=$(grep -rhoE '"pileup_threads": *[0-9]+' $out/out 2> /dev/null | head -1 | grep -oE '[0-9]+$'); echo "${pt:-unknown}" > $out/pileup_threads.txt
  grep -rhoE '"pileup_path_reason": *"[^"]*"' $out/out 2> /dev/null | head -1 > $out/pileup_path_reason.txt
  [ "${pt:-x}" = 1 ] && echo "pileup_threads=1 (stream path)" || { echo "INVALID: pileup_threads=${pt:-unknown} (expected 1)"; touch $out/INVALID; }
  echo "singlet $eng rc=$rc threads=$SINGLET_THREADS used=$used"; grep -E "Elapsed \(wall|Maximum resident" $out/time.txt; return $rc; }
drop_sra_if_done() {   # the .sra goes only when everything that reads it has delivered (whichever of rclean / rbamclean runs last gets to do it)
  ok=1; for e in s1cpu_$COMMIT s1gpu_$COMMIT; do [ -d $P/$e/out ] && [ ! -e $P/$e/INVALID ] || ok=0; done
  [ -s $P/starsolo_bam/Aligned.sortedByCoord.out.bam ] || ok=0
  tot=$(python3 -c "import json;print(json.load(open('$P/runs.json'))['pairs'])"); [ $tot -le 100000000 ] || [ -s $P/s1cap_$COMMIT/pileup_threads.txt ] || ok=0
  if [ $ok = 1 ]; then rm -rf $P/stage_$COMMIT; [ -e $P/KEEPSRA ] || rm -rf $P/sra; echo ".sra dropped"; else echo "not everything delivered yet or a run failed: .sra and stage_$COMMIT kept"; fi; }
case $ST in
rfetch)   # .sra (kept ones are reused), then decode with the new binary into stage_$COMMIT
  mkdir -p $D/capdir; [ -f $D/capdir/stage.json ] || echo '{"rate_cap_MBps": 450}' > $D/capdir/stage.json
  STAGE=stage_$COMMIT REUSE_SRA=1 python3 -u $D/pilot_fetch.py $GSM $P $D/src $BINDIR/singlet-cpu $WL $D/capdir; rc=$?; du -sh $P/sra $P/stage_$COMMIT; exit $rc ;;
rstar)   # STARsolo with a sorted BAM on the first 100 M pairs (the competitor input), reads dumped from the kept .sra
  CAPPAIRS=100000000; rm -rf $P/fq; mkdir -p $P/fq
  python3 - <<PY || exit 4
import json, subprocess, os
r = json.load(open("$P/runs.json"))["runs"]; left = $CAPPAIRS
for acc, n in r:
    if left <= 0: break
    k = min(n, left); left -= k
    subprocess.run(["$ENV/bin/fastq-dump", "-X", str(k), "--split-files", "--gzip", "-O", "$P/fq", "$P/sra/%s.sra" % acc], check=True)
PY
  BAMRUN=1 bash $D/pilot_stage.sh starsolo $GSM; rc=$?; ls -l $P/starsolo_bam | head; exit $rc ;;
rcap)   # Singlet CPU on the same 100 M-pair prefix (new binary, stream path): the like-for-like side of the genotype / mtDNA / competitor comparisons. Skipped when the sample has <= 100 M pairs.
  tot=$(python3 -c "import json;print(json.load(open('$P/runs.json'))['pairs'])"); [ $tot -gt 100000000 ] || { echo "sample <= 100M pairs: s1cpu_$COMMIT is the matched run"; exit 0; }
  rm -rf $P/cap1fq_$COMMIT; mkdir -p $P/cap1fq_$COMMIT; left=100000000
  for acc in $(python3 -c "import json;print(' '.join(a for a,n in json.load(open('$P/runs.json'))['runs']))"); do [ $left -gt 0 ] || break
    n=$(python3 -c "import json;print(dict(map(tuple,json.load(open('$P/runs.json'))['runs']))['$acc'])"); k=$(( n < left ? n : left )); left=$((left-k))
    $BINDIR/singlet-cpu download $P/sra/$acc.sra --quality binned --max-reads $k --vdb-threads 2 --no-profile --whitelist-dir $WL -o $P/cap1fq_$COMMIT/$acc.1fq 2>&1 | grep -E "singlet-gate" | cut -c1-200
  done
  STG=$P/cap1fq_$COMMIT singlet_run cpu $P/s1cap_$COMMIT ;;
rcpu) ls $P/stage_$COMMIT/*.done > /dev/null 2>&1 || { echo "no staged reads"; exit 6; }; singlet_run cpu $P/s1cpu_$COMMIT ;;
rgpu)   # one of 3 serial GPU lanes: after this sample, submit its clean-up and the next sample of the lane (at most 3 GPU jobs exist at once)
  . $D/lane/r_$GSM.env
  post() { cid=$(sb --job-name=s15-rclean-$GSM --partition=short-cpu --cpus-per-task=2 --mem=8G --time=01:00:00 --dependency=afterany:${SELF_C//+/:} --output=$W/logs/%x-%j.out $D/pilot_rerun.sh rclean $GSM); echo "rclean job $cid"
    [ -n "${LANE_REST:-}" ] || return 0
    nxt=${LANE_REST%%,*}; rest=; case $LANE_REST in *,*) rest=${LANE_REST#*,};; esac; IFS=: read g f c <<< "$nxt"
    r=$(awk -F'\t' -v g=$g '$1==g' $D/samples.tsv); mem=$(echo "$r" | cut -f6); pg=$(echo "$r" | cut -f9); tg=$(echo "$r" | cut -f10)
    printf 'LANE_REST=%s
SELF_C=%s
' "$rest" $c > $D/lane/r_$g.env
    sb --job-name=s15-rgpu-$g --partition=$pg --gres=gpu:nvidia_h100_nvl:1 --cpus-per-task=16 --mem=${mem}G --time=$tg --dependency=afterany:$f --output=$W/logs/%x-%j.out $D/pilot_rerun.sh rgpu $g; }
  trap 'post; rm -rf $SCR' EXIT
  nvidia-smi --query-gpu=name --format=csv,noheader
  ls $P/stage_$COMMIT/*.done > /dev/null 2>&1 || { echo "no staged reads (rfetch failed)"; exit 6; }
  singlet_run gpu $P/s1gpu_$COMMIT ;;
rbamclean) drop_sra_if_done; rm -rf $P/fq $P/cap1fq_$COMMIT; find $P \( -name '*.bam' -o -name '*.bai' \) -delete; du -sh $P ;;   # after the BAM tools
rclean)   # (the first-pass reads stage/fq/cap1fq belong to the competitor tools' own clean-up)  equiv of the two new runs; the .sra goes now (unless KEEPSRA: slowest to fetch, kept for another rerun); results and logs stay
  [ -d $P/s1cpu_$COMMIT/out ] && [ -d $P/s1gpu_$COMMIT/out ] && python3 $D/out_equiv.py $P/s1cpu_$COMMIT/out $P/s1gpu_$COMMIT/out > $P/equiv_$COMMIT.txt 2>&1
  drop_sra_if_done; du -sh $P; ls $P ;;   # the STARsolo BAM stays for the BAM tools; rbamclean removes it
esac
