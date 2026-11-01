#!/bin/bash
# S15 engine rerun on the fixed tip (prod/campaign 2a3b0d61): Singlet CPU and GPU only, from the .sra (refetched if the first clean-up removed it).
#   pilot_rerun.sh STAGE GSM     STAGE: rfetch | rcpu | rgpu | rclean        (Slurm options come from the submitter: submit_rerun.sh)
# Outputs: $P/s1cpu_r, $P/s1gpu_r, reads in $P/stage_r. The first-pass runs ($P/s1cpu, $P/s1gpu, binary 337b405b) stay as the before column.
set -uo pipefail
ST=$1; GSM=$2
W=/mnt/projects/debruinz_project/debruinz/singlet; PL=$W/pilot; P=$PL/$GSM; D=$W/jobs/s15
SINGLET_THREADS=16; COMMIT=2a3b0d61; BINDIR=$D/bin/$COMMIT; REFB=/mnt/projects/debruinz_project/cellarium/reference; G=$REFB/GRCh38-2024-A
WL=$W/pipeline-benchmark/whitelists; ENV=$W/envs/pbench
row=$(awk -F'\t' -v g=$GSM '$1==g' $D/samples.tsv); TIER=$(echo "$row" | cut -f5)
export SINGLET_WHITELIST_DIR=$WL PATH=$ENV/bin:$PATH SINGLET_STALL_TIMEOUT=21600   # deep samples sort the tagged BAM for more than the default 1800 s
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR $P; trap 'rm -rf $SCR' EXIT
echo "== s15 rerun $ST $GSM node=$(hostname) $(date -Is)"
[ "$(cut -c1-8 $BINDIR/commit)" = "$COMMIT" ] || { echo "WRONG BINARY COMMIT"; exit 3; }
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=no "$@"; }
# engine outdir runs...   Pooled runs must share a protocol: when the engine names a run that differs, drop it (recorded in runs_dropped.txt) and retry.
singlet_run() {
  eng=$1; out=$2; shift 2; rm -rf $out; mkdir -p $out; ST1=$P/stage_r
  used=$(ls $ST1/*.1fq | xargs -n1 basename | sed s/.1fq$// | sort | paste -sd,); : > $out/runs_dropped.txt
  while :; do
    if [ "$(echo $used | tr , '\n' | wc -l)" = 1 ]; then ra="$ST1/$used.1fq"; else ra="--runs-dir $ST1 --runs-used $used"; fi
    /usr/bin/time -v -o $out/time.txt $BINDIR/singlet-$eng $ra --genome-dir $G/star_2.7.11b --ref-base $REFB --out-prefix $out/out/ --threads $SINGLET_THREADS --pipeline \
       --output-format 1pz --engine $eng --memory-tier $TIER --tagged-bam > $out/run.log 2>&1
    rc=$?; bad=$(grep -oE 'stage_r/[A-Z0-9]+\.1fq differs from the first run in its protocol' $out/run.log | head -1 | sed 's#stage_r/##;s#\.1fq.*##')
    [ -n "$bad" ] || break
    echo "$bad" >> $out/runs_dropped.txt; used=$(echo $used | tr , '\n' | grep -vx $bad | paste -sd,); echo "dropped $bad (protocol differs), retry with $used"; rm -rf $out/out
  done
  echo "$used" > $out/runs_used.txt; echo "$SINGLET_THREADS" > $out/threads.txt
  pt=$(grep -rhoE '"pileup_threads": *[0-9]+' $out/out 2> /dev/null | head -1 | grep -oE '[0-9]+$'); echo "${pt:-unknown}" > $out/pileup_threads.txt
  [ "${pt:-x}" = 1 ] && echo "pileup_threads=1 (stream path)" || { echo "INVALID: pileup_threads=${pt:-unknown} (expected 1)"; touch $out/INVALID; }
  echo "singlet $eng rc=$rc threads=$SINGLET_THREADS used=$used"; grep -E "Elapsed \(wall|Maximum resident" $out/time.txt; return $rc; }
case $ST in
rfetch)   # .sra (kept ones are reused), then decode with the new binary into stage_r
  mkdir -p $D/capdir; [ -f $D/capdir/stage.json ] || echo '{"rate_cap_MBps": 450}' > $D/capdir/stage.json
  STAGE=stage_r REUSE_SRA=1 python3 -u $D/pilot_fetch.py $GSM $P $D/src $BINDIR/singlet-cpu $WL $D/capdir; rc=$?; du -sh $P/sra $P/stage_r; exit $rc ;;
rcpu) ls $P/stage_r/*.done > /dev/null 2>&1 || { echo "no staged reads"; exit 6; }; singlet_run cpu $P/s1cpu_r ;;
rgpu)   # one of 3 serial GPU lanes: after this sample, submit its clean-up and the next sample of the lane (at most 3 GPU jobs exist at once)
  . $D/lane/r_$GSM.env
  post() { cid=$(sb --job-name=s15-rclean-$GSM --partition=short-cpu --cpus-per-task=2 --mem=8G --time=01:00:00 --dependency=afterany:${SELF_C} --output=$W/logs/%x-%j.out $D/pilot_rerun.sh rclean $GSM); echo "rclean job $cid"
    [ -n "${LANE_REST:-}" ] || return 0
    nxt=${LANE_REST%%,*}; rest=; case $LANE_REST in *,*) rest=${LANE_REST#*,};; esac; IFS=: read g f c <<< "$nxt"
    r=$(awk -F'\t' -v g=$g '$1==g' $D/samples.tsv); mem=$(echo "$r" | cut -f6); pg=$(echo "$r" | cut -f9); tg=$(echo "$r" | cut -f10)
    printf 'LANE_REST=%s
SELF_C=%s
' "$rest" $c > $D/lane/r_$g.env
    sb --job-name=s15-rgpu-$g --partition=$pg --gres=gpu:nvidia_h100_nvl:1 --cpus-per-task=16 --mem=${mem}G --time=$tg --dependency=afterany:$f --output=$W/logs/%x-%j.out $D/pilot_rerun.sh rgpu $g; }
  trap 'post; rm -rf $SCR' EXIT
  nvidia-smi --query-gpu=name --format=csv,noheader
  ls $P/stage_r/*.done > /dev/null 2>&1 || { echo "no staged reads (rfetch failed)"; exit 6; }
  singlet_run gpu $P/s1gpu_r ;;
rclean)   # (the first-pass reads stage/fq/cap1fq belong to the competitor tools' own clean-up)  equiv of the two new runs; the .sra goes now (unless KEEPSRA: slowest to fetch, kept for another rerun); results and logs stay
  [ -d $P/s1cpu_r/out ] && [ -d $P/s1gpu_r/out ] && python3 $D/out_equiv.py $P/s1cpu_r/out $P/s1gpu_r/out > $P/equiv_r.txt 2>&1
  rm -rf $P/stage_r; [ -e $P/KEEPSRA ] || rm -rf $P/sra
  find $P \( -name '*.bam' -o -name '*.bai' \) -delete; du -sh $P; ls $P ;;
esac
