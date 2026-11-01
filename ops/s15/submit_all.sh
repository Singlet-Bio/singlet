#!/bin/bash
# S15: submit every sample's chain (login node: sbatch only).   submit_all.sh BUILD_JOBID
#   build -> fetch -> { fq -> starsolo, fq -> scap (matched-prefix Singlet CPU), scpu (full) }      and 3 serial GPU lanes (sgpu -> cleanup, next sample).
# GPU lanes keep at most 3 GPU jobs in the queue at once; each sgpu job submits its sample's cleanup and the next lane job (pilot_stage.sh post_gpu).
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; B=$1; ST=$D/pilot_stage.sh; mkdir -p $W/logs $W/pilot $D/lane
OUT=$D/jobids.tsv; : > $OUT
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes "$@"; }
declare -A F C S P
tail -n +2 $D/samples.tsv | sort -t$'\t' -k4,4nr > $D/samples.sorted
while IFS=$'\t' read -r g label five pairs tier mem pc tc pg tg; do
  L=s15-%s-$g; LOG=$W/logs/%x-%j.out
  f=$(sb --job-name=s15-fetch-$g --partition=short-cpu --cpus-per-task=8 --mem=32G --time=03:59:00 --dependency=afterok:$B --output=$LOG $ST fetch $g)
  q=$(sb --job-name=s15-fq-$g --partition=short-cpu --cpus-per-task=2 --mem=16G --time=03:59:00 --dependency=afterok:$f --output=$LOG $ST fq $g)
  s=$(sb --job-name=s15-starsolo-$g --partition=short-cpu --cpus-per-task=16 --mem=120G --time=03:59:00 --dependency=afterok:$q --output=$LOG $ST starsolo $g)
  p=$(sb --job-name=s15-scap-$g --partition=short-cpu --cpus-per-task=16 --mem=96G --time=03:59:00 --dependency=afterok:$q --output=$LOG $ST scap $g)
  c=$(sb --job-name=s15-scpu-$g --partition=$pc --cpus-per-task=32 --mem=${mem}G --time=$tc --dependency=afterok:$f --output=$LOG $ST scpu $g)
  F[$g]=$f; C[$g]=$c; S[$g]=$s; P[$g]=$p
  printf "%s\tfetch=%s\tfq=%s\tstarsolo=%s\tscap=%s\tscpu=%s\n" $g $f $q $s $p $c | tee -a $OUT
done < $D/samples.sorted
# GPU lanes: samples sorted by depth, dealt round-robin to 3 lanes
mapfile -t GS < <(cut -f1 $D/samples.sorted)
for lane in 0 1 2; do
  tup=()
  for i in "${!GS[@]}"; do [ $((i % 3)) = $lane ] && { g=${GS[$i]}; tup+=("$g:${F[$g]}:${C[$g]}:${S[$g]}:${P[$g]}"); }; done
  first=${tup[0]}; rest=$(IFS=,; echo "${tup[*]:1}")
  IFS=: read g f c s p <<< "$first"
  r=$(awk -F'\t' -v g=$g '$1==g' $D/samples.tsv); mem=$(echo "$r" | cut -f6); pg=$(echo "$r" | cut -f9); tg=$(echo "$r" | cut -f10)
  printf 'LANE_REST=%s
SELF_C=%s
SELF_S=%s
SELF_P=%s
' "$rest" $c $s $p > $D/lane/$g.env
  j=$(sb --job-name=s15-gpu-$g --partition=$pg --gres=gpu:nvidia_h100_nvl:$([ $mem -gt 375 ] && echo 2 || echo 1) --cpus-per-task=16 --mem=${mem}G --time=$tg --dependency=afterany:$f --kill-on-invalid-dep=no \
        --output=$W/logs/%x-%j.out $ST sgpu $g)
  printf "gpu-lane%s\t%s\tfirst=%s\tlane=%s\n" $lane $j $g "${tup[*]}" | tee -a $OUT
done
