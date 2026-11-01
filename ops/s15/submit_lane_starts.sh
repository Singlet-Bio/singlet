#!/bin/bash
# Re-submit the first GPU job of each lane (the first attempt exceeded the 375 GB limit of 1-GPU jobs); lanes continue from lane/*.env.
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; ST=$D/pilot_stage.sh
for g in GSM7884577:795366 GSM7064234:795371 GSM5171610:795376; do
  gsm=${g%:*}; f=${g#*:}; r=$(awk -F'\t' -v g=$gsm '$1==g' $D/samples.tsv); mem=$(echo "$r" | cut -f6); pg=$(echo "$r" | cut -f9); tg=$(echo "$r" | cut -f10)
  n=$([ $mem -gt 375 ] && echo 2 || echo 1)
  j=$(sbatch --parsable --account=debruinz --job-name=s15-gpu-$gsm --partition=$pg --gres=gpu:nvidia_h100_nvl:$n --cpus-per-task=16 --mem=${mem}G --time=$tg --dependency=afterany:$f --output=$W/logs/%x-%j.out $ST sgpu $gsm)
  echo "gpu-lane-start $gsm $j gpus=$n mem=$mem $pg $tg"
done
