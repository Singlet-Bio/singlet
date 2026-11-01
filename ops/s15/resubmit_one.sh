#!/bin/bash
# S15: rerun one sample's chain after its fetch failed (login node, sbatch only).   resubmit_one.sh GSM OLD_GPU_HEAD_JOBID
# Used for GSM7064234: the first decode ran without --whitelist-dir (harness bug), so the gate rejected the run. The kept .sra is reused (REUSE_SRA=1).
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; ST=$D/pilot_stage.sh; LOG=--output=$W/logs/%x-%j.out; g=$1; oldgpu=$2
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes "$@"; }
r=$(awk -F'\t' -v g=$g '$1==g' $D/samples.tsv); mem=$(echo "$r" | cut -f6); pc=$(echo "$r" | cut -f7); tc=$(echo "$r" | cut -f8); pg=$(echo "$r" | cut -f9); tg=$(echo "$r" | cut -f10)
touch $W/pilot/$g/KEEP
f=$(sb --export=ALL,REUSE_SRA=1 --job-name=s15-fetch-$g --partition=short-cpu --cpus-per-task=8 --mem=32G --time=03:59:00 $LOG $ST fetch $g)
q=$(sb --job-name=s15-fq-$g --partition=short-cpu --cpus-per-task=2 --mem=16G --time=03:59:00 --dependency=afterok:$f $LOG $ST fq $g)
c=$(sb --job-name=s15-scpu-$g --partition=$pc --cpus-per-task=16 --mem=${mem}G --time=$tc --dependency=afterok:$f $LOG $ST scpu $g)
s=$(sb --job-name=s15-starsolo-$g --partition=short-cpu --cpus-per-task=16 --mem=120G --time=03:59:00 --dependency=afterok:$q,afterany:$c $LOG $ST starsolo $g)
p=$(sb --job-name=s15-scap-$g --partition=short-cpu --cpus-per-task=16 --mem=96G --time=03:59:00 --dependency=afterok:$q $LOG $ST scap $g)
# GPU lane head: replace the pending job so it waits for the new fetch; the lane env keeps LANE_REST, SELF ids move to the new jobs
[ -n "$oldgpu" ] && scancel $oldgpu
python3 - $D/lane/$g.env $c $s $p <<'PY'
import sys
e, c, s, p = sys.argv[1:5]; L = open(e).read().splitlines()
L = [("SELF_C=" + c) if l.startswith("SELF_C=") else ("SELF_S=" + s) if l.startswith("SELF_S=") else ("SELF_P=" + p) if l.startswith("SELF_P=") else l for l in L]
open(e, "w").write("\n".join(L) + "\n")
PY
n=$([ $mem -gt 375 ] && echo 2 || echo 1)
j=$(sbatch --parsable --account=debruinz --job-name=s15-gpu-$g --partition=$pg --gres=gpu:nvidia_h100_nvl:$n --cpus-per-task=16 --mem=${mem}G --time=$tg --dependency=afterany:$f $LOG $ST sgpu $g)
sed -i "/^$g\tfetch=/d" $D/jobids.tsv; printf "%s\tfetch=%s\tfq=%s\tstarsolo=%s\tscap=%s\tscpu=%s\n" $g $f $q $s $p $c | tee -a $D/jobids.tsv
echo "gpu head $j (replaces $oldgpu)"; printf "%s\tgpu-head\t%s\n" $g $j >> $D/toolids.tsv
KBI=$(awk '$2=="kbidx" || $1=="kbidx"{print ($1=="kbidx")?$2:$3}' $D/toolids.tsv | tail -1) bash $D/submit_tools.sh $g
