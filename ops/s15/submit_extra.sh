#!/bin/bash
# S15 amendment 1 (login node, sbatch only): two more samples with the same core chain + tools, and the paired V(D)J libraries of GSM5171610 and GSM9005866.
#   GSM8085472  GSE256089, 10 hashed donors (title), 5'; GEX 1.42 G pairs; pool truth K=10 (HTO library GSM8085473 is in SRA, tag map unverified)
#   GSM9005866  GSE298047 sample 1001, 5' CD45+ cells; GEX 401 M; paired TCR library GSM9005868 and HTO library GSM9005867 in SRA
# V(D)J libraries: GSM5171642 (Healthydonor5_VDJ, pairs GSM5171610), GSM9005868.
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; ST=$D/pilot_stage.sh; TL=$D/pilot_tools.sh; LOG=--output=$W/logs/%x-%j.out
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes "$@"; }
declare -A F C S P
for g in GSM8085472 GSM9005866; do
  r=$(awk -F'\t' -v g=$g '$1==g' $D/samples.tsv); mem=$(echo "$r" | cut -f6); pc=$(echo "$r" | cut -f7); tc=$(echo "$r" | cut -f8)
  f=$(sb --job-name=s15-fetch-$g --partition=short-cpu --cpus-per-task=8 --mem=32G --time=03:59:00 $LOG $ST fetch $g)
  q=$(sb --job-name=s15-fq-$g --partition=short-cpu --cpus-per-task=2 --mem=16G --time=03:59:00 --dependency=afterok:$f $LOG $ST fq $g)
  c=$(sb --job-name=s15-scpu-$g --partition=$pc --cpus-per-task=16 --mem=${mem}G --time=$tc --dependency=afterok:$f $LOG $ST scpu $g)
  s=$(sb --job-name=s15-starsolo-$g --partition=short-cpu --cpus-per-task=16 --mem=120G --time=03:59:00 --dependency=afterok:$q,afterany:$c $LOG $ST starsolo $g)
  p=$(sb --job-name=s15-scap-$g --partition=short-cpu --cpus-per-task=16 --mem=96G --time=03:59:00 --dependency=afterok:$q $LOG $ST scap $g)
  F[$g]=$f; C[$g]=$c; S[$g]=$s; P[$g]=$p; mkdir -p $W/pilot/$g; touch $W/pilot/$g/KEEP
  printf "%s\tfetch=%s\tfq=%s\tstarsolo=%s\tscap=%s\tscpu=%s\n" $g $f $q $s $p $c | tee -a $D/jobids.tsv
done
# hook the two GPU runs onto the tails of existing lanes (lane 0 and lane 2); the head's env lists the rest of the lane
append() { head=$1; tup=$2; e=$D/lane/$head.env
  grep -q "^LANE_REST=" $e || { echo "no env for $head"; return 1; }
  python3 - "$e" "$tup" <<'PY'
import sys
e, t = sys.argv[1:3]; L = open(e).read().splitlines()
L = [("LANE_REST=" + (l.split("=", 1)[1] + "," + t if l.split("=", 1)[1] else t)) if l.startswith("LANE_REST=") else l for l in L]
open(e, "w").write("\n".join(L) + "\n")
PY
  echo "lane head $head: $(grep LANE_REST $e | cut -c1-200)"; }
append GSM7884577 "GSM8085472:${F[GSM8085472]}:${C[GSM8085472]}:${S[GSM8085472]}:${P[GSM8085472]}"
append GSM5171610 "GSM9005866:${F[GSM9005866]}:${C[GSM9005866]}:${S[GSM9005866]}:${P[GSM9005866]}"
# tools for the new samples (the shared kb index job is reused)
KBI=$(awk '$1=="kbidx"{print $2}' $D/toolids.tsv | head -1) bash $D/submit_tools.sh GSM8085472 GSM9005866
# demux truth for the pool: K = 10 hashed donors
for t in vireo:64G souporcell:96G; do n=${t%:*}; m=${t#*:}
  j=$(sb --export=ALL,KDON=10 --job-name=s15-$n-GSM8085472 --partition=cpu --cpus-per-task=16 --mem=$m --time=2-00:00:00 --dependency=afterok:${C[GSM8085472]} $LOG $TL $n GSM8085472); printf "GSM8085472\t%s\t%s\n" $n $j | tee -a $D/toolids.tsv
done
# V(D)J libraries: fetch reads only, then Singlet V(D)J + TRUST4 on them
for v in GSM5171642 GSM9005868; do mkdir -p $W/pilot/$v
  f=$(sb --export=ALL,NODECODE=1 --job-name=s15-fetch-$v --partition=short-cpu --cpus-per-task=8 --mem=32G --time=03:59:00 $LOG $ST fetch $v)
  j=$(sb --job-name=s15-vdjlib-$v --partition=short-bigmem --cpus-per-task=16 --mem=128G --time=03:59:00 --dependency=afterok:$f $LOG $TL vdjlib $v)
  k=$(sb --job-name=s15-vdjclean-$v --partition=short-cpu --cpus-per-task=1 --mem=2G --time=00:30:00 --dependency=afterany:$j $LOG $TL vdjclean $v)
  printf "%s\tvdj-fetch=%s\tvdjlib=%s\tvdjclean=%s\n" $v $f $j $k | tee -a $D/toolids.tsv
done
