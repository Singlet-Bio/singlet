#!/bin/bash
#SBATCH --job-name=gpu-pilot
#SBATCH --account=debruinz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --cpus-per-task=54
#SBATCH --mem=700G
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --exclude=g050
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# GPU pilot: holds one whole H100 node (2 GPUs, the 54 usable cores, 700 GB; Slurm refuses more) and runs job scripts from a queue directory, one per GPU or one on the whole
# node, so timing runs share a fixed node and nothing waits in the Slurm queue between them. README.md beside this file explains the queue.
#
#   QUEUE  queue root (default $WORK/jobs/gpu-campaign/queue): new/ holds the items, running/ and done/ are written here.
#   IDLE_S exit after this many seconds with nothing running and nothing queued (default 1800).
#
# An item is an executable bash script in new/, named <order>-<name>.g1.sh (one GPU) or <order>-<name>.g2.sh (the whole node). Items start
# in filename order. A g1 item takes the first free GPU; a g2 item waits until both GPUs are free, and no g1 item behind it starts first.
# Each item runs as its own srun step (its own cgroup, so CPU and memory accounting stay per item) with these variables set:
#   PILOT=1 PILOT_JOB PILOT_NODE PILOT_GPU (0, 1 or both) PILOT_SHARED (1 while the other GPU may be busy) RUNID=<job>-g<gpu>-<name>
#   and for a g1 item PINDEV=<gpu> CUDA_VISIBLE_DEVICES=<gpu>.
# An item may declare "# PILOT-MINUTES: <n>" (default 240); it is not started when less time than that is left in the allocation.
# Touch $QUEUE/STOP to end the pilot after the running items finish.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
Q=${QUEUE:-$W/jobs/gpu-campaign/queue}; IDLE_S=${IDLE_S:-1800}
mkdir -p $Q/new $Q/running $Q/done
NODE=$(hostname -s); T0=$(date +%s)
END=$(( T0 + $(squeue -h -j $SLURM_JOB_ID -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*86400+$2*3600+$3*60+$4); else if (NF==3) print ($1*3600+$2*60+$3); else print ($1*60+$2) }') ))
NCPU=${SLURM_CPUS_PER_TASK:-54}; HALF=$(( NCPU / 2 ))
MEMG=700; HALFMEM=$(( MEMG / 2 - 10 ))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
log() { echo "$(date -Is) $*" | tee -a $Q/pilot-$SLURM_JOB_ID.log; }
log "pilot job=$SLURM_JOB_ID node=$NODE cpus=$NCPU end=$(date -d @$END -Is) queue=$Q"
nvidia-smi --query-gpu=index,pci.bus_id,name,memory.total --format=csv,noheader | sed 's/^/  gpu /' | tee -a $Q/pilot-$SLURM_JOB_ID.log
[ -f $Q/pilot.tsv ] || printf 'item\tpilot_job\tnode\tgpu\tstart\tend\twall_s\texit\n' > $Q/pilot.tsv

declare -A PID ITEM START     # per lane: 0, 1, both
lane_busy() { [ -n "${PID[$1]:-}" ]; }
reap() {
  for l in 0 1 both; do
    [ -n "${PID[$l]:-}" ] || continue
    kill -0 ${PID[$l]} 2>/dev/null && continue
    wait ${PID[$l]}; rc=$?; it=${ITEM[$l]}; t1=$(date +%s)
    mv $Q/running/$it $Q/done/$it 2>/dev/null
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $it $SLURM_JOB_ID $NODE $l "$(date -d @${START[$l]} -Is)" "$(date -d @$t1 -Is)" $((t1 - START[$l])) $rc >> $Q/pilot.tsv
    log "done $it on gpu $l: exit $rc after $((t1 - START[$l])) s"
    unset "PID[$l]" "ITEM[$l]" "START[$l]"
  done
}
launch() {   # launch <lane> <item>
  local l=$1 it=$2 name=${2%.g?.sh} mins c m g env
  mins=$(grep -m1 -oP '^# PILOT-MINUTES:\s*\K[0-9]+' $Q/new/$it 2>/dev/null || echo 240)
  if [ $(( $(date +%s) + mins * 60 )) -gt $END ]; then log "skip $it: needs $mins min, $(( (END - $(date +%s)) / 60 )) left"; return 1; fi
  mv $Q/new/$it $Q/running/$it
  if [ $l = both ]; then c=$NCPU; m=$((MEMG - 10)); g=2; env="PILOT_GPU=both PILOT_SHARED=0 RUNID=$SLURM_JOB_ID-gboth-$name"
  else c=$HALF; m=$HALFMEM; g=1; env="PILOT_GPU=$l PILOT_SHARED=1 PINDEV=$l CUDA_VISIBLE_DEVICES=$l RUNID=$SLURM_JOB_ID-g$l-$name"; fi
  { echo "pilot job=$SLURM_JOB_ID node=$NODE gpu=$l item=$it cpus=$c mem=${m}G start=$(date -Is)"
    echo "env $env"; } > $Q/done/$name.log
  srun --exact -N1 -n1 -c $c --mem=${m}G --gres=gpu:nvidia_h100_nvl:$g --job-name=$name \
    env PILOT=1 PILOT_JOB=$SLURM_JOB_ID PILOT_NODE=$NODE $env bash $Q/running/$it >> $Q/done/$name.log 2>&1 &
  PID[$l]=$!; ITEM[$l]=$it; START[$l]=$(date +%s)
  log "start $it on gpu $l (pid ${PID[$l]}, $c cpus, ${m}G)"
}

idle_since=$(date +%s)
while :; do
  reap
  if [ -f $Q/STOP ]; then
    lane_busy 0 || lane_busy 1 || lane_busy both || { log "STOP: exiting"; break; }
    sleep 20; continue
  fi
  next=$(ls $Q/new 2>/dev/null | grep -E '\.g[12]\.sh$' | sort | head -1)
  if [ -n "$next" ]; then
    idle_since=$(date +%s)
    case $next in
      *.g2.sh) lane_busy 0 || lane_busy 1 || lane_busy both || launch both $next || mv $Q/new/$next $Q/done/$next.skipped ;;
      *.g1.sh) if ! lane_busy both; then
                 for l in 0 1; do lane_busy $l && continue; launch $l $next || mv $Q/new/$next $Q/done/$next.skipped; break; done
               fi ;;
    esac
  elif lane_busy 0 || lane_busy 1 || lane_busy both; then
    idle_since=$(date +%s)
  elif [ $(( $(date +%s) - idle_since )) -ge $IDLE_S ]; then
    log "queue empty for $IDLE_S s: exiting"; break
  fi
  sleep 20
done
