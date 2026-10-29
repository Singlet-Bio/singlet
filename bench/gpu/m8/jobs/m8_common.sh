# Shared helpers for the M8 jobs (sourced). Expects WORK, OUT, SCRATCH.
IMG=$WORK/images/cuda-12.4.1-devel-ubuntu22.04.sif
PB=$WORK/envs/pbench
M8=$WORK/jobs/w11/m8
BIN=$M8/bin/m8_ingest
PERF=$WORK/data/atlas-rerun/perf
A_SHARED=$PERF/GSM7656149/SRR25376447.1fq      # 3' v3, 263 M pairs
B_SHARED=$PERF/GSM6470208/sample.1fq           # 5' v2, 121.8 M pairs
C_SHARED=$PERF/GSM7036403/SRR23353990.1fq      # 3' v2, 111.6 M pairs
RUNS=$OUT/m8_runs.jsonl
mkdir -p $M8/lib; cp -n $PB/lib/libzstd.so.1.5.7 $M8/lib/; ln -sf libzstd.so.1.5.7 $M8/lib/libzstd.so.1   # runtime zstd 1.5.7 (the image ships 1.4.8)
m8() { apptainer exec --nv -B $WORK --env LD_LIBRARY_PATH=$M8/lib $IMG $BIN "$@"; }
# one pipeline run, one JSON line appended to $RUNS
run() { local label=$1; shift; echo "[$(date +%T)] run $label $*" >&2; m8 --label "$label" "$@" >> $RUNS || echo "FAILED rc=$? $label" >&2; }
# same, pinned to a GPU in a background process (dual-GPU runs)
runbg() { local gpu=$1 label=$2; shift 2; ( CUDA_VISIBLE_DEVICES=$gpu m8 --label "$label" "$@" >> $RUNS || echo "FAILED rc=$? $label" >&2 ) & }
busy_jiffies() { awk '/^cpu /{print $2+$3+$4+$7+$8+$9, $2+$3+$4+$5+$6+$7+$8+$9}' /proc/stat; }
# node-level CPU load around a command: node_wrap LABEL cmd...   -> JSON line with cores used node-wide
node_wrap() {
  local label=$1; shift
  read b0 t0 < <(busy_jiffies); local w0=$(date +%s.%N)
  "$@"
  read b1 t1 < <(busy_jiffies); local w1=$(date +%s.%N)
  awk -v l="$label" -v b0=$b0 -v b1=$b1 -v w0=$w0 -v w1=$w1 'BEGIN{printf "{\"node_load\":\"%s\",\"wall_s\":%.2f,\"node_cores_busy\":%.2f}\n", l, w1-w0, (b1-b0)/100.0/(w1-w0)}' >> $RUNS
}
envinfo() {
  echo "== node $(hostname) job ${SLURM_JOB_ID:-} $(date -Is)"
  echo "== taskset: $(taskset -cp $$)"; nproc
  lscpu | grep -E "Model name|Socket|Thread|Core|NUMA|MHz|L3"
  (numactl -H 2>/dev/null || true) | head -12
  nvidia-smi --query-gpu=index,name,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,clocks.max.sm,memory.total --format=csv
  nvidia-smi -q | grep -A3 "PCIe Generation\|Link Width" | head -40
  nvidia-smi topo -m
  apptainer exec -B $WORK $IMG nvcc --version | tail -2
  echo "== nsys: $(apptainer exec -B $WORK $IMG sh -c 'which nsys nsys-ui 2>&1; ls /opt/nvidia 2>&1 | head -3; ls /usr/local/cuda/bin | grep -i nsys' 2>&1 | tr '\n' ' ')"
  echo "== host nsys: $(which nsys 2>&1; ls -d /opt/nvidia/nsight-systems* /usr/local/cuda*/bin/nsys* 2>&1 | head -3; (source /etc/profile.d/lmod.sh 2>/dev/null; module -t avail 2>&1 | grep -i -E 'nsight|cuda' | head -5))"
  echo "== /tmp: $(df -h /tmp | tail -1)"; grep -E " /tmp | / " /proc/mounts | head -3
  echo "== shared fs mount: $(grep ' /mnt/projects ' /proc/mounts)"
  grep -E "MemTotal|MemFree|^Cached" /proc/meminfo
  uname -r
}
