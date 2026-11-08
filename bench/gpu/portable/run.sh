#!/usr/bin/env bash
# One timing point: NRUNS copies of pbmc1k_v3 through `singlet --engine gpu`, K in flight per GPU, T threads each, pinned to each GPU of
# GPUS with CUDA_VISIBLE_DEVICES. The singlet command line and environment are those of Clipper's throughput harness
# (bench/gpu/pilot/tp_gc.sbatch, run_item; config "base"), so a point here compares with a Clipper point of the same tag and commit.
#
#   run.sh                       k8t7s2r5 on GPU 0: 40 runs, 8 in flight, 7 threads, 2 device slots
#   K=8 T=7 SLOTS=2 ROUNDS=5 GPUS="0 1" run.sh    one independent queue per GPU
#
#   DATA        data root written by fetch.sh (default /data): DATA/ref, DATA/pbmc1k_v3/pbmc1k_v3.1fq
#   K T SLOTS   runs in flight per GPU, --threads per run, SINGLET_ENGINE_GPU_SLOTS (default 8, 7, 2)
#   ROUNDS      K*ROUNDS runs per GPU (default 5); NRUNS overrides the total per GPU
#   GPUS        GPU indices, space separated (default 0)
#   NONHOST     auto (default: screen on when the indices exist), on (fail without them), off (--no-viral-screen --no-microbial-screen)
#   SCRATCH     local disk for run outputs (default DATA/scratch); KEEP=1 keeps each run's outputs
#   OUT         results directory (default DATA/results/tp-<UTC time>-<tag>)
#   EXTRA_ARGS  extra singlet arguments (a point with them is not comparable to Clipper's base point)
# Outputs in OUT: meta.txt (machine, GPU, driver, binary, commit, settings), runs.tsv (one row per run), summary.tsv (one row per GPU and
# one for the node), host_mem.tsv and gpu.csv (1-2 s samples), g<gpu>/c<item>/ (run.log.ts, time.txt, small output files), and a
# final RESULT line with the steady throughput in read pairs per second per GPU.
set -uo pipefail
DATA=${DATA:-/data}; REF=${REF:-$DATA/ref}; G=$REF/GRCh38-2024-A
FQ=${FQ:-$DATA/pbmc1k_v3/pbmc1k_v3.1fq}
K=${K:-8}; T=${T:-7}; SLOTS=${SLOTS:-2}; ROUNDS=${ROUNDS:-5}; NRUNS=${NRUNS:-$((K * ROUNDS))}
read -r -a GPUS <<< "${GPUS:-0}"
NONHOST=${NONHOST:-auto}; EXTRA_ARGS=${EXTRA_ARGS:-}; KEEP=${KEEP:-0}; TIMEOUT_S=${TIMEOUT_S:-3000}
BIN=${SINGLET_BIN:-$(command -v singlet || true)}
TAG=k${K}t${T}s${SLOTS}r$((NRUNS / K)); [ $((NRUNS % K)) = 0 ] || TAG=k${K}t${T}s${SLOTS}n$NRUNS
OUT=${OUT:-$DATA/results/tp-$(date -u +%Y%m%dT%H%M%SZ)-$TAG}
SCRATCH=${SCRATCH:-$DATA/scratch}; RUNSCR=$SCRATCH/tp-$$
mkdir -p "$OUT" "$RUNSCR"; M=$OUT/meta.txt; : > "$M"
say() { echo "$*" | tee -a "$M"; }
die() { say "STOP: $*"; exit 2; }

# --- environment of Clipper's harness -------------------------------------------------------------------------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export SINGLET_WHITELIST_DIR=${SINGLET_WHITELIST_DIR:-$(dirname "${BIN:-/x}")/../whitelists}
export SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1
unset SINGLET_ENGINE_GPU_CHUNK SINGLET_ENGINE_GPU_NB SINGLET_ENGINE_GPU_NBM SINGLET_ENGINE_GPU_NBB SINGLET_GPU_PILEUP_ARENA_GB SINGLET_ENGINE_GPU_DEVICE
export SINGLET_ENGINE_GPU_SLOTS=$SLOTS

# --- preflight: refuse what would make the timing meaningless, label what makes it not comparable ----------------------------------
[ -n "$BIN" ] && [ -x "$BIN" ] || die "no singlet binary (set SINGLET_BIN)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found (driver, or docker run --gpus all)"
[ -x /usr/bin/time ] || die "/usr/bin/time missing (apt-get install time)"
for f in "$FQ" "$G/star_2.7.11b/SA" "$G/star_2.7.11b/Genome" "$G/genes/genes.gtf"; do [ -s "$f" ] || die "missing $f (run fetch.sh)"; done
[ -s "$SINGLET_WHITELIST_DIR/3M-february-2018.txt" ] || die "no 3M-february-2018.txt in $SINGLET_WHITELIST_DIR: the pileup would take the sorted-BAM path"
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$BIN" gpu-probe > "$OUT/gpu-probe.txt" 2>&1 || die "singlet gpu-probe: $(cat "$OUT/gpu-probe.txt")"
NOTES=()
NH_ARGS=""
NH_OK=1; for f in viral.snhskidx bacterial.snhskidx fungal.snhskidx host_21mer.bloom; do [ -s "$REF/nonhost/$f" ] || NH_OK=0; done
case $NONHOST in
  on) [ $NH_OK = 1 ] || die "NONHOST=on but $REF/nonhost is incomplete" ;;
  off) NH_ARGS="--no-viral-screen --no-microbial-screen"; NOTES+=("nonhost=off") ;;
  auto) if [ $NH_OK = 0 ]; then NH_ARGS="--no-viral-screen --no-microbial-screen"; NOTES+=("nonhost=off(indices missing)"); fi ;;
  *) die "NONHOST must be auto, on or off" ;;
esac
[ -s "$REF/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz" ] || NOTES+=("snp_panel=missing(no donor demux)")
[ -n "$EXTRA_ARGS" ] && NOTES+=("extra_args=$EXTRA_ARGS")
[ "$(stat -c %s "$FQ")" = 1838043689 ] || NOTES+=("1fq_bytes=$(stat -c %s "$FQ")(Clipper 1838043689)")
COMPARABLE=yes; for n in "${NOTES[@]:-}"; do case $n in nonhost=*|snp_panel=*|extra_args=*) COMPARABLE=no ;; esac; done

NG=${#GPUS[@]}; MEMG=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo); NCPU=$(nproc)
SHMG=$(df -BG --output=size /dev/shm 2>/dev/null | tail -1 | tr -dc 0-9)
[ "$MEMG" -ge $((NG * K * 45)) ] || say "WARNING host RAM ${MEMG} GB for $((NG * K)) runs in flight: Clipper ran 8 in flight per GPU in 370 GB (peaks 300-357 GB)"
[ "$NCPU" -ge $((NG * 28)) ] || say "WARNING $NCPU CPUs for $NG GPU(s): Clipper gives each GPU 28 cores (k8t7 = 56 threads on 28 cores)"
[ "${SHMG:-0}" -ge 32 ] || say "WARNING /dev/shm is ${SHMG:-?} GB (docker: --ipc=host)"

COMMIT=$(cat "$(dirname "$BIN")/../COMMIT" 2>/dev/null || echo unknown)
EMBED=no; [ "$COMMIT" != unknown ] && grep -aqF "$COMMIT" "$BIN" && EMBED=yes
BSHA=$(sha256sum "$BIN" | cut -d' ' -f1)
GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPUS[0]}" | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "${GPUS[0]}" | head -1)
PAIRS0=$("$BIN" decode "$FQ" --header 2>&1 | sed -n 's/.*Reads: *\([0-9]*\).*/\1/p' | head -1)
say "PROVENANCE host=$(hostname) bin=$BIN sha256=$BSHA commit=$COMMIT embedded=$EMBED tag=$TAG gpus=${GPUS[*]} $(date -Is)"
say "settings K=$K T=$T SLOTS=$SLOTS NRUNS=$NRUNS per GPU; env SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_SLOTS=$SLOTS; args $NH_ARGS $EXTRA_ARGS"
say "input $FQ ($(stat -c %s "$FQ") bytes, ${PAIRS0:-?} read pairs); ref $REF"
say "comparable_to_clipper_base=$COMPARABLE ${NOTES[*]:-}"
{ nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version --format=csv,noheader
  nvidia-smi | grep -m1 -o 'CUDA Version: [0-9.]*'
  lscpu | grep -E "^Model name|^CPU\(s\)|^Thread\(s\) per core|^Core\(s\) per socket|^Socket\(s\)|^NUMA node[0-9]"
  free -g | head -2; df -h "$SCRATCH" /dev/shm | tail -n +1
  echo "transparent_hugepage $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
  cat "$(dirname "$BIN")/../build-info.txt" 2>/dev/null | head -8; } 2>&1 | sed 's/^/  /' | tee -a "$M"

# --- one run, exactly as Clipper's run_item -------------------------------------------------------------------------------------------
next_item() { ( flock -x 9; n=$(cat "$1"); echo $((n + 1)) > "$1"; echo "$n" ) 9>>"$1.lock"; }
run_item() {   # gpu item worker
  local g=$1 i=$2 w=$3 D=$RUNSCR/g$1/c$2 O=$OUT/g$1/c$2 T0 T1 rc
  mkdir -p "$D/out" "$O"
  T0=$(date +%s.%N)
  ( cd "$D" && env E18_COPY=c$i /usr/bin/time -v -o "$O/time.txt" timeout -k 30 "$TIMEOUT_S" "$BIN" "$FQL" --ref-base "$REF" \
      --genome-dir "$G/star_2.7.11b" --exons "$G/genes/genes.gtf" --out-prefix "$D/out/" --threads "$T" --pipeline --output-format 1pz \
      --engine gpu $NH_ARGS $EXTRA_ARGS 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > "$O/run.log.ts" )
  rc=$?; T1=$(date +%s.%N)
  cp "$D"/out/*Log.final.out "$D"/out/*Log.out "$D/out/summary.json" "$D/out/pileup_stats.json" "$D/out/read_stats.tsv" "$O/" 2>/dev/null
  local L=$O/run.log.ts us ss rss pairs l5 gp st fb
  read -r us ss rss < <(awk -F': ' '/User time/{u=$2} /System time/{s=$2} /Maximum resident/{m=$2} END{print u+0, s+0, m/1048576}' "$O/time.txt" 2>/dev/null)
  pairs=$(grep -aoE '\.1fq: [0-9]+ reads' "$L" | head -1 | tr -dc 0-9); pairs=${pairs:-${PAIRS0:-0}}
  grep -aq 'Solo on the device' "$L" && l5=yes || l5=no
  grep -aq '\[gpu_pileup\] on:' "$L" && gp=yes || gp=no
  grep -aq 'unsorted STAR stream into the pileup' "$L" && st=yes || st=no
  # fallback lines: every "falls back" / "on the CPU" line except the engine's announcement at start
  fb=$(grep -aiE 'fall(s)? ?back|on the CPU' "$L" | grep -avc 'each falls back on its own')
  printf '%s\t%d\t%d\t%.3f\t%.3f\t%.3f\t%d\t%.2f\t%.2f\t%.2f\t%.1f\t%s\t%s\t%s\t%s\t%s\n' "$g" "$i" "$w" "$T0" "$T1" \
    "$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.3f", b-a}')" "$rc" "${us:-0}" "${ss:-0}" \
    "$(awk -v a="${us:-0}" -v b="${ss:-0}" 'BEGIN{print a+b}')" "${rss:-0}" "$pairs" "$l5" "$gp" "$st" "$fb" >> "$OUT/runs.tsv"
  [ "$KEEP" = 1 ] || rm -rf "$D"
}
worker() { local g=$1 w=$2 i; while :; do i=$(next_item "$RUNSCR/g$g/queue"); [ "$i" -ge "$NRUNS" ] && break; run_item "$g" "$i" "$w"; done; }

# --- the point ------------------------------------------------------------------------------------------------------------------------
FQL=$RUNSCR/pbmc1k_v3.1fq; cp "$FQ" "$FQL.part" && mv "$FQL.part" "$FQL"    # staged on the scratch disk, as the harness stages to /tmp
rm -f /tmp/singlet_engine_gpu_*.lock /tmp/singlet_admit_*.tab 2>/dev/null
printf 'gpu\titem\tworker\tstart\tend\twall_s\texit\tuser_s\tsys_s\tcore_s\tmaxrss_gb\tpairs\tlevel5\tgpu_pileup\tstream\tfallback_lines\n' > "$OUT/runs.tsv"
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "${RUNSCR:?}"' EXIT INT TERM
( while :; do awk -v t="$(date +%s)" '/MemTotal/{a=$2} /MemAvailable/{b=$2} END{printf "%s\t%.1f\n", t, (a-b)/1048576}' /proc/meminfo; sleep 2; done ) > "$OUT/host_mem.tsv" &
SMP=$!
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$OUT/gpu.csv" 2>&1 &
NVS=$!
cg0=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null); st0=$(awk '/^cpu /{print $2+$3+$4+$7+$8}' /proc/stat)
say "######## $TAG on GPU(s) ${GPUS[*]}: $NRUNS runs per GPU, $K in flight, start $(date -Is)"
BP=()
for g in "${GPUS[@]}"; do
  mkdir -p "$RUNSCR/g$g"; echo 0 > "$RUNSCR/g$g/queue"
  ( export CUDA_VISIBLE_DEVICES=$g; WP=(); for ((w = 0; w < K; w++)); do worker "$g" "$w" & WP+=($!); done; wait "${WP[@]}" ) &
  BP+=($!)
done
wait "${BP[@]}"
cg1=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null); st1=$(awk '/^cpu /{print $2+$3+$4+$7+$8}' /proc/stat)
kill $SMP $NVS 2>/dev/null; wait $SMP $NVS 2>/dev/null
say "end $(date -Is); device lock files: $(ls /tmp/singlet_engine_gpu_*.lock 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
CG_CORE_S=""; [ -n "$cg0" ] && [ -n "$cg1" ] && CG_CORE_S=$(awk -v a="$cg0" -v b="$cg1" 'BEGIN{printf "%.1f", (b-a)/1e6}')
SYS_CORE_S=$(awk -v a="$st0" -v b="$st1" -v hz="$(getconf CLK_TCK)" 'BEGIN{printf "%.1f", (b-a)/hz}')

# --- summary: span and steady rates as bench/fable/throughput/e18_steady.py computes them -------------------------------------------
python3 - "$OUT" "$K" "$T" "$SLOTS" "$TAG" "$COMMIT" "$GPUNAME" "$DRIVER" "$COMPARABLE" "${CG_CORE_S:-}" "$SYS_CORE_S" "$(hostname)" <<'PY' | tee -a "$M"
import sys, os, csv, collections
out, K, T, S, tag, commit, gpu, drv, comp, cgs, syss, host = sys.argv[1:13]
K = int(K)
runs = list(csv.DictReader(open(os.path.join(out, "runs.tsv")), delimiter="\t"))
for r in runs:
    for k in ("start", "end", "wall_s", "core_s", "maxrss_gb"): r[k] = float(r[k])
    for k in ("exit", "pairs", "fallback_lines"): r[k] = int(r[k])
mem = [float(l.split("\t")[1]) for l in open(os.path.join(out, "host_mem.tsv")) if "\t" in l]
host_peak = (max(mem) - mem[0]) if mem else 0.0
util = collections.defaultdict(list); dmem = collections.defaultdict(list)
for l in open(os.path.join(out, "gpu.csv")):
    f = [x.strip() for x in l.split(",")]
    if len(f) == 4 and f[2].replace(".", "").isdigit():
        util[f[1]].append(float(f[2])); dmem[f[1]].append(float(f[3]))
def point(rs, k):
    ok = [r for r in rs if r["exit"] == 0]
    t0 = min(r["start"] for r in rs); t1 = max(r["end"] for r in rs); span = t1 - t0
    pairs = sum(r["pairs"] for r in ok)
    first_end = min(r["end"] for r in rs)
    later = [r for r in ok if r["start"] >= first_end - 1.0]
    steady = (k * (sum(r["pairs"] for r in later) / len(later)) / (sum(r["wall_s"] for r in later) / len(later))) if later and len(rs) > k else 0.0
    walls = [r["wall_s"] for r in ok] or [0.0]
    return dict(runs=len(rs), failed=len(rs) - len(ok), span=span, pairs=pairs, span_rate=pairs / span if span else 0.0, steady=steady,
                steady_n=len(later), wmin=min(walls), wmean=sum(walls) / len(walls), wmax=max(walls),
                core_per_M=(sum(r["core_s"] for r in ok) / (pairs / 1e6)) if pairs else 0.0, rss=max(r["maxrss_gb"] for r in rs),
                fb_runs=sum(1 for r in rs if r["fallback_lines"] > 0), notl5=sum(1 for r in ok if r["level5"] != "yes"),
                nogp=sum(1 for r in ok if r["gpu_pileup"] != "yes"), nost=sum(1 for r in ok if r["stream"] != "yes"))
hdr = ["host", "gpu_model", "driver", "commit", "tag", "gpu", "k", "T", "slots", "runs", "failed_runs", "span_s", "M_pairs", "span_rate_Mps",
       "steady_rate_Mps", "steady_runs", "wall_min_s", "wall_mean_s", "wall_max_s", "runs_core_s_per_M", "cgroup_core_s_per_M",
       "max_rss_gb", "host_peak_gb", "dev_peak_mb", "gpu_util_mean", "fallback_runs", "not_level5", "no_gpu_pileup", "no_stream", "comparable"]
rows = []; steadies = []
by = collections.defaultdict(list)
for r in runs: by[r["gpu"]].append(r)
for g in sorted(by):
    p = point(by[g], K); steadies.append(p["steady"])
    rows.append([host, gpu, drv, commit, tag, g, K, T, S, p["runs"], p["failed"], f'{p["span"]:.1f}', f'{p["pairs"]/1e6:.1f}', f'{p["span_rate"]/1e6:.3f}',
                 f'{p["steady"]/1e6:.3f}', p["steady_n"], f'{p["wmin"]:.1f}', f'{p["wmean"]:.1f}', f'{p["wmax"]:.1f}', f'{p["core_per_M"]:.2f}', "",
                 f'{p["rss"]:.1f}', "", f'{max(dmem[g]) if dmem[g] else 0:.0f}', f'{sum(util[g])/len(util[g]) if util[g] else 0:.0f}',
                 p["fb_runs"], p["notl5"], p["nogp"], p["nost"], comp])
    print(f'gpu {g}: {p["runs"]} runs ({p["failed"]} failed), span {p["span"]:.1f} s, span rate {p["span_rate"]/1e6:.3f} M pairs/s, '
          f'steady {p["steady"]/1e6:.3f} M pairs/s ({p["steady_n"]} runs), wall {p["wmin"]:.1f} / {p["wmean"]:.1f} / {p["wmax"]:.1f} s, '
          f'{p["core_per_M"]:.2f} core-s/M (runs), fallback runs {p["fb_runs"]}, not level 5 {p["notl5"]}, no GPU pileup {p["nogp"]}, no stream {p["nost"]}')
ng = len(by)
P = point(runs, K)
st_per_gpu = sum(steadies) / ng if ng else 0.0          # pairs/s
cg_per_M = (float(cgs) / (P["pairs"] / 1e6)) if cgs and P["pairs"] else 0.0
rows.append([host, gpu, drv, commit, tag, "node", K, T, S, P["runs"], P["failed"], f'{P["span"]:.1f}', f'{P["pairs"]/1e6:.1f}', f'{P["span_rate"]/1e6:.3f}',
             f'{st_per_gpu*ng/1e6:.3f}', "", f'{P["wmin"]:.1f}', f'{P["wmean"]:.1f}', f'{P["wmax"]:.1f}', f'{P["core_per_M"]:.2f}',
             f'{cg_per_M:.2f}' if cg_per_M else "", f'{P["rss"]:.1f}', f'{host_peak:.0f}', "", "", P["fb_runs"], P["notl5"], P["nogp"], P["nost"], comp])
with open(os.path.join(out, "summary.tsv"), "w") as fh:
    fh.write("\t".join(hdr) + "\n")
    for r in rows: fh.write("\t".join(str(x) for x in r) + "\n")
print(f'node: {ng} GPU(s), span rate {P["span_rate"]/1e6:.3f} M pairs/s, host peak {host_peak:.0f} GB above idle, '
      f'cgroup core-s/M {cg_per_M:.2f}' + (f', system busy core-s {syss}' if syss else ""))
print(f'RESULT steady {st_per_gpu/1e6:.3f} M read pairs/s per GPU ({st_per_gpu:.0f} pairs/s); span {P["span_rate"]/ng/1e6:.3f} M/s per GPU; '
      f'{tag} on {ng} x {gpu} (driver {drv}); commit {commit[:12]}; comparable_to_clipper_base={comp}')
PY
say "results: $OUT"
