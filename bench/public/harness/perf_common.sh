# Shared paths and helpers for the Singlet 1.0.0 public performance benchmarks on Clipper (sourced by the sbatch scripts).
# Capabilities: non-host, alignment-performance, download. Contract: bench/public/CONTRACT.md.
WORK=/mnt/projects/debruinz_project/debruinz/singlet
PB=$WORK/pipeline-benchmark                    # data, references, binaries of the earlier pipeline benchmark
BP=$WORK/bench-public                          # everything produced by the public benchmark
SRC=${BP_SRC:-$BP/src/singlet}                  # checkout of this branch (sync_src.sbatch); BP_SRC overrides
ART=$WORK/bpub/artifacts                        # $ART/<capability>/: the small tables the notebook downloads (CONTRACT.md section 5)
REFBASE=/mnt/projects/debruinz_project/cellarium/reference
ENVDIR=$WORK/envs/pbench                       # STAR 2.7.11b, alevin-fry, simpleaf, salmon, samtools, sra-tools, python+pandas
PY=$ENVDIR/bin/python
# The pinned production binary (commit 02aa9b9, PGO). Every job verifies the hash before it runs.
BIN=$PB/bin/singlet-campaign
BIN_SHA256=261d3a50ed624572418d30489dfd58904323f6e0d8820bd5260dfe7fb5038e6c
BIN_COMMIT=02aa9b938b54e0f92bba00c44c07b883a8660aa2
mkdir -p $BP/logs $ART

verify_bin() {
  local got; got=$(sha256sum "$BIN" | cut -d' ' -f1)
  [ "$got" = "$BIN_SHA256" ] || { echo "binary hash mismatch: $got"; exit 3; }
  echo "[bp] singlet binary $BIN sha256 ok"
}

# dataset id -> reference directory name under $REFBASE
species_of() {
  case $1 in
    hgmm*) echo GRCh38-GRCm39-barnyard ;;
    neuron*|nuclei*) echo GRCm39-2024-A ;;
    pbmc*|gsm5815984*|hbec*|gse166766*) echo GRCh38-2024-A ;;
    *) echo "no reference for $1" >&2; return 1 ;;
  esac
}

cpu_model() { lscpu | sed -n 's/^Model name: *//p'; }

# timed OUTDIR cmd args...: runs cmd under GNU time (-v -> OUTDIR/time.txt) with a 1 Hz sampler of the peak RSS of the
# whole process tree (memwatch.py -> OUTDIR/mem.json). Returns the command's exit code.
timed() {
  local out=$1; shift; mkdir -p "$out"
  /usr/bin/time -v -o "$out/time.txt" "$@" &
  local pid=$!
  $PY -I "$SRC/bench/public/harness/memwatch.py" "$pid" "$out/mem.json" &
  local mw=$!
  local rc=0; wait $pid || rc=$?
  wait $mw 2>/dev/null || true
  return $rc
}
