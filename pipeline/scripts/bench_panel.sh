#!/bin/bash
# bench_panel.sh — 5-dataset benchmark panel for singlify pipeline.
#
# Submits 5 independent SLURM jobs (one per dataset) and collects results.
# Designed for the singlify orchestrator to avoid overfitting to any single
# protocol or assay.
#
# Usage:
#   bash bench_panel.sh submit [--threads 20]    # submit 5 SLURM jobs
#   bash bench_panel.sh collect                   # aggregate results
#   bash bench_panel.sh run INDEX [--threads 20]  # run single dataset (called by SLURM)
#   bash bench_panel.sh ssh NODE [--threads 20]   # run all 5 sequentially via SSH
#
# Panel (5 protocols, 2 species):
#   0: SRR32855204  10x-arc-gex   40.4M  Human  (primary, has ground truth)
#   1: SRR17873408  ddSEQ         55.8M  Human  (linker-based barcode)
#   2: SRR23582977  sci-RNA-seq3  48.1M  Human  (combinatorial indexing)
#   3: SRR12062565  Drop-seq      50.2M  Human  (classic open protocol, 69 cells/67.8% mapping)
#   4: SRR33424030  10xv3         11.1M  Mouse  (species canary, lung PM2.5)

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────

SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
FQ_DIR=/mnt/projects/debruinz_project/singlify_validation/1fq
WL_DIR=/mnt/home/debruinz/Singlet-AI/singlify/whitelists
REF_DIR=/mnt/projects/debruinz_project/cellarium/reference
RESULTS_DIR=/mnt/home/debruinz/Singlet-AI/singlify/scripts/bench_results
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

THREADS=20

# Dataset config: SRR|PROTOCOL|GENOME_KEY|WHITELIST_FILE|READS_APPROX|R1_FASTQ|R2_FASTQ
# R1_FASTQ/R2_FASTQ: relative to FQ_DIR. "None" = use pre-encoded .1fq only.
# singlify deletes its input .1fq after FIFO decode (by design). When FASTQs
# are provided, we encode fresh to /dev/shm before each run to preserve the source.
DATASETS=(
  "SRR32855204|10x-arc-gex|GRCh38|None|40.4M|SRR32855204_v3fix.1fq.R1.fastq|SRR32855204_v3fix.1fq.R2.fastq"
  "SRR17873408|ddSEQ|GRCh38|None|55.8M|None|None"
  "SRR23582977|sci-RNA-seq3|GRCh38|None|48.1M|None|None"
  "SRR12062565|Drop-seq|GRCh38|None|50.2M|None|None"
  "SRR33424030|10xv3-mouse|GRCm39|None|11.1M|None|None"
)

# ─── Helper functions ────────────────────────────────────────────────────────

resolve_genome() {
    local key="$1"
    case "$key" in
        GRCh38) echo "${REF_DIR}/GRCh38-2024-A" ;;
        GRCm39) echo "${REF_DIR}/GRCm39-2024-A" ;;
        *) echo "ERROR: unknown genome key $key" >&2; exit 1 ;;
    esac
}

resolve_whitelist() {
    local wl="$1"
    if [[ "$wl" == "None" ]]; then
        echo "None"
    else
        echo "${WL_DIR}/${wl}"
    fi
}

run_one() {
    local idx="$1"
    local threads="${2:-$THREADS}"

    IFS='|' read -r srr protocol genome_key wl_file reads_approx r1_fastq r2_fastq <<< "${DATASETS[$idx]}"

    local ref_base
    ref_base="$(resolve_genome "$genome_key")"
    local genome="${ref_base}/star_2.7.11b"
    local gtf="${ref_base}/genes/genes.gtf"
    local wl
    wl="$(resolve_whitelist "$wl_file")"
    local nfs_fq="${FQ_DIR}/${srr}.1fq"
    local outdir="/dev/shm/bench_panel_${srr}_$$"

    mkdir -p "$outdir" "$RESULTS_DIR"

    echo "=== BENCH PANEL [$idx]: $srr ($protocol, $reads_approx reads, ${threads}T) ==="
    echo "  genome: $genome"
    echo "  whitelist: $wl"

    # singlify deletes its input .1fq after FIFO decode. To preserve NFS files,
    # encode to /dev/shm first when FASTQ sources are provided.
    local fq
    if [[ "$r1_fastq" != "None" && -f "${FQ_DIR}/${r1_fastq}" ]]; then
        fq="/dev/shm/bench_panel_${srr}_$$.1fq"
        echo "  encoding ${srr} from FASTQs to /dev/shm..."
        if [[ "$r2_fastq" != "None" ]]; then
            "$SINGLIFY" encode --reads "${FQ_DIR}/${r1_fastq}" "${FQ_DIR}/${r2_fastq}" \
                -o "$fq" 2>&1 | grep -E "Done|Protocol|Error" || true
        else
            # Single R1 = already-encoded .1fq acting as R1 source
            cp "${FQ_DIR}/${r1_fastq}" "$fq"
        fi
        echo "  .1fq: $fq (from FASTQs)"
    elif [[ -f "$nfs_fq" ]]; then
        # Copy NFS .1fq to /dev/shm to prevent deletion of the NFS original
        echo "  copying NFS .1fq to /dev/shm to preserve original..."
        fq="/dev/shm/bench_panel_${srr}_$$.1fq"
        cp "$nfs_fq" "$fq"
        echo "  .1fq: $fq (copied from NFS)"
    else
        echo "ERROR: no .1fq or FASTQ sources found for $srr"
        echo "${srr}|${protocol}|ERROR|missing_inputs|$(date -Iseconds)" > "${RESULTS_DIR}/${srr}.tsv"
        return 1
    fi

    # Pre-load genome into shared memory so STAR uses LoadAndKeep (saves ~32GB RSS
    # and ~40s per run; matches production pipeline architecture per CLAUDE.md).
    echo "  loading genome into shared memory..."
    "$SINGLIFY" genome load "$genome" 2>&1 | tail -3

    # Build pipeline command
    local cmd=("$SINGLIFY" "$fq"
        --genome-dir "$genome"
        --genome-shared
        --exons "$gtf"
        --out-prefix "${outdir}/"
        --threads "$threads")
    # Only pass --whitelist when explicitly set (not "None")
    if [[ "$wl" != "None" ]]; then
        cmd+=(--whitelist "$wl")
    fi

    # Run with timing
    echo "  running pipeline..."
    local start_ts end_ts
    start_ts=$(date +%s.%N)

    local tmpout tmptime
    tmpout=$(mktemp /tmp/bench_out_XXXXXX)
    tmptime=$(mktemp /tmp/bench_time_XXXXXX)
    local exit_code=0
    /usr/bin/time -f "TIME_WALL=%e TIME_USER=%U TIME_SYS=%S TIME_RSS=%M" \
        "${cmd[@]}" >"$tmpout" 2>"$tmptime" || exit_code=$?
    local time_output
    time_output=$(cat "$tmpout" "$tmptime")
    rm -f "$tmpout" "$tmptime"

    end_ts=$(date +%s.%N)

    # Parse timing
    local wall_s user_s sys_s rss_kb
    wall_s=$(echo "$time_output" | grep -oP 'TIME_WALL=\K[0-9.]+' || echo "NA")
    user_s=$(echo "$time_output" | grep -oP 'TIME_USER=\K[0-9.]+' || echo "NA")
    sys_s=$(echo "$time_output" | grep -oP 'TIME_SYS=\K[0-9.]+' || echo "NA")
    rss_kb=$(echo "$time_output" | grep -oP 'TIME_RSS=\K[0-9]+' || echo "NA")

    # Parse pipeline internal timing (if available)
    local decode_s align_s pileup_s export_s
    decode_s=$(echo "$time_output" | grep -oP 'Decode.*?([0-9.]+)s' | grep -oP '[0-9.]+(?=s)' | tail -1 || echo "NA")
    align_s=$(echo "$time_output" | grep -oP 'STAR.*?([0-9.]+)s' | grep -oP '[0-9.]+(?=s)' | tail -1 || echo "NA")
    pileup_s=$(echo "$time_output" | grep -oP 'Pileup.*?([0-9.]+)s' | grep -oP '[0-9.]+(?=s)' | tail -1 || echo "NA")
    export_s=$(echo "$time_output" | grep -oP 'Export.*?([0-9.]+)s' | grep -oP '[0-9.]+(?=s)' | tail -1 || echo "NA")

    # Count outputs (auto_barcodes.tsv = auto cell-calling; exon_counts_barcodes.tsv = whitelist path)
    local n_bcs cell_caller exon_nnz
    n_bcs=0
    cell_caller="unknown"
    # Check Solo.out filtered barcodes first (STAR Solo cell caller — 10x/arc-gex)
    # singlify prefixes all STAR output files with "star_", so Solo.out is at star_Solo.out
    # Path from commit 6959027: {out_prefix}/star_Solo.out/Gene/filtered/barcodes.tsv
    local solo_bcs="${outdir}/star_Solo.out/Gene/filtered/barcodes.tsv"
    # Fallback: try GeneFull/ (alternate soloFeatures config)
    [[ -f "$solo_bcs" ]] || solo_bcs="${outdir}/star_Solo.out/GeneFull/filtered/barcodes.tsv"
    if [[ -f "$solo_bcs" ]]; then
        n_bcs=$(wc -l < "$solo_bcs")
        cell_caller="star_solo"
    elif [[ -f "${outdir}/auto_barcodes.tsv" ]]; then
        n_bcs=$(wc -l < "${outdir}/auto_barcodes.tsv")
        cell_caller="emptydrops"
    elif [[ -f "${outdir}/exon_counts_barcodes.tsv" ]]; then
        n_bcs=$(wc -l < "${outdir}/exon_counts_barcodes.tsv")
        cell_caller="whitelist"
    fi
    exon_nnz=$(ls "${outdir}/"exon_counts*.1pz 2>/dev/null | wc -l)

    # Get STAR mapping stats
    local unique_pct multi_pct
    unique_pct=$(grep "Uniquely mapped reads %" "${outdir}/star_Log.final.out" 2>/dev/null | awk '{print $NF}' || echo "NA")
    multi_pct=$(grep "% of reads mapped to multiple loci" "${outdir}/star_Log.final.out" 2>/dev/null | awk '{print $NF}' || echo "NA")

    # Write result
    local result_file="${RESULTS_DIR}/${srr}.tsv"
    cat > "$result_file" << EOF
srr	protocol	reads_approx	threads	wall_s	user_s	sys_s	rss_kb	decode_s	align_s	pileup_s	export_s	n_bcs	cell_caller	unique_pct	multi_pct	node	timestamp	exit_code
${srr}	${protocol}	${reads_approx}	${threads}	${wall_s}	${user_s}	${sys_s}	${rss_kb}	${decode_s}	${align_s}	${pileup_s}	${export_s}	${n_bcs}	${cell_caller}	${unique_pct}	${multi_pct}	$(hostname)	$(date -Iseconds)	${exit_code}
EOF

    echo ""
    echo "  RESULT: wall=${wall_s}s, user=${user_s}s, BCs=${n_bcs}, cell_caller=${cell_caller}, unique=${unique_pct}, exit=${exit_code}"
    echo "  Written to: $result_file"

    # Unload genome from shared memory (good neighbour — free shm for next job)
    "$SINGLIFY" genome unload "$genome" 2>&1 | tail -2 || true

    # Cleanup (singlify may have already deleted the .1fq; rm -f handles that cleanly)
    rm -f "$fq"
    rm -rf "$outdir"
    return 0
}

# ─── Commands ────────────────────────────────────────────────────────────────

cmd_submit() {
    local threads="$THREADS"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --threads) threads="$2"; shift 2 ;;
            *) echo "Unknown: $1"; exit 1 ;;
        esac
    done

    mkdir -p "$RESULTS_DIR"

    echo "Submitting 5 benchmark jobs (${threads}T each)..."
    local job_ids=()

    for idx in 0 1 2 3 4; do
        IFS='|' read -r srr protocol _ _ reads <<< "${DATASETS[$idx]}"

        local job_script
        job_script=$(mktemp /tmp/bench_panel_${srr}_XXXXXX.sh)
        cat > "$job_script" << SLURM
#!/bin/bash
#SBATCH --job-name=bench_${srr}
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${threads}
#SBATCH --mem=192G
#SBATCH --time=01:30:00
#SBATCH --output=${RESULTS_DIR}/slurm_${srr}_%j.out

source /opt/rh/gcc-toolset-13/enable
export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:\$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:\${LD_LIBRARY_PATH:-}
ulimit -n 10240

bash ${SCRIPT_DIR}/bench_panel.sh run ${idx} --threads ${threads}
SLURM

        local jid
        jid=$(sbatch --parsable "$job_script")
        job_ids+=("$jid")
        echo "  [$idx] $srr ($protocol, $reads): SLURM job $jid"
        rm -f "$job_script"
    done

    echo ""
    echo "All jobs submitted. Monitor with:"
    echo "  squeue -u \$USER | grep bench_"
    echo ""
    echo "Collect results with:"
    echo "  bash ${SCRIPT_DIR}/bench_panel.sh collect"
    echo ""
    echo "Job IDs: ${job_ids[*]}"

    # Save job IDs for collection
    printf "%s\n" "${job_ids[@]}" > "${RESULTS_DIR}/pending_jobs.txt"
}

cmd_ssh() {
    local node="$1"; shift
    local threads="$THREADS"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --threads) threads="$2"; shift 2 ;;
            *) echo "Unknown: $1"; exit 1 ;;
        esac
    done

    echo "Running all 5 benchmarks sequentially on $node (${threads}T)..."
    for idx in 0 1 2 3 4; do
        ssh "$node" "
            source /opt/rh/gcc-toolset-13/enable
            export TMPDIR=/dev/shm
            export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:\$PATH
            export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:\${LD_LIBRARY_PATH:-}
            ulimit -n 10240
            bash ${SCRIPT_DIR}/bench_panel.sh run $idx --threads $threads
        "
    done

    echo ""
    echo "All 5 complete. Collecting results..."
    cmd_collect
}

cmd_run() {
    local idx="$1"; shift
    local threads="$THREADS"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --threads) threads="$2"; shift 2 ;;
            *) echo "Unknown: $1"; exit 1 ;;
        esac
    done
    run_one "$idx" "$threads"
}

cmd_collect() {
    echo "=== BENCHMARK PANEL RESULTS ==="
    echo ""

    if [[ ! -d "$RESULTS_DIR" ]]; then
        echo "No results directory found: $RESULTS_DIR"
        return 1
    fi

    # Check for pending SLURM jobs
    if [[ -f "${RESULTS_DIR}/pending_jobs.txt" ]]; then
        local pending=0
        while read -r jid; do
            if squeue -j "$jid" &>/dev/null 2>&1; then
                local state
                state=$(squeue -j "$jid" -h -o "%T" 2>/dev/null)
                if [[ -n "$state" && "$state" != "" ]]; then
                    echo "  Job $jid still $state"
                    pending=$((pending + 1))
                fi
            fi
        done < "${RESULTS_DIR}/pending_jobs.txt"
        if [[ $pending -gt 0 ]]; then
            echo "WARNING: $pending jobs still running"
            echo ""
        fi
    fi

    # Print header
    printf "%-15s %-14s %6s %8s %8s %7s %10s %s\n" \
        "SRR" "Protocol" "Reads" "Wall(s)" "User(s)" "BCs" "Unique%" "Node"
    printf "%s\n" "$(printf '=%.0s' {1..85})"

    local total_wall=0
    local count=0

    for idx in 0 1 2 3 4; do
        IFS='|' read -r srr protocol _ _ reads <<< "${DATASETS[$idx]}"
        local rf="${RESULTS_DIR}/${srr}.tsv"

        if [[ -f "$rf" ]]; then
            local wall user bcs uniq node
            wall=$(tail -1 "$rf" | cut -f5)
            user=$(tail -1 "$rf" | cut -f6)
            bcs=$(tail -1 "$rf" | cut -f13)
            uniq=$(tail -1 "$rf" | cut -f14)
            node=$(tail -1 "$rf" | cut -f16)
            local ecode
            ecode=$(tail -1 "$rf" | cut -f18)

            local status="OK"
            [[ "$ecode" != "0" && "$ecode" != "" ]] && status="FAIL"

            printf "%-15s %-14s %6s %8s %8s %7s %10s %s %s\n" \
                "$srr" "$protocol" "$reads" "$wall" "$user" "$bcs" "$uniq" "$node" "$status"

            if [[ "$wall" != "NA" && "$wall" != "" ]]; then
                total_wall=$(echo "$total_wall + $wall" | bc 2>/dev/null || echo "$total_wall")
                count=$((count + 1))
            fi
        else
            printf "%-15s %-14s %6s %8s\n" "$srr" "$protocol" "$reads" "PENDING"
        fi
    done

    printf "%s\n" "$(printf '-%.0s' {1..85})"
    if [[ $count -gt 0 ]]; then
        printf "%-15s %-14s %6s %8s  (%d/%d complete)\n" \
            "TOTAL" "" "" "$total_wall" "$count" "5"
    fi
    echo ""
    echo "Results in: $RESULTS_DIR/"
}

# ─── Main ────────────────────────────────────────────────────────────────────

case "${1:-help}" in
    submit)  shift; cmd_submit "$@" ;;
    ssh)     shift; cmd_ssh "$@" ;;
    run)     shift; cmd_run "$@" ;;
    collect) shift; cmd_collect "$@" ;;
    *)
        echo "Usage: bench_panel.sh {submit|ssh|run|collect}"
        echo ""
        echo "  submit [--threads N]    Submit 5 SLURM jobs"
        echo "  ssh NODE [--threads N]  Run all 5 sequentially via SSH"
        echo "  run INDEX [--threads N] Run single dataset (0-4)"
        echo "  collect                 Aggregate results"
        exit 1
        ;;
esac
