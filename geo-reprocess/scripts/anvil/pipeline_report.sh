#!/usr/bin/env bash
# ── Pipeline Status Email Report ─────────────────────────────────────
# Generates a comprehensive status report and emails it.
# Can be called standalone or sourced by orchestrator.sh
#
# Usage:
#   bash pipeline_report.sh                    # send email
#   bash pipeline_report.sh --stdout           # print to stdout only
#   source pipeline_report.sh; send_report     # use from orchestrator
# ─────────────────────────────────────────────────────────────────────

REPORT_EMAIL="${REPORT_EMAIL:-debruinz@gvsu.edu}"
PROJECT="${PROJECT:-/anvil/projects/x-bio260157}"
SCRATCH="${SCRATCH:-/anvil/scratch/x-zdebruine}"
DL_DIR="${DL_DIR:-$SCRATCH/scgeo_downloads}"
QUANT_DIR="$PROJECT/scgeo/pipeline/quant"
RESULTS_DIR="$PROJECT/scgeo/pipeline/results"
LOG_DIR="$PROJECT/scgeo/pipeline/logs"
CLAIMS_DIR="$PROJECT/scgeo/pipeline/claims"

generate_report() {
    local TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')
    local UPTIME_S="${1:-0}"

    # ── Scratch storage ───────────────────────────────────────────────
    local DISK_GB=$(du -s --block-size=1G "$DL_DIR" 2>/dev/null | cut -f1)
    local DISK_PCT=$(awk "BEGIN{printf \"%.2f\", $DISK_GB/1000}")  # TB
    local QUOTA_TB=100
    local USAGE_PCT=$(awk "BEGIN{printf \"%.1f\", $DISK_GB/($QUOTA_TB*1000)*100}")
    local PROJECT_GB=$(du -s --block-size=1G "$QUANT_DIR" 2>/dev/null | cut -f1)

    # ── Job counts ────────────────────────────────────────────────────
    local N_DL=$(squeue -u "$USER" -n scgeo-dl -h -t RUNNING,PENDING 2>/dev/null | wc -l)
    local N_DL_RUN=$(squeue -u "$USER" -n scgeo-dl -h -t RUNNING 2>/dev/null | wc -l)
    local N_DL_PEND=$(squeue -u "$USER" -n scgeo-dl -h -t PENDING 2>/dev/null | wc -l)
    local N_QT=$(squeue -u "$USER" -n scgeo-qt -h -t RUNNING,PENDING 2>/dev/null | wc -l)
    local N_QT_RUN=$(squeue -u "$USER" -n scgeo-qt -h -t RUNNING 2>/dev/null | wc -l)
    local N_QT_PEND=$(squeue -u "$USER" -n scgeo-qt -h -t PENDING 2>/dev/null | wc -l)
    local N_ORCH=$(squeue -u "$USER" -n scgeo-orch -h -t RUNNING,PENDING 2>/dev/null | wc -l)

    # ── Download metrics ──────────────────────────────────────────────
    local TOTAL_MANIFESTS=$(find "$DL_DIR" -name "download_manifest.json" 2>/dev/null | wc -l)
    local SUCCESS_DL=$(find "$DL_DIR" -name "download_manifest.json" \
        -exec grep -l '"success": true' {} + 2>/dev/null | wc -l)
    local FAIL_DL=$((TOTAL_MANIFESTS - SUCCESS_DL))

    # ── Quant metrics ─────────────────────────────────────────────────
    local QT_DONE=$(find "$DL_DIR" -name ".quant_done" 2>/dev/null | wc -l)
    local QT_LOCKED=$(find "$DL_DIR" -name ".quant_locked" 2>/dev/null | wc -l)
    local READY=0
    while IFS= read -r mpath; do
        local dir=$(dirname "$mpath")
        [[ -f "$dir/.quant_done" ]] && continue
        [[ -f "$dir/.quant_locked" ]] && continue
        grep -q '"success": true' "$mpath" 2>/dev/null && READY=$((READY + 1))
    done < <(find "$DL_DIR" -name "download_manifest.json" 2>/dev/null)

    # ── .1pz metrics ──────────────────────────────────────────────────
    local N_1PZ=$(find "$QUANT_DIR" -name "counts.1pz" 2>/dev/null | wc -l)
    local TOTAL_1PZ_BYTES=$(find "$QUANT_DIR" -name "counts.1pz" -exec du -cb {} + 2>/dev/null | tail -1 | cut -f1)
    TOTAL_1PZ_BYTES=${TOTAL_1PZ_BYTES:-0}
    local TOTAL_1PZ_MB=$(awk "BEGIN{printf \"%.1f\", $TOTAL_1PZ_BYTES/1048576}")

    # ── Cell counts + status + protocol from result CSVs ─────────────
    # Use Python for reliable parsing — CSV fields contain Python dicts with commas
    local CSV_STATS=$(python3 -c "
import csv, glob, sys
files = glob.glob('$RESULTS_DIR/results_quant_*.csv')
rows = []
for f in files:
    with open(f) as fh:
        rows.extend(list(csv.DictReader(fh)))
total_cells = 0; max_cells = 0; sum_genes = 0; sum_counts = 0; sum_mr = 0; n_with_cells = 0
statuses = {}; qc_statuses = {}; protocols = {}; fail_stages = {}
for r in rows:
    st = r.get('status','?'); statuses[st] = statuses.get(st,0)+1
    qc = r.get('qc_status','')
    if qc: qc_statuses[qc] = qc_statuses.get(qc,0)+1
    pr = r.get('protocol','')
    if pr: protocols[pr] = protocols.get(pr,0)+1
    fs = r.get('fail_stage','')
    if fs: fail_stages[fs] = fail_stages.get(fs,0)+1
    try:
        nc = int(r.get('n_cells','0') or '0')
        if nc > 0:
            n_with_cells += 1; total_cells += nc
            if nc > max_cells: max_cells = nc
            sum_genes += int(r.get('median_genes','0') or '0')
            sum_counts += int(r.get('median_counts','0') or '0')
            mr = float(r.get('mapping_rate','0') or '0')
            sum_mr += mr
    except: pass
avg_cells = total_cells/n_with_cells if n_with_cells else 0
avg_genes = sum_genes/n_with_cells if n_with_cells else 0
avg_counts = sum_counts/n_with_cells if n_with_cells else 0
avg_mr = sum_mr/n_with_cells*100 if n_with_cells else 0
print(f'N_SAMPLES={n_with_cells}')
print(f'TOTAL_CELLS={total_cells}')
print(f'MAX_CELLS={max_cells}')
print(f'AVG_CELLS={avg_cells:.0f}')
print(f'AVG_GENES={avg_genes:.0f}')
print(f'AVG_COUNTS={avg_counts:.0f}')
print(f'AVG_MR={avg_mr:.1f}')
print(f'TOTAL_RESULT_ROWS={len(rows)}')
for k,v in sorted(statuses.items(), key=lambda x:-x[1]):
    print(f'STATUS:{k}={v}')
for k,v in sorted(qc_statuses.items(), key=lambda x:-x[1]):
    print(f'QC:{k}={v}')
for k,v in sorted(protocols.items(), key=lambda x:-x[1]):
    print(f'PROTO:{k}={v}')
for k,v in sorted(fail_stages.items(), key=lambda x:-x[1]):
    print(f'FAIL:{k}={v}')
" 2>/dev/null || echo "N_SAMPLES=0")

    local N_SAMPLES=$(echo "$CSV_STATS" | grep '^N_SAMPLES=' | cut -d= -f2-)
    local TOTAL_CELLS=$(echo "$CSV_STATS" | grep '^TOTAL_CELLS=' | cut -d= -f2-)
    local MAX_CELLS=$(echo "$CSV_STATS" | grep '^MAX_CELLS=' | cut -d= -f2-)
    local AVG_CELLS=$(echo "$CSV_STATS" | grep '^AVG_CELLS=' | cut -d= -f2-)
    local AVG_GENES=$(echo "$CSV_STATS" | grep '^AVG_GENES=' | cut -d= -f2-)
    local AVG_COUNTS=$(echo "$CSV_STATS" | grep '^AVG_COUNTS=' | cut -d= -f2-)
    local AVG_MR=$(echo "$CSV_STATS" | grep '^AVG_MR=' | cut -d= -f2-)
    local TOTAL_RESULT_ROWS=$(echo "$CSV_STATS" | grep '^TOTAL_RESULT_ROWS=' | cut -d= -f2-)
    N_SAMPLES=${N_SAMPLES:-0}; TOTAL_CELLS=${TOTAL_CELLS:-0}; MAX_CELLS=${MAX_CELLS:-0}
    AVG_CELLS=${AVG_CELLS:-0}; AVG_GENES=${AVG_GENES:-0}; AVG_COUNTS=${AVG_COUNTS:-0}
    AVG_MR=${AVG_MR:-0}; TOTAL_RESULT_ROWS=${TOTAL_RESULT_ROWS:-0}

    local STATUS_BREAKDOWN=$(echo "$CSV_STATS" | grep '^STATUS:' | sed 's/^STATUS://' | \
        awk -F= '{printf "    %-20s %s\n", $1, $2}')
    local QC_BREAKDOWN=$(echo "$CSV_STATS" | grep '^QC:' | sed 's/^QC://' | \
        awk -F= '{printf "    %-20s %s\n", $1, $2}')
    local PROTO_BREAKDOWN=$(echo "$CSV_STATS" | grep '^PROTO:' | sed 's/^PROTO://' | \
        awk -F= '{printf "    %-20s %s\n", $1, $2}')
    local FAIL_BREAKDOWN=$(echo "$CSV_STATS" | grep '^FAIL:' | sed 's/^FAIL://' | \
        awk -F= '{printf "    %-20s %s\n", $1, $2}')

    # ── Claims ────────────────────────────────────────────────────────
    local TOTAL_CLAIMS=0 TOTAL_GSMS_CLAIMED=0
    if [[ -f "$CLAIMS_DIR/ledger.tsv" ]]; then
        TOTAL_CLAIMS=$(wc -l < "$CLAIMS_DIR/ledger.tsv")
        TOTAL_GSMS_CLAIMED=$(cut -f3 "$CLAIMS_DIR/ledger.tsv" | tr ',' '\n' | grep -c . 2>/dev/null || echo 0)
    fi

    # ── Failed SLURM jobs (today) ─────────────────────────────────────
    local FAILED_SUMMARY=$(sacct -u "$USER" --starttime="$(date +%Y-%m-%d)" \
        --format=JobName%12,State%14 --noheader 2>/dev/null | \
        grep -v "\.\(batch\|extern\)" | grep -vE "RUNNING|PENDING|COMPLETED" | \
        sort | uniq -c | sort -rn | head -10)
    local COMPLETED_TODAY=$(sacct -u "$USER" --starttime="$(date +%Y-%m-%d)" \
        --format=JobName%12,State%14 --noheader 2>/dev/null | \
        grep -v "\.\(batch\|extern\)" | grep "COMPLETED" | wc -l)

    # ── SU estimate ───────────────────────────────────────────────────
    # Anvil shared: billing = max(n_cores, ceil(mem_GB / 1.97))
    # DL: 1 CPU, 6GB → billing=ceil(6/1.97)=4 → 4 SU/hr each
    # QT: 4 CPU, 16GB → billing=max(4,ceil(16/1.97))=9 → 9 SU/hr each
    # Orch: 1 CPU, 2GB → billing=max(1,ceil(2/1.97))=2 → 2 SU/hr
    local SU_DL_HR=$((N_DL_RUN * 4))
    local SU_QT_HR=$((N_QT_RUN * 9))
    local SU_ORCH_HR=$((N_ORCH * 2))
    local SU_TOTAL_HR=$((SU_DL_HR + SU_QT_HR + SU_ORCH_HR))

    # ── Throughput estimate ───────────────────────────────────────────
    local ELIGIBLE=23749  # phase 4a mouse droplet
    local COMPLETION_PCT=$(awk "BEGIN{printf \"%.2f\", $N_1PZ/$ELIGIBLE*100}")
    local AVG_1PZ_KB=0
    if [[ $N_1PZ -gt 0 ]]; then
        AVG_1PZ_KB=$(awk "BEGIN{printf \"%.0f\", $TOTAL_1PZ_BYTES/$N_1PZ/1024}")
    fi

    # ── Build report ──────────────────────────────────────────────────
    cat <<EOF
╔══════════════════════════════════════════════════════════════════╗
║          scGEO REPROCESSING PIPELINE — STATUS REPORT            ║
║          $TIMESTAMP                          ║
╚══════════════════════════════════════════════════════════════════╝

━━━ PROGRESS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Eligible samples (phase 4a):  $ELIGIBLE
  GSMs claimed in ledger:       $TOTAL_GSMS_CLAIMED ($TOTAL_CLAIMS batches)
  Successful downloads:         $SUCCESS_DL
  Failed downloads:             $FAIL_DL
  Quant completed:              $QT_DONE
  .1pz files produced:          $N_1PZ
  COMPLETION:                   ${COMPLETION_PCT}% ($N_1PZ / $ELIGIBLE)

━━━ CELLS & QUALITY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Samples with metrics:         $N_SAMPLES
  Total cells across all .1pz:  $(printf "%'d" "$TOTAL_CELLS")
  Avg cells per sample:         $(printf "%.0f" "$AVG_CELLS")
  Max cells in one sample:      $(printf "%'d" "$MAX_CELLS")
  Avg median genes/cell:        $(printf "%.0f" "$AVG_GENES")
  Avg median UMIs/cell:         $(printf "%.0f" "$AVG_COUNTS")
  Avg mapping rate:             ${AVG_MR}%

━━━ PIPELINE STATUS BREAKDOWN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total result rows:    $TOTAL_RESULT_ROWS
$STATUS_BREAKDOWN

  QC status:
$QC_BREAKDOWN

  Protocol mix:
$PROTO_BREAKDOWN

  Failure stages:
$FAIL_BREAKDOWN

━━━ STORAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Scratch (downloads):  ${DISK_PCT}TB / ${QUOTA_TB}TB (${USAGE_PCT}%)
  Project (.1pz):       ${PROJECT_GB}GB (${TOTAL_1PZ_MB}MB in .1pz)
  Avg .1pz size:        ${AVG_1PZ_KB}KB

━━━ CLUSTER UTILIZATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Download jobs:    $N_DL ($N_DL_RUN running, $N_DL_PEND pending)
  Quant jobs:       $N_QT ($N_QT_RUN running, $N_QT_PEND pending)
  Orchestrator:     $N_ORCH
  Ready for quant:  $READY (awaiting pickup)
  In-flight quant:  $QT_LOCKED (locked)

  SU burn rate:     ~${SU_TOTAL_HR} SU/hr
    Downloads:      ${SU_DL_HR} SU/hr ($N_DL_RUN jobs × 4 SU)
    Quant:          ${SU_QT_HR} SU/hr ($N_QT_RUN jobs × 9 SU)
    Orchestrator:   ${SU_ORCH_HR} SU/hr (coordinator overhead)

━━━ JOB HEALTH (today) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Completed jobs today: $COMPLETED_TODAY
  Failed/cancelled:
$FAILED_SUMMARY

━━━ EFFICIENCY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Download success rate:  $(if [[ $TOTAL_MANIFESTS -gt 0 ]]; then awk "BEGIN{printf \"%.1f\", $SUCCESS_DL/$TOTAL_MANIFESTS*100}"; else echo "N/A"; fi)%
  Quant yield (1pz/dl):   $(if [[ $SUCCESS_DL -gt 0 ]]; then awk "BEGIN{printf \"%.1f\", $N_1PZ/$SUCCESS_DL*100}"; else echo "N/A"; fi)%
  Pipeline idle samples:  $READY (downloaded but not yet quantified)
  Quant backlog ratio:    $(if [[ $N_QT_RUN -gt 0 ]]; then awk "BEGIN{printf \"%.1f\", $READY/$N_QT_RUN}"; else echo "N/A"; fi) samples/job

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Report generated by scgeo-orch job ${SLURM_JOB_ID:-standalone}
  Anvil HPC — Purdue University — bio260157 allocation
EOF
}

send_report() {
    local REPORT_FILE="$PROJECT/scgeo/pipeline/latest_report.txt"
    generate_report "$@" > "$REPORT_FILE"
    echo "  ✉ Report written to $REPORT_FILE"
}

# If run directly (not sourced), send email or print to stdout
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ "${1:-}" == "--stdout" ]]; then
        generate_report
    else
        generate_report
        send_report
    fi
fi
