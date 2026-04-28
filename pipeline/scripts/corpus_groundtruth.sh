#!/usr/bin/env bash
# corpus_groundtruth.sh — Download corpus samples + run ground truth pipelines
#
# Usage: bash scripts/corpus_groundtruth.sh [download|index|starsolo|salmon|singlify|all]
#
# Directories:
#   CORPUS  = /mnt/projects/debruinz_project/singlify_validation/corpus/
#   STARSOLO = /mnt/projects/debruinz_project/singlify_validation/starsolo/
#   SALMON  = /mnt/projects/debruinz_project/singlify_validation/salmon/
#   SINGLIFY = /mnt/projects/debruinz_project/singlify_validation/singlify_out/
#   PROFILES = /mnt/projects/debruinz_project/singlify_validation/profiles/
set -euo pipefail

# ── Paths ──
VALDIR=/mnt/projects/debruinz_project/singlify_validation
CORPUS=$VALDIR/corpus
STARSOLO_DIR=$VALDIR/starsolo
SALMON_DIR=$VALDIR/salmon
SINGLIFY_DIR=$VALDIR/singlify_out
PROFILES=$VALDIR/profiles

REFDIR=/mnt/projects/debruinz_project/cellarium/reference
GRCH38=$REFDIR/GRCh38-2024-A
GRCH38_STAR=$GRCH38/star_2.7.11b
GRCH38_FA=$GRCH38/fasta/genome.fa
GRCH38_GTF=$GRCH38/genes/genes.gtf.gz

GRCM39=$REFDIR/GRCm39-2024-A
GRCM39_STAR=$GRCM39/star_2.7.11b

SINGLIFY_BIN=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
OFQ_BIN=/mnt/home/debruinz/Singlet-AI/singlify/build/1fq
STAR_STOCK=/mnt/home/debruinz/Singlet-AI/STAR/source/STAR_stock_baseline

WL_10X_V3=/mnt/home/debruinz/Singlet-AI/singlify/whitelists/3M-february-2018.txt
WL_10X_V2=/mnt/home/debruinz/Singlet-AI/singlify/whitelists/737K-august-2016.txt
WL_ARC=/mnt/home/debruinz/Singlet-AI/singlify/whitelists/gex_737K-arc-v1.txt

CONDA_BIN=/mnt/home/debruinz/.conda/envs/cellarium/bin
STAR_CONDA=$CONDA_BIN/STAR
SALMON_BIN=$CONDA_BIN/salmon
SIMPLEAF_BIN=$CONDA_BIN/simpleaf
ALEVIN_FRY_BIN=$CONDA_BIN/alevin-fry
SAMTOOLS=$CONDA_BIN/samtools
FASTERQ=$CONDA_BIN/fasterq-dump

THREADS=16

# ── Corpus definition ──
# Format: SRR|SPECIES|PROTOCOL|WHITELIST|NOTES
SAMPLES=(
  "SRR32855204|human|10x-arc-gex|$WL_ARC|40M reads, primary benchmark"
  "SRR13352022|human|10x-v3|$WL_10X_V3|PBMC, well-characterized"
  "SRR6470091|human|10x-v2|$WL_10X_V2|PBMC, 10x public dataset"
  "SRR17959728|human|10x-v4|$WL_10X_V3|GEX, post-2022 chemistry"
  "SRR10211374|human|10x-5p|$WL_10X_V3|5-prime gene expression"
  "SRR5702400|human|Drop-seq|NONE|Macosko et al."
  "SRR5953799|human|inDrop|NONE|Klein lab inDrop v3"
  "SRR13218441|mouse|sci-RNA-seq3|NONE|Cusanovich/Shendure"
  "ERR9993498|human|BD-Rhapsody|NONE|BD Genomics WTA"
  "SRR9895822|mouse|Smart-seq2|NONE|Full-length, no UMI"
  "SRR11546745|mouse|10x-v3|$WL_10X_V3|GRCm39 test"
)

parse_sample() {
  local IFS='|'
  read -r SRR SPECIES PROTOCOL WHITELIST NOTES <<< "$1"
}

genome_for() {
  # Returns STAR genome dir for a species
  if [[ "$1" == "human" ]]; then
    echo "$GRCH38_STAR"
  elif [[ "$1" == "mouse" ]]; then
    echo "$GRCM39_STAR"
  fi
}

gtf_for() {
  if [[ "$1" == "human" ]]; then
    echo "$GRCH38_GTF"
  elif [[ "$1" == "mouse" ]]; then
    echo "$GRCM39/genes/genes.gtf.gz"
  fi
}

fasta_for() {
  if [[ "$1" == "human" ]]; then
    echo "$GRCH38_FA"
  elif [[ "$1" == "mouse" ]]; then
    echo "$GRCM39/fasta/genome.fa"
  fi
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Phase 1: Download GRCm39 reference + build STAR index ──
build_grcm39_index() {
  if [[ -f "$GRCM39_STAR/SA" ]]; then
    log "GRCm39 STAR index already exists at $GRCM39_STAR"
    return 0
  fi

  log "Downloading GRCm39 reference from Ensembl..."
  mkdir -p "$GRCM39/fasta" "$GRCM39/genes"

  if [[ ! -f "$GRCM39/fasta/genome.fa" ]]; then
    cd "$GRCM39/fasta"
    # Use Ensembl release 113 (latest as of 2024)
    curl -L -o genome.fa.gz \
      "https://ftp.ensembl.org/pub/release-113/fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.primary_assembly.fa.gz"
    gunzip genome.fa.gz
    $SAMTOOLS faidx genome.fa
    log "GRCm39 FASTA downloaded and indexed"
  fi

  if [[ ! -f "$GRCM39/genes/genes.gtf.gz" ]]; then
    cd "$GRCM39/genes"
    curl -L -o genes.gtf.gz \
      "https://ftp.ensembl.org/pub/release-113/gtf/mus_musculus/Mus_musculus.GRCm39.113.gtf.gz"
    log "GRCm39 GTF downloaded"
  fi

  log "Building GRCm39 STAR index (this takes ~20 minutes)..."
  mkdir -p "$GRCM39_STAR"
  export TMPDIR=/dev/shm
  $STAR_CONDA \
    --runMode genomeGenerate \
    --genomeDir "$GRCM39_STAR" \
    --genomeFastaFiles "$GRCM39/fasta/genome.fa" \
    --sjdbGTFfile "$GRCM39/genes/genes.gtf.gz" \
    --runThreadN "$THREADS" \
    --outTmpDir /dev/shm/star_grcm39_tmp \
    2>&1 | tee "$PROFILES/grcm39_index_build.log"
  log "GRCm39 STAR index built at $GRCM39_STAR"
}

# ── Phase 2: Download all corpus samples ──
download_corpus() {
  for entry in "${SAMPLES[@]}"; do
    parse_sample "$entry"
    local onefq="$CORPUS/${SRR}.1fq"
    local fastq_dir="$CORPUS/${SRR}_fastq"

    if [[ -f "$onefq" ]]; then
      log "SKIP $SRR — already downloaded as .1fq"
      continue
    fi

    log "Downloading $SRR ($PROTOCOL, $SPECIES)..."
    mkdir -p "$fastq_dir"

    # Download .1fq via singlify (archives the sample)
    export TMPDIR=/dev/shm
    /usr/bin/time -v "$SINGLIFY_BIN" download "$SRR" -o "$onefq" \
      2>&1 | tee "$PROFILES/${SRR}_download.log" &

    # Also download raw FASTQ for ground-truth pipelines that need FASTQ input
    (
      export TMPDIR=/dev/shm
      /usr/bin/time -v $FASTERQ --threads 4 --split-files --outdir "$fastq_dir" "$SRR" \
        2>&1 | tee "$PROFILES/${SRR}_fasterq.log"
    ) &

    # Rate limit: max 4 concurrent downloads to avoid NCBI throttling
    local njobs=$(jobs -r | wc -l)
    if (( njobs >= 4 )); then
      wait -n
    fi
  done
  wait
  log "All downloads complete"
}

# ── Phase 3: Run STARsolo ground truth on all samples ──
run_starsolo() {
  for entry in "${SAMPLES[@]}"; do
    parse_sample "$entry"
    local outdir="$STARSOLO_DIR/$SRR"
    local fastq_dir="$CORPUS/${SRR}_fastq"
    local genome
    genome=$(genome_for "$SPECIES")

    if [[ -d "$outdir/Solo.out" ]]; then
      log "SKIP STARsolo $SRR — already completed"
      continue
    fi

    if [[ ! -d "$genome" ]] || [[ ! -f "$genome/SA" ]]; then
      log "SKIP STARsolo $SRR — genome index not ready ($genome)"
      continue
    fi

    # Need FASTQ files
    local r1="" r2=""
    if [[ "$PROTOCOL" == "Smart-seq2" ]]; then
      # Smart-seq2: single-end or paired-end, no STARsolo
      log "SKIP STARsolo $SRR — Smart-seq2 uses standard STAR alignment, not STARsolo"
      continue
    fi

    # Find FASTQ files
    r1=$(ls "$fastq_dir"/*_1.fastq* 2>/dev/null | head -1)
    r2=$(ls "$fastq_dir"/*_2.fastq* 2>/dev/null | head -1)

    if [[ -z "$r1" ]] || [[ -z "$r2" ]]; then
      log "SKIP STARsolo $SRR — FASTQ files not found in $fastq_dir"
      continue
    fi

    mkdir -p "$outdir"

    # Determine STARsolo parameters based on protocol
    local solo_type="CB_UMI_Simple"
    local cb_start=1 cb_len=16 umi_start=17 umi_len=12
    local wl_file="$WHITELIST"
    local read_cmd=""
    local extra_args=""

    # Check if files are gzipped
    if [[ "$r1" == *.gz ]]; then
      read_cmd="--readFilesCommand zcat"
    fi

    case "$PROTOCOL" in
      10x-v2)
        cb_len=16; umi_start=17; umi_len=10
        ;;
      10x-v3|10x-v4)
        cb_len=16; umi_start=17; umi_len=12
        ;;
      10x-5p)
        cb_len=16; umi_start=17; umi_len=12
        # 5' data: R1=cDNA, R2=barcode — swap
        local tmp="$r1"; r1="$r2"; r2="$tmp"
        ;;
      10x-arc-gex)
        cb_len=16; umi_start=17; umi_len=12
        wl_file="$WL_ARC"
        ;;
      Drop-seq)
        cb_len=12; umi_start=13; umi_len=8
        solo_type="CB_UMI_Simple"
        wl_file="None"
        extra_args="--soloCBmatchWLtype 1MM"
        ;;
      inDrop)
        solo_type="CB_UMI_Complex"
        # inDrop needs special handling — skip for now
        log "SKIP STARsolo $SRR — inDrop requires CB_UMI_Complex, manual config needed"
        continue
        ;;
      sci-RNA-seq3)
        # sci-RNA-seq3: combinatorial barcodes
        log "SKIP STARsolo $SRR — sci-RNA-seq3 requires custom barcode handling"
        continue
        ;;
      BD-Rhapsody)
        # BD Rhapsody: complex barcode structure
        log "SKIP STARsolo $SRR — BD Rhapsody requires custom barcode handling"
        continue
        ;;
    esac

    log "Running STARsolo on $SRR ($PROTOCOL, $SPECIES, $(genome_for $SPECIES))..."
    export TMPDIR=/dev/shm

    /usr/bin/time -v $STAR_STOCK \
      --runThreadN "$THREADS" \
      --genomeDir "$genome" \
      --readFilesIn "$r2" "$r1" \
      $read_cmd \
      --soloType "$solo_type" \
      --soloCBwhitelist "$wl_file" \
      --soloCBstart "$cb_start" --soloCBlen "$cb_len" \
      --soloUMIstart "$umi_start" --soloUMIlen "$umi_len" \
      --outSAMtype BAM SortedByCoordinate \
      --outSAMattributes NH HI nM AS CR UR CB UB GX GN sS sQ sM \
      --soloFeatures Gene GeneFull SJ \
      --outFileNamePrefix "${outdir}/" \
      $extra_args \
      2>&1 | tee "$PROFILES/${SRR}_starsolo.log"

    log "STARsolo $SRR complete"
  done
}

# ── Phase 4: Run salmon/alevin-fry ground truth ──
build_salmon_index() {
  local species=$1
  local ref_dir
  if [[ "$species" == "human" ]]; then
    ref_dir=$GRCH38
  else
    ref_dir=$GRCM39
  fi
  local sal_idx="$ref_dir/salmon_idx"

  if [[ -d "$sal_idx" ]]; then
    log "Salmon index for $species already exists"
    return 0
  fi

  log "Building salmon index for $species via simpleaf..."
  mkdir -p "$sal_idx"

  local fa=$(fasta_for "$species")
  local gtf=$(gtf_for "$species")

  # Build splici reference (spliced+intronic) for alevin-fry
  export TMPDIR=/dev/shm
  export ALEVIN_FRY_HOME=$ref_dir/af_home
  mkdir -p "$ALEVIN_FRY_HOME"

  /usr/bin/time -v $SIMPLEAF_BIN index \
    --output "$sal_idx" \
    --fasta "$fa" \
    --gtf "$gtf" \
    --threads "$THREADS" \
    2>&1 | tee "$PROFILES/${species}_salmon_index.log"

  log "Salmon index for $species built"
}

run_salmon() {
  for entry in "${SAMPLES[@]}"; do
    parse_sample "$entry"
    local outdir="$SALMON_DIR/$SRR"
    local fastq_dir="$CORPUS/${SRR}_fastq"

    if [[ -d "$outdir" ]] && [[ -f "$outdir/alevin/quants_mat.gz" || -f "$outdir/af_quant/alevin/quants_mat.mtx" ]]; then
      log "SKIP salmon $SRR — already completed"
      continue
    fi

    # Salmon/simpleaf only works well for UMI-based protocols
    case "$PROTOCOL" in
      Smart-seq2|sci-RNA-seq3|BD-Rhapsody|inDrop)
        log "SKIP salmon $SRR — $PROTOCOL not suitable for simpleaf"
        continue
        ;;
    esac

    local r1=$(ls "$fastq_dir"/*_1.fastq* 2>/dev/null | head -1)
    local r2=$(ls "$fastq_dir"/*_2.fastq* 2>/dev/null | head -1)

    if [[ -z "$r1" ]] || [[ -z "$r2" ]]; then
      log "SKIP salmon $SRR — FASTQ files not found"
      continue
    fi

    local ref_dir
    if [[ "$SPECIES" == "human" ]]; then
      ref_dir=$GRCH38
    else
      ref_dir=$GRCM39
    fi
    local sal_idx="$ref_dir/salmon_idx"

    if [[ ! -d "$sal_idx" ]]; then
      log "SKIP salmon $SRR — salmon index not built for $SPECIES"
      continue
    fi

    # Determine simpleaf chemistry string
    local chem=""
    case "$PROTOCOL" in
      10x-v2) chem="10xv2" ;;
      10x-v3|10x-v4|10x-arc-gex) chem="10xv3" ;;
      10x-5p) chem="10xv3" ;; # 5' uses same barcode layout
      Drop-seq) chem="dropseq" ;;
    esac

    if [[ -z "$chem" ]]; then
      log "SKIP salmon $SRR — unknown chemistry mapping for $PROTOCOL"
      continue
    fi

    mkdir -p "$outdir"
    log "Running simpleaf quant on $SRR ($PROTOCOL, $SPECIES)..."
    export TMPDIR=/dev/shm
    export ALEVIN_FRY_HOME=$ref_dir/af_home

    /usr/bin/time -v $SIMPLEAF_BIN quant \
      --reads1 "$r1" \
      --reads2 "$r2" \
      --index "$sal_idx/index" \
      --chemistry "$chem" \
      --resolution cr-like \
      --threads "$THREADS" \
      --output "$outdir" \
      --expected-ori fw \
      2>&1 | tee "$PROFILES/${SRR}_salmon.log"

    log "Salmon/alevin-fry $SRR complete"
  done
}

# ── Phase 5: Run singlify pipeline ──
run_singlify() {
  for entry in "${SAMPLES[@]}"; do
    parse_sample "$entry"
    local outdir="$SINGLIFY_DIR/$SRR"
    local onefq="$CORPUS/${SRR}.1fq"

    if [[ -d "$outdir" ]] && ls "$outdir"/*.1pz 2>/dev/null | head -1 > /dev/null; then
      log "SKIP singlify $SRR — already completed"
      continue
    fi

    if [[ ! -f "$onefq" ]]; then
      log "SKIP singlify $SRR — .1fq not found"
      continue
    fi

    local genome
    genome=$(genome_for "$SPECIES")
    if [[ ! -d "$genome" ]] || [[ ! -f "$genome/SA" ]]; then
      log "SKIP singlify $SRR — genome index not ready"
      continue
    fi

    local gtf
    gtf=$(gtf_for "$SPECIES")
    local snps="/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.vcf.gz"

    mkdir -p "$outdir"
    log "Running singlify on $SRR ($PROTOCOL, $SPECIES)..."
    export TMPDIR=/dev/shm

    local singlify_args=(
      "$onefq"
      --genome-dir "$genome"
      --exons "$gtf"
      --out-prefix "$outdir/"
      --threads "$THREADS"
    )

    if [[ "$WHITELIST" != "NONE" ]] && [[ -f "$WHITELIST" ]]; then
      singlify_args+=(--whitelist "$WHITELIST")
    fi

    # Only use SNP VCF for human samples
    if [[ "$SPECIES" == "human" ]] && [[ -f "$snps" ]]; then
      singlify_args+=(--snps "$snps")
    fi

    /usr/bin/time -v "$SINGLIFY_BIN" "${singlify_args[@]}" \
      2>&1 | tee "$PROFILES/${SRR}_singlify.log"

    log "singlify $SRR complete"
  done
}

# ── Phase 6: Generate profiling summary ──
generate_profile_summary() {
  log "Generating profiling summary..."
  local summary="$PROFILES/SUMMARY.tsv"
  echo -e "SRR\tPipeline\tWallTime_s\tUserTime_s\tSysTime_s\tMaxRSS_KB\tCPU_pct\tExitCode" > "$summary"

  for logfile in "$PROFILES"/*.log; do
    local base=$(basename "$logfile" .log)
    local srr=$(echo "$base" | sed 's/_[^_]*$//')
    local pipeline=$(echo "$base" | sed 's/.*_//')

    # Parse /usr/bin/time -v output
    local wall=$(grep "Elapsed (wall clock)" "$logfile" 2>/dev/null | awk '{print $NF}')
    local user=$(grep "User time" "$logfile" 2>/dev/null | awk '{print $NF}')
    local sys=$(grep "System time" "$logfile" 2>/dev/null | awk '{print $NF}')
    local maxrss=$(grep "Maximum resident" "$logfile" 2>/dev/null | awk '{print $NF}')
    local cpu=$(grep "Percent of CPU" "$logfile" 2>/dev/null | awk '{print $NF}')
    local exit_code=$(grep "Exit status" "$logfile" 2>/dev/null | awk '{print $NF}')

    echo -e "${srr}\t${pipeline}\t${wall:-NA}\t${user:-NA}\t${sys:-NA}\t${maxrss:-NA}\t${cpu:-NA}\t${exit_code:-NA}" >> "$summary"
  done

  log "Profile summary written to $summary"
  column -t "$summary"
}

# ── Main ──
case "${1:-all}" in
  download)
    download_corpus
    ;;
  index)
    build_grcm39_index
    ;;
  starsolo)
    run_starsolo
    ;;
  salmon)
    build_salmon_index human
    build_salmon_index mouse
    run_salmon
    ;;
  singlify)
    run_singlify
    ;;
  profile)
    generate_profile_summary
    ;;
  all)
    build_grcm39_index
    download_corpus
    run_starsolo
    build_salmon_index human
    build_salmon_index mouse
    run_salmon
    run_singlify
    generate_profile_summary
    ;;
  *)
    echo "Usage: $0 [download|index|starsolo|salmon|singlify|profile|all]"
    exit 1
    ;;
esac
