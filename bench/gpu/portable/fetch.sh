#!/usr/bin/env bash
# Everything one timing run of run.sh needs, from public sources, under $DATA in the layout Clipper's harness reads
# (bench/gpu/pilot/tp_gc.sbatch: --ref-base REF, REF/GRCh38-2024-A/{star_2.7.11b,genes/genes.gtf}, the sample's .1fq).
#
#   fetch.sh            both phases (prepare steps whose tools are missing are reported and skipped)
#   fetch.sh download   downloads only: curl, tar and gzip; safe to run on the host while the image builds
#   fetch.sh prepare    STAR index, .1fq encoding, host Bloom filter, checksums: needs singlet and STAR (run it inside the image)
#
# Layout under DATA (default /data):
#   ref/GRCh38-2024-A/{fasta/genome.fa, genes/genes.gtf, star_2.7.11b/}   10x refdata-gex-GRCh38-2024-A; STAR 2.7.11b index as on Clipper
#   ref/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz   default SNP panel of --pipeline
#   ref/nonhost/{viral,bacterial,fungal}.snhskidx, host_21mer.bloom      non-host screen (see NONHOST below)
#   pbmc1k_v3/pbmc1k_v3.1fq                                               10x "1k PBMCs v3", both lanes, encoded with singlet encode
#
# Clipper's STAR index and non-host indices are published as tars with a manifest (sources, licenses, build commands, checksums):
#   https://data.singlet.bio/reference/v1/manifest.json
# fetch.sh downloads both tars, checks their sha256 and unpacks them. Nothing here needs credentials. Optional overrides:
#   PUB_REF_BASE       another prefix serving the same tars (default https://data.singlet.bio/reference/v1)
#   STAR_INDEX_TAR_URL another tar of a ready star_2.7.11b/ directory (its checksum is then not checked)
#   BUILD_STAR=1       build the STAR index with STAR instead of downloading it (about an hour; byte-identical to Clipper's only
#                      with the same STAR build)
#   NONHOST_BASE_URL   a URL prefix serving viral.snhskidx, bacterial.snhskidx, fungal.snhskidx (and optionally host_21mer.bloom)
#                      instead of the published tar
#   ONEFQ_URL          a ready pbmc1k_v3.1fq (skips the FASTQ download and the encoding)
#   JOBS               threads for STAR and the encoder (default: all cores)
set -euo pipefail
PHASE=${1:-all}
DATA=${DATA:-/data}; REF=${REF:-$DATA/ref}; G=$REF/GRCh38-2024-A; S1=$DATA/pbmc1k_v3
SCRATCH=${SCRATCH:-$DATA/scratch}; JOBS=${JOBS:-$(nproc)}
SINGLET_BIN=${SINGLET_BIN:-$(command -v singlet || true)}
SINGLET_SRC=${SINGLET_SRC:-/opt/singlet/src}
WL=${SINGLET_WHITELIST_DIR:-${SINGLET_BIN:+$(dirname "$SINGLET_BIN")/../whitelists}}
REPORT=$DATA/fetch-report.txt
mkdir -p "$G/fasta" "$G/genes" "$REF/pileup_indices" "$REF/nonhost" "$S1" "$SCRATCH"

# Public sources, each as the repository records it
REFDATA_URL=https://cf.10xgenomics.com/supp/cell-exp/refdata-gex-GRCh38-2024-A.tar.gz          # state/reference-manifest-v1.yaml
FASTQ_URL=https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_fastqs.tar     # bench/pipeline/datasets.tsv
SNP_URL=https://sourceforge.net/projects/cellsnp/files/SNPlist/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz/download   # bench/modalities/genotype_pool/fetch_convert.sbatch
PUB_REF_BASE=${PUB_REF_BASE:-https://data.singlet.bio/reference/v1}
STAR_TAR_URL=${STAR_INDEX_TAR_URL:-$PUB_REF_BASE/star_2.7.11b-GRCh38-2024-A.tar}
STAR_TAR_SHA256=80b5aed9b8e48d4d270204bb35874175a713567084a816e69081a946ab2fc661; [ -z "${STAR_INDEX_TAR_URL:-}" ] || STAR_TAR_SHA256=""   # another tar: no published checksum
NONHOST_TAR_URL=$PUB_REF_BASE/nonhost-refseq223.tar; NONHOST_TAR_SHA256=5c5dc7a2dd0b220aa5dd9a853a86ff3fe1950b83cd4283fd2c12e27bda9397dd
PANEL=$REF/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz
# Clipper's pbmc1k_v3.1fq (pipeline-benchmark/data/pbmc1k_v3): 66,601,887 reads, 134 blocks, 1,838,043,689 bytes, no base qualities
CLIPPER_1FQ_READS=66601887; CLIPPER_1FQ_BYTES=1838043689

log() { echo "[fetch $(date +%H:%M:%S)] $*" | tee -a "$REPORT"; }
get() {   # get URL FILE: resumable download through a .part file
  local url=$1 out=$2
  [ -s "$out" ] && return 0
  log "download $url"
  curl -fL --retry 5 --retry-delay 10 -C - -o "$out.part" "$url"
  mv "$out.part" "$out"
}
get_tar() {   # get_tar URL SHA256 DIR: download, check the sha256 (when given), unpack into DIR, delete the tar
  local url=$1 want=$2 dir=$3 f; f="$SCRATCH/$(basename "$url")"
  get "$url" "$f"
  if [ -n "$want" ]; then
    log "sha256 $(basename "$f")"
    local got; got=$(sha256sum "$f" | cut -d' ' -f1)
    [ "$got" = "$want" ] || { log "CHECKSUM MISMATCH $(basename "$f"): $got, published $want"; rm -f "$f"; exit 1; }
  fi
  tar -xf "$f" -C "$dir" && rm -f "$f"
  log "unpacked $(basename "$f") into $dir"
}
have_tools() { [ -n "$SINGLET_BIN" ] && [ -x "$SINGLET_BIN" ]; }
star_cmd() { if command -v STAR >/dev/null; then echo STAR; else echo "$SINGLET_BIN star"; fi; }

download() {
  # 1. 10x GRCh38-2024-A: only the FASTA and the GTF (the tarball's own STAR index is not the one Clipper uses)
  if [ ! -s "$G/fasta/genome.fa" ] || [ ! -s "$G/genes/genes.gtf" ]; then
    get "$REFDATA_URL" "$SCRATCH/refdata-gex-GRCh38-2024-A.tar.gz"
    log "extract genome.fa and genes.gtf"
    tar -xzf "$SCRATCH/refdata-gex-GRCh38-2024-A.tar.gz" -C "$SCRATCH" --wildcards '*/fasta/genome.fa' '*/genes/genes.gtf*'
    mv "$SCRATCH"/refdata-gex-GRCh38-2024-A/fasta/genome.fa "$G/fasta/genome.fa"
    if [ -f "$SCRATCH/refdata-gex-GRCh38-2024-A/genes/genes.gtf.gz" ]; then
      gzip -dc "$SCRATCH/refdata-gex-GRCh38-2024-A/genes/genes.gtf.gz" > "$G/genes/genes.gtf.part" && mv "$G/genes/genes.gtf.part" "$G/genes/genes.gtf"
    else mv "$SCRATCH/refdata-gex-GRCh38-2024-A/genes/genes.gtf" "$G/genes/genes.gtf"; fi
    rm -rf "$SCRATCH/refdata-gex-GRCh38-2024-A" "$SCRATCH/refdata-gex-GRCh38-2024-A.tar.gz"
  fi

  # 2. pbmc1k_v3 FASTQs (two lanes), unless a ready .1fq is given or already present
  if [ ! -s "$S1/pbmc1k_v3.1fq" ]; then
    if [ -n "${ONEFQ_URL:-}" ]; then get "$ONEFQ_URL" "$S1/pbmc1k_v3.1fq"
    elif [ ! -f "$S1/fastqs/.extracted" ]; then
      get "$FASTQ_URL" "$SCRATCH/pbmc_1k_v3_fastqs.tar"
      mkdir -p "$S1/fastqs"; tar -xf "$SCRATCH/pbmc_1k_v3_fastqs.tar" -C "$S1/fastqs"
      rm -f "$SCRATCH/pbmc_1k_v3_fastqs.tar"; touch "$S1/fastqs/.extracted"
    fi
  fi

  # 3. Default SNP panel. ASSUMPTION: Clipper's file is the cellsnp-lite list with "chr" added to the contig names and sorted; the exact
  #    command that made it is not in the repository. Only the order of the panel's columns in the outputs depends on it.
  if [ ! -s "$PANEL" ]; then
    get "$SNP_URL" "$SCRATCH/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz"
    log "SNP panel: chr prefix and sort"
    { gzip -dc "$SCRATCH/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz" | grep '^#' || true
      gzip -dc "$SCRATCH/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz" | grep -v '^#' \
        | awk 'BEGIN{FS=OFS="\t"} { if ($1 !~ /^chr/) $1 = "chr" $1; print }' | LC_ALL=C sort -t$'\t' -k1,1V -k2,2n -S 2G; } \
      | gzip -c > "$PANEL.part"
    mv "$PANEL.part" "$PANEL"; rm -f "$SCRATCH/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz"
  fi

  # 4. Non-host screening indices (NCBI RefSeq 223 sketches, built with singlet build-nonhost-db) and the host Bloom filter:
  #    the published tar, or the files under NONHOST_BASE_URL. Without them run.sh turns the screen off and labels its results as
  #    not comparable to Clipper's default runs.
  if [ -n "${NONHOST_BASE_URL:-}" ]; then
    for f in viral.snhskidx bacterial.snhskidx fungal.snhskidx host_21mer.bloom; do
      [ -s "$REF/nonhost/$f" ] || get "${NONHOST_BASE_URL%/}/$f" "$REF/nonhost/$f" || log "no $f at $NONHOST_BASE_URL"
    done
  elif [ ! -s "$REF/nonhost/viral.snhskidx" ] || [ ! -s "$REF/nonhost/bacterial.snhskidx" ] || [ ! -s "$REF/nonhost/fungal.snhskidx" ]        || [ ! -s "$REF/nonhost/host_21mer.bloom" ]; then
    get_tar "$NONHOST_TAR_URL" "$NONHOST_TAR_SHA256" "$REF"
  fi
  if [ ! -s "$REF/nonhost/viral.snhskidx" ] || [ ! -s "$REF/nonhost/bacterial.snhskidx" ] || [ ! -s "$REF/nonhost/fungal.snhskidx" ]; then
    log "non-host indices missing in $REF/nonhost"
  fi

  # 5a. Clipper's STAR 2.7.11b index (the published tar), unless BUILD_STAR=1
  if [ ! -s "$G/star_2.7.11b/SA" ] && [ "${BUILD_STAR:-0}" != 1 ]; then get_tar "$STAR_TAR_URL" "$STAR_TAR_SHA256" "$G"; fi
}

prepare() {
  if ! have_tools; then log "prepare: no singlet binary (SINGLET_BIN); run 'fetch.sh prepare' inside the image"; return 0; fi
  local star; star=$(star_cmd)

  # 5b. STAR 2.7.11b index (BUILD_STAR=1, or the download step was skipped) with Clipper's parameters (genomeParameters.txt of $REF/GRCh38-2024-A/star_2.7.11b on Clipper:
  #    --genomeSAsparseD 2 --sjdbGTFfile genes.gtf, defaults otherwise: sjdbOverhang 100, genomeSAindexNbases 14)
  if [ ! -s "$G/star_2.7.11b/SA" ]; then
    if [ "${BUILD_STAR:-0}" != 1 ]; then get_tar "$STAR_TAR_URL" "$STAR_TAR_SHA256" "$G"
    else
      log "STAR genomeGenerate ($star, $JOBS threads)"
      rm -rf "$G/star_2.7.11b.tmp" "$SCRATCH/star_gg"; mkdir -p "$G/star_2.7.11b.tmp"
      local t0; t0=$(date +%s)
      $star --runMode genomeGenerate --runThreadN "$JOBS" --genomeDir "$G/star_2.7.11b.tmp" --genomeFastaFiles "$G/fasta/genome.fa" \
        --genomeSAsparseD 2 --sjdbGTFfile "$G/genes/genes.gtf" --limitGenomeGenerateRAM "${STAR_RAM:-80000000000}" \
        --outTmpDir "$SCRATCH/star_gg" --outFileNamePrefix "$SCRATCH/star_gg_" > "$SCRATCH/star_gg.stdout" 2>&1 \
        || { tail -20 "$SCRATCH/star_gg_Log.out" 2>/dev/null; exit 1; }
      mv "$G/star_2.7.11b.tmp" "$G/star_2.7.11b"; rm -rf "$SCRATCH/star_gg"
      log "STAR index built in $(( $(date +%s) - t0 )) s"
    fi
  fi

  # 6. pbmc1k_v3.1fq: lane 1 then lane 2 (bench/engine_gpu_gate/s8_encode.sbatch), no base qualities as in Clipper's file
  #    (docs/gpu-engine/exploration.md 12.10, docs/research-beta/ingest-format.md 6)
  if [ ! -s "$S1/pbmc1k_v3.1fq" ]; then
    local F=$S1/fastqs/pbmc_1k_v3_fastqs/pbmc_1k_v3_S1_L00
    [ -s "${F}1_R1_001.fastq.gz" ] || { log "no FASTQs under $S1/fastqs: run 'fetch.sh download' first"; exit 1; }
    rm -f "$SCRATCH/R1.fastq.gz" "$SCRATCH/R2.fastq.gz"
    for l in 1 2; do cat "${F}${l}_R1_001.fastq.gz" >> "$SCRATCH/R1.fastq.gz"; cat "${F}${l}_R2_001.fastq.gz" >> "$SCRATCH/R2.fastq.gz"; done
    log "singlet encode (--quality none)"
    local t0; t0=$(date +%s)
    "$SINGLET_BIN" encode --reads "$SCRATCH/R1.fastq.gz" "$SCRATCH/R2.fastq.gz" -o "$SCRATCH/pbmc1k_v3.1fq" --quality none \
      ${WL:+--whitelist-dir "$WL"} > "$S1/encode.log" 2>&1 || { tail -20 "$S1/encode.log"; exit 1; }
    mv "$SCRATCH/pbmc1k_v3.1fq" "$S1/pbmc1k_v3.1fq"; rm -f "$SCRATCH/R1.fastq.gz" "$SCRATCH/R2.fastq.gz"
    log "encoded in $(( $(date +%s) - t0 )) s"
  fi
  local reads bytes
  reads=$("$SINGLET_BIN" decode "$S1/pbmc1k_v3.1fq" --header 2>&1 | sed -n 's/.*Reads: *\([0-9]*\).*/\1/p' | head -1)
  bytes=$(stat -c %s "$S1/pbmc1k_v3.1fq")
  log "pbmc1k_v3.1fq: ${reads:-?} reads (Clipper $CLIPPER_1FQ_READS), $bytes bytes (Clipper $CLIPPER_1FQ_BYTES)"
  [ "${reads:-0}" = "$CLIPPER_1FQ_READS" ] || log "WARNING read count differs from Clipper's file"
  [ "$bytes" = "$CLIPPER_1FQ_BYTES" ] || log "NOTE size differs from Clipper's file (another encoder version or settings): same reads, outputs not byte-comparable"

  # 7. Host Bloom filter for the non-host screen, when the screen's indices are present and no filter came with them
  #    (state/reference-manifest-v1.yaml: singlet build-host-filter, k 21, w 11)
  if [ -s "$REF/nonhost/viral.snhskidx" ] && [ ! -s "$REF/nonhost/host_21mer.bloom" ]; then
    log "singlet build-host-filter (k 21, w 11)"
    "$SINGLET_BIN" build-host-filter --genome-fasta "$G/fasta/genome.fa" --output "$REF/nonhost/host_21mer.bloom.part" \
      --kmer 21 --window 11 --threads "$JOBS" > "$SCRATCH/bloom.log" 2>&1 || { tail -20 "$SCRATCH/bloom.log"; exit 1; }
    mv "$REF/nonhost/host_21mer.bloom.part" "$REF/nonhost/host_21mer.bloom"
  fi

  # 8. Checksums against state/reference-manifest-v1.yaml (Clipper's files). A mismatch is reported, not fatal: timing does not depend
  #    on it, but outputs will not be byte-identical to Clipper's.
  local M=$SINGLET_SRC/state/reference-manifest-v1.yaml
  if [ -f "$M" ]; then
    log "checksums against $M"
    awk '/^  - path:/{p=$3} /^    sha256:/{if ($2 ~ /^[0-9a-f]{64}$/) print p, $2}' "$M" | while read -r p want; do
      if [ ! -f "$REF/$p" ]; then log "  absent    $p"; continue; fi
      case $p in */genomeParameters.txt) log "  skipped   $p (records the build's paths)"; continue ;; esac
      got=$(sha256sum "$REF/$p" | cut -d' ' -f1)
      [ "$got" = "$want" ] && log "  match     $p" || log "  DIFFERS   $p ($got)"
    done
  else log "no manifest at $M: checksums skipped"; fi
}

case $PHASE in
  download) download ;;
  prepare) prepare ;;
  all) download; prepare ;;
  *) echo "usage: fetch.sh [download|prepare]"; exit 2 ;;
esac
log "disk: $(du -sh "$DATA" 2>/dev/null | cut -f1) under $DATA"
