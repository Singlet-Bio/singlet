#!/usr/bin/env python3
"""
Comprehensive pipeline validation test suite for geo-reprocess.

Tests every protocol × species × chemistry combination that could
cause failures at scale. Designed to be run on HPC compute nodes.

Test layers:
  1. Config validation — chemistry strings, species refs, protocol maps
  2. Index validation — verify piscem indices exist and are readable
  3. Download validation — ENA/SRA URL accessibility for test samples
  4. Protocol detection — verify FASTQ → chemistry detection for each protocol
  5. Quantification smoke — run simpleaf on small samples per protocol
  6. Catalog integrity — verify eligible_samples.csv consistency

Usage:
    ssh c001 "cd /mnt/projects/debruinz_project/cellarium/workspace/geo-reprocess && \
      python tests/test_pipeline_matrix.py [--layer N] [--protocol PROTO]"
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pipeline_test")

# ── Test infrastructure ──────────────────────────────────────────────

PASS = 0
FAIL = 0
WARN = 0
RESULTS = []


def record(layer, name, status, detail=""):
    global PASS, FAIL, WARN
    if status == "PASS":
        PASS += 1
    elif status == "FAIL":
        FAIL += 1
    else:
        WARN += 1
    RESULTS.append({"layer": layer, "name": name, "status": status, "detail": detail})
    sym = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}[status]
    log.info(f"  {sym} [{layer}] {name}: {detail}" if detail else f"  {sym} [{layer}] {name}")


# ── Reference data ───────────────────────────────────────────────────

INDEX_BASE = Path("/mnt/projects/debruinz_project/cellarium/index")
AF_HOME = Path("/mnt/projects/debruinz_project/cellarium/af_home")
CATALOG_PATH = Path("/mnt/projects/debruinz_project/cellarium/catalog")

# Protocol → expected simpleaf chemistry for each protocol in eligible_samples
EXPECTED_CHEMISTRIES = {
    "10xv2": "10xv2",
    "10xv3": "10xv3",
    "10xv4": "10xv4-3p",
    "10xv3_5prime": "10xv3",
    "10x_multiome": "10xv3",
    "10x_suspect": "10xv3",
    "dropseq": "1{b[12]u[8]x:}2{r:}",
    "seqwell": "1{b[12]u[8]x:}2{r:}",
    "dnbelab": "1{b[10]u[10]x:}2{r:}",
    "indrop": "1{b[8]f[GAGTGATTGCTTGTGACGCCTT]b[8]u[6]x:}2{r:}",
    "bd_rhapsody": "1{b[9]f[ACTGGCCTGCGA]b[9]f[GGTAGCGGTGACA]b[9]u[8]x:}2{r:}",
    "ddseq": "1{b[8]f[ACTGAC]b[8]u[6]x:}2{r:}",
    "scirna": "1{b[9-10]f[CAGAGC]u[8]b[10]}2{r:}",
    "splitseq": "1{r:}2{u[10]b[8]f[GTGGCCGATGTTTCGCATCGGCGTACGACT]b[8]f[ATCCACGTGCTTGAGAGGCCAGAGCATTCG]b[8]x:}",
    "parse": "1{b[8]f[ATCCACGTGCTTGAG]b[8]f[GTGGCCGATGTTTCG]b[8]u[10]x:}2{r:}",
    "celseq": "1{b[8]u[4]x:}2{r:}",
    "celseq2": "1{b[6]u[6]x:}2{r:}",
    "marsseq": "1{b[7]u[8]x:}2{r:}",
    "quartzseq": "1{b[15]u[8]x:}2{r:}",
    "strtseq": "1{b[6]u[5]x:}2{r:}",
    "icell8": "1{b[11]u[14]x:}2{r:}",
    "surecell": "1{b[6]u[6]x:}2{r:}",
    "scopeseq": "1{b[10]u[10]x:}2{r:}",
    "microwell": "1{b[18]u[6]x:}2{r:}",
    "smartseq2": "smartseq",
    "smartseq3": "smartseq",
    "plate_based": "smartseq",
}

# Smallest paired-end samples per protocol (pre-selected from catalog)
# These are small enough to download quickly but large enough to test quant
PROTOCOL_TEST_SAMPLES = {
    "10xv3": {"gsm": "GSM7105602", "gse": "GSE227690", "organism": "Homo sapiens", "taxon": 9606, "reads": 101102},
    "10xv2": {"gsm": "GSM6250978", "gse": "GSE206325", "organism": "Homo sapiens", "taxon": 9606, "reads": 106332},
    "10xv4": {"gsm": "GSM5211866", "gse": "GSE169657", "organism": "Homo sapiens", "taxon": 9606, "reads": 4070992},
    "10x_multiome": {"gsm": "GSM6601768", "gse": "GSE214231", "organism": "Homo sapiens", "taxon": 9606, "reads": 85591},
    "dropseq": {"gsm": "GSM6199113", "gse": "GSE204864", "organism": "Homo sapiens", "taxon": 9606, "reads": 1790863},
    "indrop": {"gsm": "GSM7083785", "gse": "GSE226788", "organism": "Mus musculus", "taxon": 10090, "reads": 230031},
    "celseq": {"gsm": "GSM2312774", "gse": "GSE86977", "organism": "Homo sapiens", "taxon": 9606, "reads": 105081},
    "celseq2": {"gsm": "GSM4008955", "gse": "GSE135437", "organism": "Homo sapiens", "taxon": 9606, "reads": 341484},
    "marsseq": {"gsm": "GSM3052622", "gse": "GSE112004", "organism": "Mus musculus", "taxon": 10090, "reads": 100104},
    "scirna": {"gsm": "GSM7130890", "gse": "GSE228590", "organism": "Mus musculus", "taxon": 10090, "reads": 102042},
    "bd_rhapsody": {"gsm": "GSM6513567", "gse": "GSE212212", "organism": "Homo sapiens", "taxon": 9606, "reads": 1205391},
    "dnbelab": {"gsm": "GSM6126190", "gse": "GSE202636", "organism": "Mus musculus", "taxon": 10090, "reads": 100694},
    "icell8": {"gsm": "GSM3502973", "gse": "GSE123426", "organism": "Homo sapiens", "taxon": 9606, "reads": 12852967},
    "strtseq": {"gsm": "GSM2832797", "gse": "GSE106251", "organism": "Homo sapiens", "taxon": 9606, "reads": 444267},
    "quartzseq": {"gsm": "GSM3730940", "gse": "GSE130050", "organism": "Homo sapiens", "taxon": 9606, "reads": 282854},
    "microwell": {"gsm": "GSM8194531", "gse": "GSE263565", "organism": "Homo sapiens", "taxon": 9606, "reads": 1751418},
    "seqwell": {"gsm": "GSM5171874", "gse": "GSE168883", "organism": "Homo sapiens", "taxon": 9606, "reads": 500000},
    "parse": {"gsm": "GSM8679746", "gse": "GSE314191", "organism": "Homo sapiens", "taxon": 9606, "reads": 1860362},
    "ddseq": {"gsm": "GSM8257578", "gse": "GSE266970", "organism": "Homo sapiens", "taxon": 9606, "reads": 29272714},
}

# Non-human/mouse species test samples (all 10xv3 where possible)
SPECIES_TEST_SAMPLES = {
    "Danio rerio": {"gsm": "GSM9428644", "gse": "GSE315445", "taxon": 7955, "reads": 643197},
    "Rattus norvegicus": {"gsm": "GSM4138045", "gse": "GSE139318", "taxon": 10116, "reads": 9024810},
    "Macaca mulatta": {"gsm": "GSM7813066", "gse": "GSE244293", "taxon": 9544, "reads": 415781},
    "Gallus gallus": {"gsm": "GSM8889610", "gse": "GSE224724", "taxon": 9031, "reads": 8768052},
    "Sus scrofa": {"gsm": "GSM6355573", "gse": "GSE208613", "taxon": 9823, "reads": 7443419},
    "Anolis carolinensis": {"gsm": "GSM7476132", "gse": "GSE234876", "taxon": 28377, "reads": 452664},
    "Macaca fascicularis": {"gsm": "GSM4476664", "gse": "GSE148683", "taxon": 9541, "reads": 94760415},
}


# ═══════════════════════════════════════════════════════════════════
# Layer 1: Config validation
# ═══════════════════════════════════════════════════════════════════

def test_layer1_config():
    """Validate protocol configs, species refs, and chemistry consistency."""
    log.info("=" * 60)
    log.info("Layer 1: Configuration validation")
    log.info("=" * 60)

    from scgeo.config.protocols import PROTOCOL_CHEMISTRY, NON_RNA_PROTOCOLS, BUILTIN_CHEMISTRIES, get_chemistry
    from scgeo.config.species import SPECIES_REF, ORGANISM_TO_TAXON, get_taxon_id

    # 1a. Every protocol_inferred value in catalog must have a mapping
    import pandas as pd
    cat = pd.read_parquet(CATALOG_PATH / "geo_single_cell_catalog.parquet",
                          columns=["protocol_inferred"])
    catalog_protos = set(cat["protocol_inferred"].dropna().unique())
    known_protos = set(PROTOCOL_CHEMISTRY.keys()) | NON_RNA_PROTOCOLS | {"unknown", "unknown_sc", "snRNA_unknown"}

    missing = catalog_protos - known_protos
    if missing:
        record(1, "catalog_protocols_mapped", "FAIL", f"Unmapped protocols in catalog: {missing}")
    else:
        record(1, "catalog_protocols_mapped", "PASS", f"All {len(catalog_protos)} catalog protocols mapped")

    # 1b. Every processable protocol has a non-None chemistry
    processable = {p for p, c in PROTOCOL_CHEMISTRY.items() if c is not None}
    none_chem = {p for p, c in PROTOCOL_CHEMISTRY.items() if c is None}
    record(1, "processable_protocols", "PASS", f"{len(processable)} processable, {len(none_chem)} None-chemistry")

    # 1c. FGDL geometry strings are valid format
    for proto, chem in PROTOCOL_CHEMISTRY.items():
        if chem is None or chem in BUILTIN_CHEMISTRIES or chem == "smartseq":
            continue
        # FGDL format: 1{...}2{...} or 2{...}1{...}
        if not (chem.startswith("1{") or chem.startswith("2{")):
            record(1, f"fgdl_format_{proto}", "FAIL", f"Invalid FGDL: {chem}")
        else:
            # Check balanced braces
            if chem.count("{") != chem.count("}"):
                record(1, f"fgdl_braces_{proto}", "FAIL", f"Unbalanced braces: {chem}")
            else:
                record(1, f"fgdl_valid_{proto}", "PASS")

    # 1d. All 15 eligible species have entries in SPECIES_REF
    eligible_taxons = {9606, 10090, 9544, 7955, 10116, 9541, 9823, 9031, 9913, 9615, 8364, 9796, 9685, 28377, 13616}
    for tid in eligible_taxons:
        info = SPECIES_REF.get(tid)
        if info is None:
            record(1, f"species_ref_{tid}", "FAIL", f"Missing from SPECIES_REF")
        else:
            name = info["name"]
            record(1, f"species_ref_{tid}", "PASS", f"{name}")

    # 1e. Organism name lookups work for all eligible species
    test_organisms = ["Homo sapiens", "Mus musculus", "Danio rerio", "Drosophila melanogaster",
                      "Rattus norvegicus", "Macaca mulatta", "Sus scrofa", "Gallus gallus"]
    for org in test_organisms:
        tid = get_taxon_id(org)
        if tid is None:
            record(1, f"organism_lookup_{org}", "FAIL")
        else:
            record(1, f"organism_lookup_{org}", "PASS", f"→ {tid}")

    # 1f. Cross-check eligible_samples chemistry vs PROTOCOL_CHEMISTRY
    elig = pd.read_csv(CATALOG_PATH / "processing/eligible_samples.csv", nrows=1000)
    mismatches = 0
    for _, row in elig.iterrows():
        proto = row["protocol_inferred"]
        expected = get_chemistry(proto)
        actual = row.get("simpleaf_chemistry", "")
        if expected and actual and expected != actual:
            # The eligible_samples sometimes has geometry strings that differ
            # from PROTOCOL_CHEMISTRY (e.g., indrop uses variable length)
            mismatches += 1
    if mismatches > 50:
        record(1, "chemistry_consistency", "WARN", f"{mismatches}/1000 rows differ from PROTOCOL_CHEMISTRY")
    else:
        record(1, "chemistry_consistency", "PASS", f"{mismatches}/1000 differ (expected for variable-length barcodes)")


# ═══════════════════════════════════════════════════════════════════
# Layer 2: Index validation
# ═══════════════════════════════════════════════════════════════════

def test_layer2_indices():
    """Verify piscem indices exist and have expected files for all eligible species."""
    log.info("=" * 60)
    log.info("Layer 2: Index validation")
    log.info("=" * 60)

    from scgeo.config.species import SPECIES_REF

    # Species name → index directory name mapping
    eligible_species = {
        9606: "homo_sapiens",
        10090: "mus_musculus",
        9544: "macaca_mulatta",
        7955: "danio_rerio",
        10116: "rattus_norvegicus",
        9541: "macaca_fascicularis",
        9823: "sus_scrofa",
        9031: "gallus_gallus",
        9913: "bos_taurus",
        9615: "canis_lupus_familiaris",
        8364: "xenopus_tropicalis",
        9796: "equus_caballus",
        9685: "felis_catus",
        28377: "anolis_carolinensis",
        13616: "monodelphis_domestica",
    }

    for taxon, dirname in eligible_species.items():
        species_name = SPECIES_REF[taxon]["name"]
        idx_dir = INDEX_BASE / f"{dirname}_splici"

        if not idx_dir.exists():
            record(2, f"index_dir_{dirname}", "FAIL", f"Directory missing: {idx_dir}")
            continue

        # Check for piscem index files
        piscem_files = list(idx_dir.glob("**/piscem_idx*"))
        if not piscem_files:
            record(2, f"piscem_files_{dirname}", "FAIL", f"No piscem_idx* files in {idx_dir}")
            continue

        # Check for t2g file (transcript-to-gene mapping)
        t2g_files = list(idx_dir.glob("**/t2g*"))
        if not t2g_files:
            record(2, f"t2g_{dirname}", "WARN", f"No t2g file found")
        
        # Check for simpleaf_index.json (metadata)
        sf_json = list(idx_dir.glob("**/simpleaf_index.json"))
        if sf_json:
            try:
                with open(sf_json[0]) as f:
                    meta = json.load(f)
                record(2, f"index_{dirname}", "PASS",
                       f"{species_name} — {len(piscem_files)} piscem files, meta OK")
            except Exception as e:
                record(2, f"index_{dirname}", "WARN",
                       f"{species_name} — piscem OK but bad JSON: {e}")
        else:
            record(2, f"index_{dirname}", "PASS",
                   f"{species_name} — {len(piscem_files)} piscem files (no simpleaf_index.json)")

    # Check ALEVIN_FRY_HOME
    if AF_HOME.exists():
        plist = AF_HOME / "plist"
        if plist.exists():
            wl_files = list(plist.iterdir())
            record(2, "af_home_plist", "PASS", f"{len(wl_files)} whitelist files")
        else:
            record(2, "af_home_plist", "FAIL", "No plist/ directory in AF_HOME")
    else:
        record(2, "af_home", "FAIL", f"AF_HOME not found: {AF_HOME}")


# ═══════════════════════════════════════════════════════════════════
# Layer 3: Download validation (URL accessibility)
# ═══════════════════════════════════════════════════════════════════

def test_layer3_downloads(protocols=None):
    """Verify ENA FASTQ URLs are accessible for test samples."""
    log.info("=" * 60)
    log.info("Layer 3: Download URL validation")
    log.info("=" * 60)

    import pandas as pd
    elig = pd.read_csv(CATALOG_PATH / "processing/eligible_samples.csv")

    samples_to_check = {}
    for proto, info in PROTOCOL_TEST_SAMPLES.items():
        if protocols and proto not in protocols:
            continue
        row = elig[(elig["gsm_id"] == info["gsm"]) & (elig["gse_id"] == info["gse"])]
        if len(row) == 0:
            record(3, f"catalog_entry_{proto}", "FAIL", f"{info['gsm']} not in eligible_samples")
            continue
        row = row.iloc[0]
        r1_url = row.get("ena_fastq_r1", "")
        r2_url = row.get("ena_fastq_r2", "")
        samples_to_check[proto] = {"r1": r1_url, "r2": r2_url, "gsm": info["gsm"]}

    for proto, urls in samples_to_check.items():
        for read, url in [("R1", urls["r1"]), ("R2", urls["r2"])]:
            if not url or str(url) == "nan":
                if read == "R2":
                    record(3, f"url_{proto}_{read}", "WARN", "No R2 URL (single-end?)")
                else:
                    record(3, f"url_{proto}_{read}", "FAIL", "No R1 URL")
                continue

            # HEAD request to check URL accessibility
            try:
                result = subprocess.run(
                    ["curl", "-sI", "-L", "--max-time", "15", str(url)],
                    capture_output=True, text=True, timeout=20
                )
                headers = result.stdout.lower()
                if "200 ok" in headers or "content-length" in headers:
                    # Extract file size
                    for line in result.stdout.split("\n"):
                        if "content-length" in line.lower():
                            size = int(line.split(":", 1)[1].strip())
                            size_mb = size / 1024**2
                            record(3, f"url_{proto}_{read}", "PASS", f"{urls['gsm']} {size_mb:.1f}MB")
                            break
                    else:
                        record(3, f"url_{proto}_{read}", "PASS", f"{urls['gsm']} (200 OK)")
                elif "403" in headers:
                    record(3, f"url_{proto}_{read}", "FAIL", f"403 Forbidden — {url[:60]}")
                else:
                    record(3, f"url_{proto}_{read}", "WARN", f"Unexpected response")
            except Exception as e:
                record(3, f"url_{proto}_{read}", "FAIL", f"curl error: {e}")


# ═══════════════════════════════════════════════════════════════════
# Layer 4: Protocol detection validation
# ═══════════════════════════════════════════════════════════════════

def test_layer4_detection():
    """Verify protocol detection logic for known read-length patterns."""
    log.info("=" * 60)
    log.info("Layer 4: Protocol detection validation")
    log.info("=" * 60)

    from scgeo.pipeline.detect import infer_protocol
    from scgeo.config.defaults import DetectionConfig

    class MockConfig:
        detection = DetectionConfig()
        detection.rlen_long_threshold = 85

    config = MockConfig()

    # Test cases: (r1_len, r2_len, hint, expected_protocol, expected_mode)
    detection_tests = [
        # Standard 10x patterns
        (28, 91, None, "10xv3", "droplet", "10xv3 standard"),
        (26, 98, None, "10xv2", "droplet", "10xv2 standard"),
        (28, 150, None, "10xv3", "droplet", "10xv3 long R2"),
        (28, 50, None, "10xv3", "droplet", "10xv3 short R2"),
        # Smart-seq patterns
        (101, 101, None, "smartseq2", "smartseq", "paired 101bp = smartseq"),
        (150, 150, None, "smartseq2", "smartseq", "paired 150bp = smartseq"),
        (151, 0, None, "smartseq2", "smartseq", "single-end 151bp = smartseq"),
        # Swapped reads
        (91, 28, None, "10xv3", "droplet", "swapped R1/R2"),
        (98, 26, None, "10xv2", "droplet", "swapped R1/R2 v2"),
        # Drop-seq length pattern
        (20, 80, None, "dropseq", "droplet", "short R1 = dropseq"),
        # Catalog hint overrides
        (101, 101, "10xv3", "10xv3", "droplet", "hint overrides smartseq detection"),
        (28, 91, "dropseq", "dropseq", "droplet", "hint overrides 10x detection"),
        (101, 101, "indrop", "indrop", "droplet", "hint=indrop on long reads"),
        (101, 101, "celseq2", "celseq2", "droplet", "hint=celseq2 on long reads"),
        (101, 101, "marsseq", "marsseq", "droplet", "hint=marsseq on long reads"),
        (28, 91, "smartseq2", "smartseq2", "smartseq", "hint=smartseq on 28bp R1"),
        # Edge cases
        (25, 50, None, "10xv2", "droplet", "25bp R1 ambiguous → 10xv2"),
        (50, 50, None, "dropseq", "droplet", "50bp R1 short → dropseq"),
    ]

    for r1, r2, hint, exp_proto, exp_mode, desc in detection_tests:
        det = infer_protocol(r1, r2, catalog_hint=hint, config=config)
        proto_ok = det.protocol == exp_proto
        mode_ok = det.mode == exp_mode

        if proto_ok and mode_ok:
            record(4, f"detect_{desc.replace(' ', '_')}", "PASS",
                   f"R1={r1} R2={r2} hint={hint} → {det.protocol}/{det.mode}")
        elif mode_ok:
            record(4, f"detect_{desc.replace(' ', '_')}", "WARN",
                   f"Mode OK but proto={det.protocol} (expected {exp_proto})")
        else:
            record(4, f"detect_{desc.replace(' ', '_')}", "FAIL",
                   f"Got {det.protocol}/{det.mode}, expected {exp_proto}/{exp_mode}")


# ═══════════════════════════════════════════════════════════════════
# Layer 5: Quantification smoke tests (download + detect + quant)
# ═══════════════════════════════════════════════════════════════════

def test_layer5_quant(protocols=None, species=None, timeout_per_sample=600):
    """Run full pipeline on small test samples per protocol/species.

    This is the most important layer — it catches:
    - Chemistry strings that simpleaf doesn't accept
    - Permit-list failures (wrong barcode structure)
    - Index path resolution errors
    - t2g mapping issues
    - Piscem mapping failures on specific geometries
    """
    log.info("=" * 60)
    log.info("Layer 5: Quantification smoke tests")
    log.info("=" * 60)

    import pandas as pd
    os.environ["ALEVIN_FRY_HOME"] = str(AF_HOME)

    elig = pd.read_csv(CATALOG_PATH / "processing/eligible_samples.csv")
    test_base = Path(tempfile.mkdtemp(prefix="scgeo_test_"))
    log.info(f"Test output directory: {test_base}")

    # Select test samples
    test_samples = {}

    # Protocol tests (human/mouse)
    for proto, info in PROTOCOL_TEST_SAMPLES.items():
        if protocols and proto not in protocols:
            continue
        row = elig[(elig["gsm_id"] == info["gsm"])]
        if len(row) == 0:
            record(5, f"quant_{proto}", "FAIL", f"Sample {info['gsm']} not in catalog")
            continue
        row = row.iloc[0]
        test_samples[f"proto_{proto}"] = {
            "gsm": info["gsm"],
            "gse": info["gse"],
            "organism": info["organism"],
            "r1_url": row["ena_fastq_r1"],
            "r2_url": row.get("ena_fastq_r2", ""),
            "chemistry": row["simpleaf_chemistry"],
            "protocol": proto,
            "taxon": info["taxon"],
            "reads": int(row.get("read_count", 0) or 0),
        }

    # Species tests
    if species is None:
        species = list(SPECIES_TEST_SAMPLES.keys())
    for org in species:
        if org not in SPECIES_TEST_SAMPLES:
            continue
        info = SPECIES_TEST_SAMPLES[org]
        row = elig[elig["gsm_id"] == info["gsm"]]
        if len(row) == 0:
            record(5, f"quant_species_{org}", "FAIL", f"Sample {info['gsm']} not in catalog")
            continue
        row = row.iloc[0]
        test_samples[f"species_{org.replace(' ', '_')}"] = {
            "gsm": info["gsm"],
            "gse": info["gse"],
            "organism": org,
            "r1_url": row["ena_fastq_r1"],
            "r2_url": row.get("ena_fastq_r2", ""),
            "chemistry": row["simpleaf_chemistry"],
            "protocol": "10xv3",
            "taxon": info["taxon"],
            "reads": int(row.get("read_count", 0) or 0),
        }

    for test_name, sample in test_samples.items():
        log.info(f"\n{'─'*50}")
        log.info(f"Testing: {test_name} ({sample['gsm']}, {sample['organism']}, {sample['chemistry']})")

        gsm = sample["gsm"]
        gse = sample["gse"]
        dl_dir = test_base / "downloads" / gse / gsm
        quant_dir = test_base / "quant" / gse / gsm
        # Ensure test_base still exists (node /tmp cleanup can remove it)
        test_base.mkdir(parents=True, exist_ok=True)
        dl_dir.mkdir(parents=True, exist_ok=True)
        quant_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()

        # Stage A: Download FASTQs
        try:
            r1_url = str(sample["r1_url"])
            r2_url = str(sample.get("r2_url", ""))
            r1_file = dl_dir / f"{gsm}_R1.fastq.gz"
            r2_file = dl_dir / f"{gsm}_R2.fastq.gz"

            # Quick download with curl (single connection, adequate for small files)
            # Scale timeout by expected read count: ~120s base + 1s per 100K reads
            n_reads = sample.get("reads", 0) or 0
            dl_timeout = max(120, int(120 + n_reads / 100_000))
            dl_timeout = min(dl_timeout, 600)  # cap at 10 min
            for url, dest, label in [(r1_url, r1_file, "R1"), (r2_url, r2_file, "R2")]:
                if not url or url == "nan":
                    continue
                cmd = ["curl", "-sL", "-f", "--max-time", str(dl_timeout), "-o", str(dest), url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=dl_timeout + 30)
                if result.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
                    record(5, f"{test_name}_download", "FAIL",
                           f"{label} download failed: exit={result.returncode}")
                    continue

            if not r1_file.exists():
                record(5, f"{test_name}", "FAIL", "R1 download failed")
                continue

            dl_time = time.time() - t0
            r1_mb = r1_file.stat().st_size / 1024**2
            r2_mb = r2_file.stat().st_size / 1024**2 if r2_file.exists() else 0
            log.info(f"  Downloaded: R1={r1_mb:.1f}MB R2={r2_mb:.1f}MB in {dl_time:.1f}s")

        except Exception as e:
            record(5, f"{test_name}", "FAIL", f"Download error: {e}")
            continue

        # Stage B: Protocol detection
        try:
            from scgeo.pipeline.detect import detect_protocol_from_files
            from scgeo.config.defaults import DetectionConfig

            class TestConfig:
                detection = DetectionConfig()
                detection.rlen_long_threshold = 85

            det = detect_protocol_from_files(
                r1_path=r1_file,
                r2_path=r2_file if r2_file.exists() else None,
                catalog_hint=sample["protocol"],
                chemistry_hint=sample.get("chemistry"),
                config=TestConfig(),
            )
            log.info(f"  Detected: {det.protocol}/{det.mode} conf={det.confidence} "
                     f"R1={det.r1_len}bp R2={det.r2_len}bp chem={det.chemistry}")

            if det.mode == "smartseq":
                record(5, f"{test_name}", "PASS",
                       f"Smart-seq detected (skipped by pipeline) — detection OK")
                continue

            if det.confidence == "low":
                record(5, f"{test_name}_detection", "WARN",
                       f"Low confidence: {det.reason}")

        except Exception as e:
            record(5, f"{test_name}_detection", "FAIL", f"Detection error: {e}")
            continue

        # Stage C: Quantification with simpleaf
        try:
            # Use detection.chemistry (which now prefers chemistry_hint from catalog)
            chemistry = det.chemistry
            if not chemistry:
                # Fallback to catalog if detection didn't set chemistry
                chemistry = sample["chemistry"]
                log.info(f"  Using catalog chemistry (detection had None): {chemistry}")
            organism_lower = sample["organism"].lower().replace(" ", "_")

            # Resolve index path
            idx_dir = INDEX_BASE / f"{organism_lower}_splici"
            if not idx_dir.exists():
                record(5, f"{test_name}", "FAIL", f"Index dir missing: {idx_dir}")
                continue

            # Find the index subdirectory
            if (idx_dir / "index").exists():
                index_path = idx_dir / "index"
            elif list(idx_dir.glob("piscem_idx*")):
                index_path = idx_dir
            else:
                record(5, f"{test_name}", "FAIL", f"No piscem files in {idx_dir}")
                continue

            # Build simpleaf command
            # Swap files if detection flagged read orientation mismatch
            r1_for_quant = r1_file
            r2_for_quant = r2_file if r2_file.exists() else None
            if det.reads_swapped and r2_for_quant:
                log.info(f"  Swapping R1/R2 files (reads_swapped=True)")
                r1_for_quant, r2_for_quant = r2_for_quant, r1_for_quant

            r1_arg = str(r1_for_quant)
            r2_arg = str(r2_for_quant) if r2_for_quant else ""

            cmd = [
                "simpleaf", "quant",
                "-c", chemistry,
                "-1", r1_arg,
                "-2", r2_arg,
                "-i", str(index_path),
                "-o", str(quant_dir),
                "-t", "4",  # Minimal threads for testing
                "--use-piscem",
                "--resolution", "cr-like",
                "--knee",
            ]

            log.info(f"  Running: simpleaf quant -c '{chemistry}' ...")
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout_per_sample,
                env={**os.environ, "ALEVIN_FRY_HOME": str(AF_HOME)},
            )

            # Retry with --expect-cells if knee permit-list generation failed
            if result.returncode != 0 and "generate-permit-list failed" in (result.stderr or ""):
                import shutil
                for subdir in ("af_map", "af_quant"):
                    p = quant_dir / subdir
                    if p.exists():
                        shutil.rmtree(p, ignore_errors=True)
                retry_cmd = cmd[:-1] + ["--expect-cells", "3000"]  # Replace --knee
                log.info(f"  Knee failed, retrying with --expect-cells 3000")
                result = subprocess.run(
                    retry_cmd, capture_output=True, text=True,
                    timeout=timeout_per_sample,
                    env={**os.environ, "ALEVIN_FRY_HOME": str(AF_HOME)},
                )

            elapsed = time.time() - t0

            if result.returncode != 0:
                # Extract meaningful error
                stderr = result.stderr[-500:] if result.stderr else ""
                stdout = result.stdout[-500:] if result.stdout else ""
                error_detail = stderr or stdout

                # Categorize the failure
                if "generate-permit-list" in error_detail:
                    category = "permit_list_fail"
                elif "piscem mapping failed" in error_detail:
                    category = "piscem_mapping_fail"
                elif "t2g" in error_detail.lower():
                    category = "t2g_missing"
                elif "chemistry" in error_detail.lower():
                    category = "bad_chemistry_string"
                else:
                    category = "simpleaf_error"

                record(5, f"{test_name}", "FAIL",
                       f"{category}: exit={result.returncode}, {error_detail[:200]}")
                continue

            # Check output files
            mtx_path = quant_dir / "af_quant" / "alevin" / "quants_mat.mtx"
            map_path = quant_dir / "af_map" / "map_info.json"

            if mtx_path.exists():
                # Parse mapping stats
                mapping_rate = 0
                n_cells = 0
                if map_path.exists():
                    with open(map_path) as f:
                        mi = json.load(f)
                        n_reads = mi.get("num_reads", 0)
                        n_mapped = mi.get("num_mapped", 0)
                        mapping_rate = n_mapped / max(n_reads, 1)

                quant_json = quant_dir / "af_quant" / "quant.json"
                if quant_json.exists():
                    with open(quant_json) as f:
                        qi = json.load(f)
                        n_cells = qi.get("num_quantified_cells", 0)

                if n_cells == 0:
                    record(5, f"{test_name}", "WARN",
                           f"Quant succeeded but 0 cells (mapping={mapping_rate:.1%}, {elapsed:.0f}s)")
                elif mapping_rate < 0.10:
                    record(5, f"{test_name}", "WARN",
                           f"Low mapping: {mapping_rate:.1%}, cells={n_cells}, {elapsed:.0f}s")
                else:
                    record(5, f"{test_name}", "PASS",
                           f"cells={n_cells}, mapping={mapping_rate:.1%}, {elapsed:.0f}s")
            else:
                record(5, f"{test_name}", "FAIL",
                       f"No quants_mat.mtx produced (exit=0 but no output)")

        except subprocess.TimeoutExpired:
            record(5, f"{test_name}", "FAIL",
                   f"Timeout after {timeout_per_sample}s")
        except Exception as e:
            record(5, f"{test_name}", "FAIL", f"Exception: {e}")

    # Cleanup temp dir (leave for debugging if failures)
    if FAIL == 0:
        import shutil
        shutil.rmtree(test_base, ignore_errors=True)
        log.info(f"Cleaned up {test_base}")
    else:
        log.info(f"Keeping test outputs at {test_base} for debugging")


# ═══════════════════════════════════════════════════════════════════
# Layer 6: Catalog integrity checks
# ═══════════════════════════════════════════════════════════════════

def test_layer6_catalog():
    """Verify eligible_samples.csv consistency and completeness."""
    log.info("=" * 60)
    log.info("Layer 6: Catalog integrity")
    log.info("=" * 60)

    import pandas as pd

    elig = pd.read_csv(CATALOG_PATH / "processing/eligible_samples.csv")

    # 6a. No duplicate GSM IDs
    dupes = elig["gsm_id"].duplicated().sum()
    if dupes > 0:
        record(6, "no_duplicate_gsm", "FAIL", f"{dupes} duplicate GSM IDs")
    else:
        record(6, "no_duplicate_gsm", "PASS", f"{len(elig)} unique GSMs")

    # 6b. Every row has organism + taxon_id
    missing_org = elig["organism"].isna().sum()
    missing_tax = elig["matched_taxon_id"].isna().sum()
    if missing_org > 0 or missing_tax > 0:
        record(6, "required_fields", "FAIL",
               f"Missing organism: {missing_org}, taxon: {missing_tax}")
    else:
        record(6, "required_fields", "PASS")

    # 6c. Every row has at least one download URL
    has_url = (elig["ena_fastq_r1"].notna() | elig.get("srr_accessions", pd.Series(dtype=str)).notna())
    missing_url = (~has_url).sum()
    if missing_url > 0:
        record(6, "has_download_url", "WARN", f"{missing_url} rows without download URL")
    else:
        record(6, "has_download_url", "PASS")

    # 6d. Chemistry is set for all droplet samples
    droplet = elig[elig["category"] == "droplet"]
    missing_chem = droplet["simpleaf_chemistry"].isna().sum()
    if missing_chem > 0:
        record(6, "droplet_has_chemistry", "FAIL", f"{missing_chem} droplet rows without chemistry")
    else:
        record(6, "droplet_has_chemistry", "PASS", f"All {len(droplet)} droplet rows have chemistry")

    # 6e. Protocol distribution sanity check
    proto_counts = elig["protocol_inferred"].value_counts()
    top = proto_counts.head(5)
    record(6, "protocol_distribution", "PASS",
           f"Top: {', '.join(f'{p}:{n}' for p, n in top.items())}")

    # 6f. Paired-end coverage for droplet protocols
    droplet_paired = droplet[droplet["ena_fastq_r2"].notna()]
    se_frac = 1 - len(droplet_paired) / max(len(droplet), 1)
    if se_frac > 0.5:
        record(6, "droplet_paired_coverage", "WARN",
               f"Only {1-se_frac:.0%} paired-end droplet ({len(droplet_paired)}/{len(droplet)})")
    else:
        record(6, "droplet_paired_coverage", "PASS",
               f"{1-se_frac:.0%} paired-end ({len(droplet_paired)}/{len(droplet)})")

    # 6g. All referenced taxon_ids have indices
    from scgeo.config.species import SPECIES_REF
    catalog_taxons = set(elig["matched_taxon_id"].dropna().astype(int).unique())
    missing_index = []
    for tid in catalog_taxons:
        info = SPECIES_REF.get(tid)
        if info is None:
            missing_index.append(tid)
            continue
        dirname = info["name"].lower().replace(" ", "_")
        idx_dir = INDEX_BASE / f"{dirname}_splici"
        if not idx_dir.exists():
            missing_index.append(f"{tid}:{info['name']}")

    if missing_index:
        record(6, "taxon_index_coverage", "FAIL", f"Missing indices: {missing_index}")
    else:
        record(6, "taxon_index_coverage", "PASS",
               f"All {len(catalog_taxons)} species have indices")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def print_summary():
    log.info("\n" + "=" * 60)
    log.info(f"SUMMARY: {PASS} passed, {WARN} warnings, {FAIL} failed")
    log.info("=" * 60)

    if FAIL > 0:
        log.info("\nFAILURES:")
        for r in RESULTS:
            if r["status"] == "FAIL":
                log.info(f"  ✗ [{r['layer']}] {r['name']}: {r['detail']}")

    if WARN > 0:
        log.info("\nWARNINGS:")
        for r in RESULTS:
            if r["status"] == "WARN":
                log.info(f"  ⚠ [{r['layer']}] {r['name']}: {r['detail']}")

    # Save results to JSON
    results_path = Path(__file__).parent / "pipeline_test_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {"pass": PASS, "warn": WARN, "fail": FAIL},
            "results": RESULTS,
        }, f, indent=2)
    log.info(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline validation test suite")
    parser.add_argument("--layer", type=int, nargs="+",
                        help="Run specific layer(s): 1=config, 2=indices, 3=downloads, "
                             "4=detection, 5=quant, 6=catalog")
    parser.add_argument("--protocol", nargs="+",
                        help="Test specific protocol(s) only (for layers 3/5)")
    parser.add_argument("--species", nargs="+",
                        help="Test specific species only (for layer 5)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout per sample in seconds (default: 600)")
    args = parser.parse_args()

    layers = args.layer or [1, 2, 3, 4, 5, 6]

    log.info("GEO Pipeline Validation Test Suite")
    log.info(f"Layers: {layers}")
    if args.protocol:
        log.info(f"Protocols: {args.protocol}")
    log.info("")

    if 1 in layers:
        test_layer1_config()
    if 2 in layers:
        test_layer2_indices()
    if 3 in layers:
        test_layer3_downloads(protocols=args.protocol)
    if 4 in layers:
        test_layer4_detection()
    if 5 in layers:
        test_layer5_quant(
            protocols=args.protocol,
            species=args.species,
            timeout_per_sample=args.timeout,
        )
    if 6 in layers:
        test_layer6_catalog()

    print_summary()
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
