#!/usr/bin/env python
"""Smoke test for scgeo.metadata module imports."""
import sys
sys.path.insert(0, "/mnt/home/debruinz/Singlet-AI/geo-reprocess")

from scgeo.metadata.barcodes import normalize_barcode, match_barcodes
from scgeo.metadata.soft import parse_characteristics, load_soft_metadata, build_gsm_level_obs
from scgeo.metadata.api import MetadataResult, classify_supplementary_files, build_metadata
from scgeo.metadata.extract import extract_metadata_from_h5ad, extract_metadata_from_tabular, extract_metadata_from_loom
from scgeo.metadata.description import fetch_geo_description
from scgeo.metadata.download import download_supplementary_file

# Test 1: normalize_barcode
assert normalize_barcode("Vehicle1_AAACCCAAGCATGAAT-1") == "AAACCCAAGCATGAAT"
assert normalize_barcode("AAACCCAAGCATGAAT-1") == "AAACCCAAGCATGAAT"
assert normalize_barcode("AAACCCAAGCATGAAT") == "AAACCCAAGCATGAAT"
print("PASS: normalize_barcode")

# Test 2: parse_characteristics
chars = parse_characteristics("tissue: brain ;; cell_type: neuron ;; age: P30")
assert chars == {"tissue": "brain", "cell_type": "neuron", "age": "P30"}, f"Got: {chars}"
print("PASS: parse_characteristics")

# Test 3: MetadataResult dataclass
r = MetadataResult(gse_id="GSE123", gsm_id="GSM456")
assert r.status == "pending"
assert r.tier2_source is None
print(f"PASS: MetadataResult ({list(r.__dataclass_fields__.keys())})")

# Test 4: barcode matching
mapping = match_barcodes(
    ["ACGT" * 4 + "-1", "TGCA" * 4],
    ["ACGT" * 4, "TGCA" * 4],
)
assert len(mapping) >= 1
print(f"PASS: match_barcodes ({len(mapping)} matched)")

print("\nAll smoke tests passed!")
