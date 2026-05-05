"""Test Stage 2c extractors on real supplementary files."""
import sys, os, tempfile, urllib.request, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

from scgeo.metadata.extract_2c import (
    extract_metadata_from_xlsx,
    extract_metadata_from_h5seurat,
    extract_metadata_from_hdf5,
)
from scgeo.metadata.barcodes import align_author_metadata

TMPDIR = Path("/tmp/2c_test")
TMPDIR.mkdir(exist_ok=True)

def download(url, name):
    dest = TMPDIR / name
    if not dest.exists():
        https_url = url.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/")
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(https_url, str(dest))
    return dest

# ── Test 1: XLSX with cell barcodes ──
print("=" * 60)
print("TEST 1: XLSX - Cell Barcodes")
print("=" * 60)
try:
    dest = download(
        "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE109nnn/GSE109447/suppl/GSE109447_13055_Cell_Barcodes.xlsx",
        "GSE109447_13055_Cell_Barcodes.xlsx"
    )
    df = extract_metadata_from_xlsx(dest)
    print(f"  Result: {df.shape}")
    if not df.empty:
        print(f"  Columns: {list(df.columns[:10])}")
        print(f"  Sample barcodes: {df['barcode'].head(3).tolist()}")
        print(f"  Has barcode col: {'barcode' in df.columns}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Test 2: XLSX with metadata annotations ──
print("\n" + "=" * 60)
print("TEST 2: XLSX - Sample Annotations")
print("=" * 60)
try:
    dest = download(
        "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE155nnn/GSE155109/suppl/GSE155109_barcode_sample_annotations.xlsx",
        "GSE155109_barcode_sample_annotations.xlsx"
    )
    df = extract_metadata_from_xlsx(dest)
    print(f"  Result: {df.shape}")
    if not df.empty:
        print(f"  Columns: {list(df.columns[:15])}")
        print(f"  Sample barcodes: {df['barcode'].head(3).tolist()}")
        # Try alignment against a processed GSM from this GSE
        print(f"  First row: {dict(list(df.iloc[0].items())[:6])}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Test 3: H5Seurat ──
print("\n" + "=" * 60)
print("TEST 3: H5Seurat - cluster3")
print("=" * 60)
try:
    dest = download(
        "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE178nnn/GSE178088/suppl/GSE178088_cluster3.h5Seurat",
        "GSE178088_cluster3.h5Seurat"
    )
    df = extract_metadata_from_h5seurat(dest)
    print(f"  Result: {df.shape}")
    if not df.empty:
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample barcodes: {df['barcode'].head(3).tolist()}")
        print(f"  active_ident values: {df['active_ident'].value_counts().head(5).to_dict() if 'active_ident' in df.columns else 'N/A'}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Test 4: H5Seurat - CD4Atlas (richer metadata) ──
print("\n" + "=" * 60)
print("TEST 4: H5Seurat - CD4Atlas")
print("=" * 60)
try:
    dest = download(
        "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE182nnn/GSE182320/suppl/GSE182320_CD4Atlas.h5Seurat",
        "GSE182320_CD4Atlas.h5Seurat"
    )
    df = extract_metadata_from_h5seurat(dest)
    print(f"  Result: {df.shape}")
    if not df.empty:
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample barcodes: {df['barcode'].head(3).tolist()}")
        for col in ['Condition', 'functional.cluster', 'Sample']:
            if col in df.columns:
                print(f"  {col}: {df[col].value_counts().head(3).to_dict()}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Test 5: HDF5 generic (10x format) ──
print("\n" + "=" * 60)
print("TEST 5: HDF5 - 10x CellRanger format")
print("=" * 60)
try:
    dest = download(
        "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE117nnn/GSE117963/suppl/GSE117963_10X_lineage_positive_filtered_gene_bc_matrices_h5.h5",
        "GSE117963_10x.h5"
    )
    df = extract_metadata_from_hdf5(dest)
    print(f"  Result: {df.shape}")
    if df.empty:
        print(f"  (Expected: CellRanger format has no metadata, only barcodes)")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Test 6: Barcode alignment end-to-end ──
print("\n" + "=" * 60)
print("TEST 6: End-to-end alignment (GSE182320)")
print("=" * 60)
try:
    # Check if we have processed data for GSE182320
    ds_path = Path("/mnt/projects/debruinz_project/cellarium/dataset/GSE182320")
    if ds_path.exists():
        gsms = [d.name for d in ds_path.iterdir() if d.is_dir() and d.name.startswith("GSM")]
        print(f"  GSE182320 has {len(gsms)} processed GSMs: {gsms[:5]}")
        if gsms:
            # Load the h5seurat metadata
            dest = TMPDIR / "GSE182320_CD4Atlas.h5Seurat"
            if dest.exists():
                df = extract_metadata_from_h5seurat(dest)
                print(f"  Author metadata: {len(df)} cells")
                # Align to first GSM
                aligned, stats = align_author_metadata(df, "GSE182320", gsms[0])
                print(f"  Alignment stats for {gsms[0]}: {stats}")
                if not aligned.empty:
                    print(f"  Aligned: {aligned.shape}, columns: {list(aligned.columns[:10])}")
    else:
        print(f"  GSE182320 not in processed dataset, trying GSE178088")
        ds_path = Path("/mnt/projects/debruinz_project/cellarium/dataset/GSE178088")
        if ds_path.exists():
            gsms = [d.name for d in ds_path.iterdir() if d.is_dir() and d.name.startswith("GSM")]
            print(f"  GSE178088 has {len(gsms)} processed GSMs: {gsms[:5]}")
            if gsms:
                dest = TMPDIR / "GSE178088_cluster3.h5Seurat"
                if dest.exists():
                    df = extract_metadata_from_h5seurat(dest)
                    aligned, stats = align_author_metadata(df, "GSE178088", gsms[0])
                    print(f"  Alignment stats for {gsms[0]}: {stats}")
                    if not aligned.empty:
                        print(f"  Aligned: {aligned.shape}")
        else:
            print(f"  Neither GSE in processed dataset")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
