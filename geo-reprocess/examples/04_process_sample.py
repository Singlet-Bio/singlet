"""
Single sample processing example.

This script demonstrates how to process a single sample for testing.
"""
from scgeo import process_sample

# Process a known 10x v2 sample
print("Processing sample GSM3308545...")
result = process_sample(
    gsm_id="GSM3308545",
    gse_id="GSE115978",
    organism="human",
)

print(f"\nResult:")
print(f"  Status: {result.status}")

if result.status == "success":
    print(f"\n  Download:")
    print(f"    Time: {result.download.download_time_s:.1f}s")
    print(f"    Size: {result.download.total_bytes / 1e9:.2f} GB")
    
    print(f"\n  Detection:")
    print(f"    Protocol: {result.detection.protocol}")
    print(f"    Confidence: {result.detection.confidence:.2f}")
    print(f"    Chemistry: {result.detection.chemistry}")
    
    print(f"\n  Quantification:")
    print(f"    Mapping rate: {result.quantification.mapping_rate:.1%}")
    print(f"    Time: {result.quantification.quant_time_s:.1f}s")
    
    print(f"\n  QC:")
    print(f"    Cells: {result.qc.n_cells:,}")
    print(f"    Median UMI/cell: {result.qc.median_umi_per_cell:,.0f}")
    print(f"    Median genes/cell: {result.qc.median_genes_per_cell:,.0f}")
    print(f"    Passed QC: {result.qc.passed_qc}")
    
    print(f"\n  Total time: {result.total_time_s:.1f}s")
else:
    print(f"  Error: {result.error}")
