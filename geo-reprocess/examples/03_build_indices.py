"""
Index building example.

This script demonstrates how to build reference indices for quantification.
"""
from scgeo import build_index

# Build index for human
print("Building human reference index...")
print("This will:")
print("  1. Download genome FASTA (~3 GB)")
print("  2. Download GTF annotations (~60 MB)")
print("  3. Build splici reference")
print("  4. Build piscem index (~10 min)")
print()

human_index = build_index(
    organism="human",
    index_type="piscem",
)

print(f"\n✓ Human index ready: {human_index}")

# Build index for mouse
print("\nBuilding mouse reference index...")
mouse_index = build_index(
    organism="mouse",
    index_type="piscem",
)

print(f"✓ Mouse index ready: {mouse_index}")

# Check available indices
from scgeo.indices import list_available_indices

indices = list_available_indices()
print("\nAvailable indices:")
for organism, index_types in sorted(indices.items()):
    print(f"  {organism:20s} {', '.join(index_types)}")
