# PyTorch Integration

singlet provides native PyTorch sparse tensor support, optimized for GPU machine learning.

## Zero-Copy Sparse Tensors

The fastest path from .1pz to GPU:

```python
from singlet.torch import to_sparse_csr

# Load as sparse CSR tensor (preferred for GPU — cuSPARSE backend)
tensor = to_sparse_csr("data.spz", device="cuda")
print(tensor.shape)   # (n_cells, n_genes)
print(tensor.device)  # cuda:0

# COO format (alternative)
from singlet.torch import to_sparse_coo
tensor = to_sparse_coo("data.spz", device="cuda")
```

CSR format is preferred for GPU because PyTorch routes CSR operations through cuSPARSE, which is highly optimized for sparse linear algebra.

## From AnnData

If you already have an AnnData object:

```python
from singlet.torch import from_anndata

tensor = from_anndata(adata, device="cuda", dtype="float32")
```

## DataLoader for Training

```python
from singlet.torch import DataLoader

# Single dataset
loader = DataLoader("GSE136831", batch_size=512, shuffle=True, device="cuda")

for batch in loader:
    # batch shape: (512, n_genes), dtype: float32
    embeddings = encoder(batch)
    loss = criterion(embeddings, ...)

# Multiple datasets concatenated
loader = DataLoader(
    ["GSE136831", "GSE171524", "GSE184462"],
    batch_size=1024,
    device="cuda",
)

# Gene subset (reduce memory)
loader = DataLoader(
    "GSE136831",
    genes=["TP53", "BRCA1", "KRAS"],
    batch_size=256,
)
```

## Custom Dataset

```python
from singlet.torch import SpzDataset

dataset = SpzDataset(
    "GSE136831",
    genes=["TP53", "BRCA1"],  # Optional gene subset
    device="cuda",
    sparse=True,  # Return sparse tensors instead of dense
)

print(len(dataset))       # Number of cells
print(dataset.n_genes)    # Number of genes

cell = dataset[0]         # Single cell tensor
```

## Sparse Matrix Operations on GPU

```python
import torch
from singlet.torch import to_sparse_csr

# Load sparse data
X = to_sparse_csr("data.spz", device="cuda")

# Sparse × dense matrix multiply (cuSPARSE)
W = torch.randn(X.shape[1], 100, device="cuda")
embeddings = torch.sparse.mm(X, W)  # (n_cells, 100)

# Row-wise operations
row_sums = torch.sparse.sum(X, dim=1)
```

## Memory Considerations

| Format | Memory | GPU Transfer |
|--------|--------|-------------|
| Dense | `n_cells × n_genes × 4B` | Slow |
| Sparse CSR | `~nnz × 8B` | Fast |
| .spz on disk | `~nnz × 0.8B` | N/A (decompressed) |

For a typical dataset with 50k cells, 30k genes, 5% density:
- Dense: 6.0 GB
- Sparse CSR: ~600 MB
- .spz file: ~60 MB
