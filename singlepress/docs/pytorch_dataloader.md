# PyTorch Zero-Copy Dataloader

## Overview

`singlepress.torch` provides PyTorch-compatible dataloaders that decode .1pz chunks
directly into sparse CSR tensors compatible with cuSPARSE operations on GPU.

The key design principles:

1. **Chunk-aligned reads** — each DataLoader item decodes exactly one .1pz chunk (~1024 cells)
2. **Zero-copy to cuSPARSE** — CSR indptr/indices/values arrays map directly to `torch.sparse_csr_tensor`
3. **Multi-worker safe** — each worker opens independent file handles, no coordination needed
4. **Pin memory** — tensors can be pinned for async GPU transfer with `non_blocking=True`

## Map-Style Dataset

```python
from singlepress.torch import OnePZDataset, collate_sparse
import torch

dataset = OnePZDataset(
    "counts.1pz",
    chunk_size=1024,     # cells per item (matches .1pz chunk alignment)
    normalize=True,      # apply log-normalization using stored colsums
    scale=10000,         # normalization scale factor
    dtype="float32",     # value dtype for tensors
    sparse_format="csr", # "csr" (cuSPARSE) or "coo"
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,                    # 4 chunks = ~4096 cells per batch
    collate_fn=collate_sparse,       # concatenate sparse tensors
    num_workers=4,                   # parallel chunk decoding
    pin_memory=True,                 # for async GPU transfer
    shuffle=True,                    # shuffle chunk order
)

for batch in loader:
    # batch is a sparse CSR tensor on CPU (pinned memory)
    gpu_batch = batch.to("cuda", non_blocking=True)

    # Use with cuSPARSE-backed operations:
    # torch.sparse.mm(gpu_batch, weight_matrix)
    # torch.sparse.sum(gpu_batch, dim=1)
```

## Iterable-Style Dataset

For streaming large files without random access:

```python
from singlepress.torch import OnePZIterableDataset

dataset = OnePZIterableDataset(
    "counts.1pz",
    normalize=True,
)

# Automatically shards across workers
loader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
)

for chunk in loader:
    gpu = chunk.to("cuda", non_blocking=True)
```

## Sparse Format Details

### CSR (Default)

The default `sparse_format="csr"` produces `torch.sparse_csr_tensor` objects:

```python
tensor = dataset[0]
# Access cuSPARSE-compatible arrays:
tensor.crow_indices()  # int32, shape (nrows+1,)
tensor.col_indices()   # int32, shape (nnz,)
tensor.values()        # float32, shape (nnz,)
```

CSR tensors support efficient operations via cuSPARSE:
- `torch.sparse.mm(csr, dense)` — sparse-dense matrix multiply
- `torch.sparse.sum(csr, dim=...)` — sparse reductions
- `.to_dense()` — explicit densification

### COO (Alternative)

Use `sparse_format="coo"` for operations that prefer COO layout:

```python
dataset = OnePZDataset("counts.1pz", sparse_format="coo")
tensor = dataset[0]
tensor.coalesce().indices()  # int64, shape (2, nnz)
tensor.coalesce().values()   # float32, shape (nnz,)
```

## Training Loop Example

```python
import torch
import torch.nn as nn
from singlepress.torch import OnePZDataset, collate_sparse

class SparseAutoencoder(nn.Module):
    def __init__(self, n_genes, n_latent=128):
        super().__init__()
        self.encoder = nn.Linear(n_genes, n_latent)
        self.decoder = nn.Linear(n_latent, n_genes)

    def forward(self, x):
        # x is sparse CSR; convert to dense for linear layers
        x_dense = x.to_dense()
        z = self.encoder(x_dense)
        return self.decoder(z)

dataset = OnePZDataset("counts.1pz", normalize=True)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, collate_fn=collate_sparse,
    num_workers=4, pin_memory=True, shuffle=True,
)

model = SparseAutoencoder(dataset.num_features).cuda()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for batch in loader:
        batch = batch.to("cuda", non_blocking=True)
        output = model(batch)
        loss = nn.functional.mse_loss(output, batch.to_dense())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Performance Notes

- **Chunk decode** is typically I/O-bound on NVMe; `num_workers=4` saturates most SSDs
- **CSR GPU transfer** with `pin_memory=True` and `non_blocking=True` overlaps with decode
- **No Python GIL contention** — the C++ decoder releases the GIL during decompression
- **Memory efficient** — only one chunk (~1024 cells) materialized per worker at a time
