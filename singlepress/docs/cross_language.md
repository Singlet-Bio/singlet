# Cross-Language Compatibility

## Overview

`.1pz` files are binary-identical across all language implementations. A file written
by Python singlepress can be read by the R/Rcpp implementation, and vice versa.

## Python → R

```python
# Python: write .1pz
import singlepress as sp
import scipy.sparse as ss

mat = ss.random(10000, 5000, format="csc")
sp.write_1pz("shared.1pz", mat,
              rownames=[f"gene_{i}" for i in range(10000)],
              colnames=[f"cell_{i}" for i in range(5000)])
```

```r
# R: read the same file
mat <- read_1pz("shared.1pz")
dim(mat)           # [1] 10000  5000
rownames(mat)[1:3] # "gene_0" "gene_1" "gene_2"
```

## R → Python

```r
# R: write .1pz
write_1pz(mat, "shared.1pz")
```

```python
# Python: read the same file
mat = sp.read_1pz("shared.1pz")
print(mat.shape, mat.rownames[:3])
```

## What's Cross-Compatible

| Feature | Python | R |
|---------|--------|---|
| Matrix values | ✓ | ✓ |
| Rownames / colnames | ✓ | ✓ |
| Column sums | ✓ | ✓ |
| CRC32 validation | ✓ | ✓ |
| obs/var DataFrames | ✓ | ✓ |
| uns key-value pairs | ✓ | ✓ |
| Transpose section | ✓ | Read only |
| Column-range reads | ✓ | Planned |
| `[i, j]` indexing | ✓ | ✓ |
| `cbind_1pz` | ✓ | ✓ |
| `rbind_1pz` | ✓ | ✓ |
| `subset_1pz` | ✓ | ✓ |
| `sample_1pz` | ✓ | ✓ |
| Summary stats | ✓ | ✓ |
| head / tail | ✓ | ✓ |
| AnnData aliases | ✓ | ✓ |
| MTX / CSV / loom I/O | ✓ | — |
| PyTorch dataloaders | ✓ | — |
| Seurat integration | — | ✓ |
| SCE integration | — | ✓ |

## Verification

```python
# Python: validate round-trip integrity
sp.validate_1pz("shared.1pz")
```

```r
# R: validate the same file
validate_1pz("shared.1pz")
```
