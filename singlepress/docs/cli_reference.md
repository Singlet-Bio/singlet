# CLI Reference

## Overview

singlepress provides a command-line interface for inspecting and validating .1pz files.

```bash
singlepress <command> <file.1pz>
```

## Commands

### info

Display file header information without decompression.

```bash
$ singlepress info counts.1pz
.1pz v3 | 30000 × 10000 | nnz=15000000
Codec: vocsc+zstd (level 3)
Chunks: 10 (1024 cols/chunk)
Metadata: yes | Colsums: yes | Transpose: no | Obs/Var: yes
```

### validate

Check CRC32 integrity of the file.

```bash
$ singlepress validate counts.1pz
File CRC32: OK
Footer: OK
Valid: True
```

### colsums

Print column sum statistics.

```bash
$ singlepress colsums counts.1pz
Column sums (first 10):
  [0] 1523  [1] 2847  [2] 956 ...
Total: 150000000
Min: 234  Max: 8912  Mean: 15000.0
```

### inspect

Detailed file inspection with compression ratio.

```bash
$ singlepress inspect counts.1pz
Path: counts.1pz
Size: 45.2 MB
Format: .1pz v3
Shape: 30000 × 10000
NNZ: 15,000,000 (density: 5.0%)
Compression ratio: 13.2×
Has metadata: yes
Has obs/var: yes
Has transpose: no
```
