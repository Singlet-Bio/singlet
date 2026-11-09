# Barcode whitelists

The engine looks for these files in `--whitelist-dir`, then `$SINGLET_WHITELIST_DIR`, then `<binary dir>/../whitelists`.
Each protocol names its file in `include/singlet/fq/protocol.h` (`known_protocols()`).

Most lists are in this directory. Lists rebuilt from 10x public datasets, with their sources and checksums, are described in
`ATTRIBUTION.md`.

The 10x GEM-X lists are fetched instead of bundled:

```
singlet whitelists fetch [--dir DIR]   # downloads them over HTTPS and checks each sha256
singlet whitelists list  [--dir DIR]   # shows whether each is installed
```

| file | chemistry | source | sha256 |
|---|---|---|---|
| `3M-3pgex-may-2023.txt` | GEM-X 3' v4 (`10x-3p-v4`) | https://storage.googleapis.com/gcp-public-data--broad-references/optimus_whitelists/3M-3pgex-may-2023.txt | `62df04844ed0d1281b302f862ec0d005d3298fc169f507f68dd6e8d4cb5ccccd` |
| `3M-5pgex-jan-2023.txt` | GEM-X 5' v3 (`10x-5p-v4`) | https://storage.googleapis.com/gcp-public-data--broad-references/optimus_whitelists/3M-5pgex-jan-2023.txt | `b402db8acbcd3a076bf50991538d6c08464ddb137e0b81916a9fa92cf222dd0d` |
