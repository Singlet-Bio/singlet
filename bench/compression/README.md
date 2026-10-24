# bench/compression

Compression study of `.singlet` transcriptomics matrices (`.1pz` members). Read `PLAN.md` first.

- `build.sh` builds everything with the Rtools 4.4 GCC toolchain (no CMake, no MSVC needed).
- `harness/` shared types, `.1pz` loading, roundtrip checks, the `bench` driver.
- `kernels/` one header per experiment track plus the reference codecs; `registry.h` wires them together.
- `tools/` Python helpers (corpus fetch over HTTP range requests, byte breakdown, training scripts).
- `corpus/` downloaded data (git-ignored); `corpus.tsv` says what and why.
- `results/` TSV rows appended by the harness; `notes/` per-track findings; `REPORT.md` the synthesis.
