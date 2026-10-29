# singlet (R) — build and portability notes

How the package's native code is compiled, for maintainers and for users who hit an install-time build failure.

## What is compiled

- `src/pz_io.cpp`, `src/pz2_io.cpp`: Rcpp bindings for the header-only C++ `.1pz` / `.s1` matrix codecs in
  `inst/include/singlet/` (the same headers the Singlet Python package compiles).
- `src/zstd_unity.c`: the bundled Zstandard 1.5.6 library (`src/zstd/`, BSD licence, see `inst/COPYRIGHTS`), compiled as one
  translation unit in the way zstd's own single-file build does, without the multithreaded compressor, the dictionary
  builder, legacy formats or the x86-64 assembly decoder.

The only requirement is a C++17 compiler (`SystemRequirements: C++17`): R's own toolchains (Rtools on Windows, the CRAN
macOS toolchain, gcc/clang on Linux) are enough. No system `libzstd` is needed.

## Why zstd is bundled

Earlier versions linked the system `libzstd` (`-lzstd`). That needs `libzstd-dev` / `libzstd-devel` / `brew install zstd` on
the build host and is not guaranteed on every CRAN builder. Bundling removes the system dependency on every platform. The
cost is about 0.4 MB in the source tarball and one larger object file. R loads package shared objects with local symbol
scope, so the bundled copy does not collide with another package's zstd.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `C++17 standard requested but CXX17 is not defined` | update the compiler (gcc >= 7, clang >= 5) or R's `Makeconf` |
| compile errors inside `src/zstd/` | report them with the compiler version: the sources are upstream zstd 1.5.6 |
