# singlet (R) — build and portability notes

This document explains how the package's C++ source is compiled and what
system requirements must be satisfied on the build host. It's aimed at
package maintainers and at users who hit install-time build failures.

## System requirements

The package links **`libzstd`** (>= 1.4). zstd is widely available as a
system package and is a build dependency of many CRAN packages already
(e.g. `arrow`, `hdf5r`, any package using the **zstd** codec in Arrow),
so in practice most users already have it installed.

| Platform | Install command |
|---|---|
| Debian / Ubuntu | `sudo apt-get install libzstd-dev` |
| RHEL / Fedora   | `sudo dnf install libzstd-devel` (or `yum`) |
| macOS (Homebrew)| `brew install zstd` |
| macOS (MacPorts)| `sudo port install zstd` |
| Windows (Rtools)| Bundled with Rtools 4.x and later — no action needed |
| Conda           | `conda install -c conda-forge zstd` |

The package's `DESCRIPTION` declares `SystemRequirements: C++17, libzstd
(>= 1.4)` so that `install.packages()` and CRAN's package-check farm
both know to surface the dependency clearly if a build fails.

## Why we link system zstd instead of vendoring it

An earlier draft of the package considered bundling a copy of the zstd
sources inside `src/`. That approach was rejected for three reasons:

1. **CRAN size limits**. Bundled zstd adds ~1 MB of object code and
   pushes the package close to CRAN's 5 MB tarball cap — we'd lose
   headroom for future additions.

2. **Duplicate symbol hazards**. If a user's R session already has
   another CRAN package linked against system zstd (like `arrow`), a
   bundled copy would produce a second set of zstd symbols in a
   separate compilation unit. On Linux this usually works, but on
   macOS with some linker configurations it can cause
   hard-to-diagnose crashes when both copies fight for the same
   global state.

3. **Security**. A vendored copy gets a permanent version pin whereas
   the system copy updates with the OS. zstd has had CVEs and we want
   users to get system-level patches automatically.

## How the C++ decoder ends up in the R source tree

The reader is a **header-only** C++ library maintained in
`../include/singlet-pileup/pz_reader.h` at the top of the repo. It's
staged (copied) into the R package's `inst/include/singlet-pileup/`
directory so that `R CMD INSTALL` can find it via the
`-I../inst/include` flag in `src/Makevars`. The same header is also
included by the Python wrapper's `_pz_io.cpp`, so there is **one
source of truth** for the decoder.

When the canonical header changes, run:

```bash
cp ../include/singlet-pileup/pz_reader.h    inst/include/singlet-pileup/
cp ../include/singlet-pileup/pz_writer.h    inst/include/singlet-pileup/
cp ../include/singlet-pileup/sparse_accumulator.h inst/include/singlet-pileup/
```

from the R package root. A CI workflow under `.github/workflows/` does
this automatically on every push.

## Running `R CMD check --as-cran`

```bash
cd singlet/r
R CMD build .
R CMD check --as-cran singlet_0.2.0.tar.gz
```

Expected output: **0 ERRORs, 0 WARNINGs**. One NOTE is acceptable if
it's just the "SystemRequirements" line — CRAN accepts that form.

Any other NOTE or WARNING should be treated as a failing CI check.

## Cross-platform CI

The package's GitHub Actions workflow
(`.github/workflows/R-CMD-check.yml`) runs `R CMD check --as-cran` on
the matrix of:

- Ubuntu 22.04, R 4.3 and R 4.4
- macOS 13, R 4.3 and R 4.4
- Windows Server 2022, R 4.3 and R 4.4

On Linux and macOS the CI job installs `libzstd-dev` before R setup. On
Windows, Rtools 4.x ships zstd so no extra step is needed.

## Build failures — what to check

1. **`zstd.h: No such file or directory`**: missing `libzstd-dev`. Install
   it from your system package manager.

2. **`undefined reference to 'ZSTD_decompress'`**: the linker can't find
   `libzstd.so` / `libzstd.dylib` / `zstd.lib`. Check
   `pkg-config --libs libzstd` — it should print `-lzstd`. On macOS
   with Homebrew you may need `PKG_LIBS="-L/opt/homebrew/lib -lzstd"`
   in `~/.R/Makevars`.

3. **`error: 'auto' in function return type without a trailing return
   type`**: your compiler doesn't support C++17. Upgrade to gcc >= 7,
   clang >= 5, or MSVC 2017+.

4. **`inst/include/singlet-pileup/pz_reader.h: No such file or
   directory`**: the header staging didn't happen. Run the `cp`
   commands in the "How the C++ decoder ends up in the R source tree"
   section above.
