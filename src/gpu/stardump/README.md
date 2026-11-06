# stardump: the STAR-side oracle of the GPU kernel gates

This directory is not dead code. No CMake file builds it, because it is built by hand from a scratch copy of `src/star` by the bench jobs below. It is the oracle against which the GPU mapping kernels are checked.

**What it is.** The vendored STAR 2.7.11b of `prod/campaign` with two files replaced in a scratch copy:

| File here | Replaces in the scratch copy | What it does |
|---|---|---|
| `ReadAlign_maxMappableLength2strands_dump.cpp` | `ReadAlign_maxMappableLength2strands.cpp` | dumps every seed search (the seed records) |
| `ReadAlign_oneRead_dump.cpp` | `ReadAlign_oneRead.cpp` | maps each distinct sequence with the run's STAR parameters and writes the M5 truth (STAR's final transcripts in the sequence-cache encoding) |
| `stardump_main.cpp`, `m5_dump.h`, `dump_format.h` | | driver, the parameters as the GPU harness reads them, the dump file format |

The harnesses `bench/gpu/m5/m5.cu` and `bench/gpu/m3m4` read these dumps and compare their own records byte for byte. The matches quoted for the exact STAR port (every distinct sequence of the 2 M set, and the seed records) come from this chain.

**Which scripts build or use it**

- `bench/gpu/jobs/stardump_build.sbatch`: builds the binary into `$W/jobs/w11/bin/stardump`.
- `bench/gpu/jobs/m3m4_compile.sbatch`, `bench/gpu/jobs/m3m4_run.sbatch`: seed search gate.
- `bench/fable/engine/fable_gate.sbatch`: the engine gate.
- `bench/fable/order/e4m_gate.sbatch`: the E4 merge gate.
- `bench/fable/census/jobs/census_run.sbatch`: the E2 census.

The scratch copy compiles against the current `src/star` headers, so the duplicate headers in `include/singlet/star/` do not affect it.

**Rule for cleanups.** Keep this directory as long as a kernel-level identity gate against STAR is part of the project. Remove it only together with those scripts, by an explicit decision, and note its last commit SHA in the removal commit.
