# PR-2 gate G1: ctest on the host

First test builds (jobs 800972 to 800974) ran ctest inside the Ubuntu 22.04 container. 27 test binaries need libc `GLIBC_2.38` and did not start. 4 tests (`test_pz2_*`) were never built because `zstd.h` was not on the include path. The same 31 tests failed on the base `perf/gpu-next` f15eebb6.

Fix (script `ctest-host2.sbatch`, in `$W/jobs/gpu-pr2/`): build in the container with `-I$ENV/include`, run ctest on the host (glibc 2.39).

| Build | Job | Result |
|---|---|---|
| candidate cd6a630e | 801021 | 100% passed of 157. 156 passed, 1 skipped (`test_gpu_pileup_labels`, needs a GPU, the build node has none) |
| base f15eebb6 | 801022 | 100% passed of 156. 155 passed, 1 skipped (same test) |

`test_strand_restart_env` and `test_stream_strand_restart` pass. Not covered by this job: the end-to-end restart test (needs the full `singlet` binary) and the GPU-only `test_gpu_pileup_labels`. Logs are in `ctest-host-*.txt` next to this file. The two earlier runs in the same job family (801018, 801019) without the include fix gave 4 not-built tests on both builds.
