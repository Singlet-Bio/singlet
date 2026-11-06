# PR-2 gate G3: 16-input corpus on the candidate (c8849bb2)

Job 800978 (GPU, g052, T=8, 655 s). CPU references: array 800977. Binary `gpu-pr2/bin2/c8849bb2/singlet-gpu` (build job 800974, fixed build script, sha256 f1bbc00f...). `ALLOW_NO_EMBED=1`. The branch head is later than c8849bb2 only by gate-result files.

**Verdict: corpus FAILED as the harness counts it (11 of 16 pass). Output is IDENTICAL to the CPU engine on every input that produced output. strand100 does not hang.**

The result matches the restart-fix build (job 800976) input by input.

| Input | Harness verdict | Reason |
|---|---|---|
| strand100 | FAIL (path) | identity IDENTICAL, no hang. The expected log line `low-quality masks for N reads` is absent in the run. Not explained. |
| strand150, v5p150, mask3syn, long_q4 | FAIL (path, fallback) | Reads over 128 bases decline level 5 and the GPU pileup on this branch. Identity IDENTICAL. The 128-base lift is a separate change. |
| 11 other inputs | PASS | |

E20 and WSHRINK are on the default path here only as far as their defaults go: the E20 hold split and the device clip are on, WSHRINK is off. WSHRINK is exercised in the 10-sample gate (jobs 800981 and 800982) and in the identity and timing jobs.
