# PR-1 gate: 16-input corpus on the strand-restart fix (65c82935)

Job 800976 (GPU, g052, T=8, 631 s). CPU references: array 800975. Binary: `gpu-pr2/bin2/65c82935/singlet-gpu`, built by job 800973 with the fixed build script (commit file written by the script, sha256 ba69c417...). Scripts: `bench/gpu-paths` tip 43cf834e. `ALLOW_NO_EMBED=1`.

**Verdict: corpus FAILED as the harness counts it (11 of 16 pass). The restart gate itself is met: strand100 is identical to the CPU engine and does not hang.**

| Input | Result | Reason |
|---|---|---|
| strand100 | identity IDENTICAL, no hang (51 s), restarted True. Harness verdict FAIL | The path check expects the log line `low-quality masks for N reads`. This run never prints it. All other checks pass. Same shape in the earlier integration run (e80aeb65). |
| strand150, v5p150, mask3syn, long_q4 | identity IDENTICAL. Harness verdict FAIL | Reads over 128 bases decline level 5, ingest and the GPU pileup on this branch. The 128-base lift is a separate change. |
| the other 11 inputs | PASS | |

Before the fix the same input timed out (jobs 800495 and 800496 on c92a494b and e42c0636).

Open: why strand100 prints no mask line. It is not explained, so the corpus is reported as FAILED and not PASS.
