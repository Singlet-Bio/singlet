# The GPU path on any single device

A run needs no per-device flag. The engine reads the device's memory and compute capability, the pool reads the size of each genome index, and
`include/singlet/gpu_plan.h` turns the two into settings. Anything a user sets is kept as given.

## What is chosen

| Setting | Chosen from | User override |
|---|---|---|
| Compute floor | default 70, so the V100S (7.0) and RTX 8000 (7.5) run; kernels are compiled for 70, 75, 80 and 90 | `SINGLET_ENGINE_GPU_MIN_CC` |
| GPU path or CPU engine | genome index + 2.5 GB working set + 2 GB margin must fit the device | none (the sample reports one line and runs on the CPU engine) |
| Non-host screen | resident server when the device holds screen index (25 GB), genome index and one sample beside each other; loaded per sample when it holds the screen index alone; CPU screen below that | `--nonhost-server on\|off\|auto` |
| Slots per device | room left beside the shared index and the screen, in units of 16 GB, at most 2 | `--slots N`, `SINGLET_ENGINE_GPU_SLOTS` |
| Shared index | on with 2 or more slots, off with 1 | `--shared-index on\|off\|auto`, `SINGLET_GPU_SHARED_INDEX` |

`singlet gpu-pool` logs the result in one line (`device plan: ...`). `singlet gpu-info [N]` prints a device's memory, capability and name.

## What each device gets

| Device | Memory | Human index (16.4 GB) | Mouse (25.0 GB) | Barnyard (52.7 GB) |
|---|---|---|---|---|
| H100 NVL | 93.6 GiB | 2 slots, shared index, resident screen | 2 slots, shared index, resident screen | 1 slot, screen loaded per sample |
| RTX 8000 | 44.5 GiB | 1 slot, screen loaded per sample | 1 slot, screen loaded per sample | CPU engine |
| V100S | 31.7 GiB | 1 slot, screen loaded per sample | 1 slot, screen loaded per sample | CPU engine |

## Measurements behind the table

Single-device pool runs on pbmc1k_v3 (3') and pbmc5p_v1 (5'), HLA and editing off, no resident screen, 4 threads per sample
(`results/xdev-*`, jobs 805776 to 805779). Rates are pairs per second over the span of the queue.

| Device | 1 slot, own index | 1 slot, shared index | 2 slots, shared index |
|---|---|---|---|
| RTX 8000, 3' | 0.720 M | 0.660 M | see below |
| V100S, 3' | 0.698 M | 0.631 M | 0.620 M, 2 of 9 samples failed the memory budget |
| V100S, 5' | 0.575 M | 0.609 M | see below |

On the 32 GB device a second slot leaves each sample 6.7 GB beside the shared index, below the working set of the lean reads phase, so some samples
fail the budget; the plan gives it one slot. The 46 GB device has the same limit with a smaller margin.

## Checks

`tests/cpp/test_gpu_plan.cpp` covers the table above, the fit limit to the byte, the compute floor, overrides, the unknown-device case and the
hand-over of the device facts to child processes.
