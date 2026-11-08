# GPU slot scheduling: where a loaded node waits

This page explains why a fully loaded GPU node maps fewer reads per second than its kernels allow, and what the any-slot lock wait changes. It extends §1 of [RUNTIME-REFERENCE.md](RUNTIME-REFERENCE.md), which describes the device lock, its permits and the memory budget. All numbers come from real 10x samples on a Clipper H100 NVL node with two GPUs, two permits (slots) per GPU and the non-host server off.

## How a sample holds the device

A sample takes the device lock several times. The prefill hold is the long one: it uploads the index and reads, runs the clip, entries, dedup and the three mapping lanes, and copies the E10 entries back. The other holds are short. The record upload, the Solo pass and the pileup each take a few seconds, and the non-host screen holds the device while it loads its own index.

One loaded sample from the node A/B shows the shape.

| Hold | Held (s) | Waited before it (s) |
|---|---|---|
| Prefill | 26.2 | — |
| Record upload | 2.0 | 72.9 |
| Solo | about 1 | 7.2 |
| Pileup | 2.6 | — |
| Non-host screen (index load) | 5 to 8 | — |

Inside the prefill, the mapping lanes A, M and B run 5.8, 5.4 and 4.6 s. The index upload, the read upload and the E10 download each take about 2 s, since staged copies move about 4 GB/s.

## Why the record upload waited 73 s

Before the any-slot change, a process that found every permit busy picked one permit from its process id and blocked on it. A sample that needed a two-second record upload could therefore sit behind a 26-second prefill on its chosen permit while the other permit of the same GPU came free and went to someone else. Short holds queued behind long ones, and the sample kept its host memory, threads and output stream open the whole time. On a node running many samples at once, that wait adds directly to every sample's wall time.

A third permit per GPU did not help. It was measured slower than two, because three prefills compete for the same copy engines and memory, and the device budget per slot shrinks.

## The any-slot wait

The any-slot wait (`gpu/anyslot`) polls every permit instead of blocking on one.

- A waiting process tries each permit of each eligible device with a non-blocking `flock` every 10 ms and takes the first one that frees.
- A process waiting for a short hold also takes a shared lock on `/tmp/singlet_engine_gpu_<PCI bus id>.prio`.
- A prefill skips any device whose `.prio` file it cannot lock exclusively. Short holds therefore go first whenever one is waiting, and a prefill starts only on a device where none is.
- `SINGLET_ENGINE_GPU_DEVICE` still pins the device. `SINGLET_ENGINE_GPU_LOCKWAIT=fixed` restores the earlier wait for A/B runs.

The change touches only scheduling. Its outputs are byte-identical to the CPU engine on the eight gate samples and on the mixed-species hgmm sample, and the unit tests pass. A node A/B against the fixed wait, with and without the non-host server, measures its throughput effect.

## Device memory under load

Every arm of the launch A/B retried at least one sample after a device allocation failed. Most retries succeed on the second or third attempt. A sample that fails every attempt is mapped on the CPU, which is correct but slow: in one arm two such samples took nearly twice as long as any other run. A failed host-side allocation inside the GPU mapping step now also falls back to the CPU, with a device reset, where it used to end the process. The retries show that two prefills plus the non-host index leave little headroom on a 94 GB device. Shrinking the per-slot plan, or sharing one device copy of the index between the two slots of a GPU, would remove most of them.

## Next candidates

- One device copy of the index per GPU, shared by its slots. This saves an upload and about 16 GB of device memory per extra slot.
- CUDA context creation and pinned-buffer setup before the lock, so that the prefill hold starts with the first copy.
- Overlapping the E10 download with the last lane, which takes about 2 s off each prefill hold.
