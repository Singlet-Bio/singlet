# Heavy reads on a second GPU: design and feasibility

Status: design only. Nothing here is built. Numbers come from the b0 throughput runs of gpu/bold (commit c3a6096c, pbmc1k_v3, level 5, k8 T7 2 slots, job 800570) and from the two-GPU runs of branch bench/tx-first (`bench/tx_first/s2_multigpu.sbatch`, summary at `jobs/tx-first/results/overnight/summary.md` on Clipper).

## Question

Could one GPU serve the heavy-lane reads of processes that run on the other GPU?

## Measured starting point

The heavy lane (lane B) holds repeats and multimappers. In one pbmc1k_v3 run it takes 1.36 M of 31.7 M distinct reads. That is 2.0% of read pairs.

Lane kernel time per run is 3.5 s for lane A, 3.7 s for lane M and 5.5 s for lane B. Lane B is 43% of kernel time.

Each run holds the device lock for about 42 s. The prefill, the lanes and the record build all sit inside that time. On a busy single-GPU node the processes wait about 46% of their wall time for the lock. The throughput is about 1.7 M pairs/s per GPU.

Lane B needs 17.5 GB of work space per batch on top of the 16.4 GB index. The device peak of a process is 85 to 89 GB.

## What a helper GPU would need

**Index copy.** The helper GPU needs the full genome index (16.4 GB) once. All client processes can share it. CUDA IPC handles already work between processes (`bench/engine_gpu_gate/ipc_test.cu`). Only the helper process owns the copy.

**Work arena.** The helper keeps one lane B work arena (17.5 GB). Client processes drop their own lane B arena. Their device budget falls from 35 GB toward the lean plan of 19 GB. That is the main side benefit.

**Reads in.** A heavy read is a 2-bit sequence and its quality clip. About 1.4 M reads at roughly 100 bytes is 0.14 GB per run.

**Results out.** Lane B writes 4096 bytes per sequence, so 1.4 M reads give about 5.6 GB per run. The records must return in the original order to stay byte-identical. Over PCIe 5 at about 50 GB/s this is 0.1 s per run. Over NVLink it is less. The transfer is small next to the lane B kernel time.

**Scheduling.** Clients need a queue on the helper. Each client sends whole chunks of heavy reads (31744 reads per batch today). The helper runs them in arrival order. Chunk order per client stays fixed, so results do not depend on arrival order.

**Failure handling.** A client must fall back to its local lane B when the helper is busy or absent. Output stays byte-identical because the kernel is the same code.

## Expected gain

Lane B removes at most 5.5 s of 42 s of lock hold time on the home GPU. That is 13%. If the node were lock bound on one GPU, throughput would rise by up to 15%, minus queueing on the helper.

A second GPU used for whole processes doubles the lock capacity. That gives up to 2x at the same k when cores allow. Helper mode would then add at most 13% to the busy GPU and nothing to an idle one. Whole-process spreading is the larger and simpler gain. The overnight two-GPU runs measure it.

Helper mode only wins when one GPU is saturated and the other is idle. That happens when the account has one GPU job and the node has a free GPU. It does not happen when jobs fill both GPUs.

## Comparison

| Option | Gain estimate | Cost | Risk |
|---|---|---|---|
| Whole processes on both GPUs (no code change) | up to 2x lock capacity | none | CPU cores and host memory become the limit |
| Heavy lane on the second GPU | 10 to 15% on a saturated GPU | IPC index sharing, queue, ordered results | medium, byte-identity must hold |
| Heavy lane on the CPU | Lane B costs about 0.45 ms per read on one core in the host lane (38.8 core-s for 85.6 k reads). 1.4 M reads would take about 600 core-s per run against 5.5 GPU-s | none for the device | loses unless the step budget cuts the cost by more than 10x. Another track tests this tonight |
| Pooled batching service per GPU | removes per-process index copy and lane B arena, allows 3 to 4 slots | long-lived service, shared memory protocol | high, changes process model |

## Recommendation

First use both GPUs for whole processes. This needs no code. Then measure whether the lock wait stays high at k=16 on two GPUs. Build helper mode only if one GPU stays saturated while the other idles. A pooled service is the larger project. Its benefit is device memory freed per process, which allows more slots. Test slots first with MPS and 3 slots per GPU, as in the overnight run.
