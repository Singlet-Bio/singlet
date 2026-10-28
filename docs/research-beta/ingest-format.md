# Ingest format for the split pipeline (CPU nodes download and transform, GPU nodes align)

> Copy of the W11 research note; the prototype code, jobs and raw results are in `bench/gpu/ingest/` on the research branch (PR Singlet-Bio/singlet#61). Measured summary: `bench/engine_gpu/results/ingest.json`.

Status: design note with a measured prototype, written overnight 2026-10-28 (W11 side session). Code and raw results:
`bench/gpu/ingest/` (README with job ids), `bench/gpu/results/ingest.json`. Nothing here has been committed by this session.

## 0. Summary

| question | answer (all numbers measured on Clipper, real data; details below) |
|---|---|
| What to fetch | NCBI **SRA Lite** (`.sra` with simplified qualities), fetched from NCBI's own host with parallel ranged GETs to local scratch, then decoded from the local file. Full-quality `.sra` from public S3 is the fallback. ENA `fastq.gz` is a last resort (not mirrored for some runs, drops the barcode read for others, 1 MB/s from our nodes). |
| Why not VDB streaming | The existing `singlet download SRR...` (VDB streams over HTTPS) took 463.6 s and 5.9 core-s per M spots for one 61 M-spot run; 87% of that wall time is "VDB read". Fetch + local decode of a same-size run of the same sample: 16 s + 53 s wall, 2.24 core-s per M spots (6.7x less wall including the fetch, 8.8x for the decode alone; 2.6x fewer core-s). The first 4 M spots of one run: 50.5 s / 34 core-s streamed vs 7.6-8.0 s / 12.4-13.6 core-s from a local file. |
| New format | **`.1fg`**: chunked, struct-of-arrays, GPU-ready container of the reads the aligner needs: CB+UMI word (8 B), R2 2-bit slot (24 B for <= 96 nt), N/unknown-barcode exception list, per-section zstd frames (nvCOMP-compatible), CRC32C per section, no base qualities. 32,768 pairs per chunk. |
| Recommended codec | zstd level 3 on every section (level 1 if CPU-node time matters more than bytes; raw if the bytes are free). zstd 3: 20.8 B per pair on disk for PBMC 1k (`.1fq`: 27.6) and 16.9 for GSM6470208 (`.1fq`: 14.4). Bytes depend on the source being barcode-sorted per block (+17-37% if not). |
| CPU transform `.1fq` -> `.1fg` | 0.24-0.42 core-s per M pairs at zstd 1-3 including reading and decompressing the `.1fq`; 0.15-0.27 without that read (4-12% of the 2.24-5.9 core-s per M that the source decode costs). 1 thread: 3.4-4.4 M pairs/s; 8 threads: 17-22 M pairs/s. |
| GPU reader, host cost | **0.010-0.014 core-s per M pairs** (file read + H2D, zstd files), 0.019-0.020 for raw files; CRC32C check adds 0.004. The `.1fq` path (M8) costs 0.067 (GSM6470208) to 0.126-0.136 (PBMC 1k): 6-10x more. |
| GPU decode | nvCOMP 6.0 batched zstd on H100 NVL: **320 M pairs/s (10 GB/s decoded) at 128 frames per batch, 600 M pairs/s (19.3 GB/s) at 256**. With 256K-pair frames only 56 M pairs/s: nvCOMP runs one thread block per frame, so frames must be small and batches large. |
| End to end into GPU memory | 49-107 M pairs/s from cold node-local NVMe with 2-3 reader threads (1-1.9 GB/s file reads: the disk, not the GPU or PCIe, is the limit). |
| Identity | Every read of two real samples (66,601,887 and 121,774,155 pairs) decoded on the GPU equals the `singlet decode` FASTQ text (R1 and R2, 0 mismatches, 3 file variants each); the host decoder's byte-numeric reads hash equal to the `.1fq` library decode on 3 samples. |

## 1. What the pipeline needs from the format

The GPU aligner (M3-M6 in `docs/gpu-engine/exploration.md`) consumes, per read pair: cell barcode, UMI, R2 bases and R2 length. Base qualities
are not used (the atlas `.1fq` files have `qual_mode` 0). The format must therefore (1) be produced by CPU nodes at roughly download speed,
(2) cost almost no host CPU on the GPU node (M8 measured 0.085-0.175 core-s per M pairs for `.1fq`, which is 2.3-4.7 cores for a node at 27 M
pairs/s), (3) decode on the GPU, (4) be integrity-checkable per chunk, and (5) stay convertible to what the CPU engine reads today.

## 2. Download: which public format to fetch

Sizes (read-only: ENA `filereport`, NCBI SDL; jobs 789839, 789879, 789925; `bench/gpu/ingest/download_results.json`). Bytes in GB (1e9):

| run (sample) | pairs (M) | ENA fastq.gz | NCBI `.sra` | SRA Lite | `.1fq` | Lite / `.sra` |
|---|---|---|---|---|---|---|
| SRR9900123 (GSM4006843) | 66.7 | 7.48 (2 files) | 5.20 | 2.18 | - | 0.42 |
| SRR14999752 (GSM5411066) | 69.6 | 4.23 | 2.62 | 2.13 | - | 0.81 |
| SRR26840957 (GSM7901264) | 100.3 | 7.58 | 4.85 | 3.33 | - | 0.69 |
| SRR23353990 (GSM7036403) | 111.6 | 5.70 | 5.37 | 3.46 | 1.86 | 0.64 |
| SRR25376447 (GSM7656149) | 263.4 | **not in ENA** (read_count 0) | 26.85 | 13.91 | 5.93 | 0.52 |
| SRR21102187 (GSM6470208) | 61.1 | 2.81, **one file, 90 bases per spot: no barcode read** | 2.80 | 2.19 | 0.93 | 0.78 |

Observations:

- **ENA is not a complete source.** One of six runs is absent (read_count 0), one yields only the cDNA read (the barcode read is missing, so the
  FASTQ is useless for 10x), and a 1 GB ranged HTTPS read ran at **0.93 MB/s** (1071 s; job 789839) from a Clipper node. Use it only as a
  fallback when NCBI has no object.
- **SRA Lite** is 19-58% smaller than the normal `.sra` (median 0.67). It only simplifies qualities; for the first 4 M spots of SRR21102188 the
  `.1fq` decoded from the local Lite file and from the full-quality file are **byte-identical** (md5 of the R1 and R2 FASTQ equal, job 789933; the
  streaming accession path gave the same md5, job 789920). Decode CPU is the same (12.4 vs 12.8 user-s per 4 M spots), so Lite saves bytes only.
  One run, first 4 M spots: W4 should re-check equality on a few more runs.
- **Lite access.** The S3 (`sra-pub-zq-*`) and GCS copies are requester-pays (anonymous GET: 403 `AccessDenied` / `UserProjectMissing`, job 789925),
  but NCBI's own host (`sra-downloadb.be-md.ncbi.nlm.nih.gov`, `service: "ncbi"` in the SDL answer) serves anonymous ranged GETs (HTTP 206). The
  SDL query that lists it needs `accept-alternate-locations=yes`; the plain query returns only the full-quality object.
- **Throughput.** NCBI host, Lite, 2.17 GB: 99 MB/s with 1 stream, 106-234 MB/s with 8 ranged streams (four jobs). Public S3 full `.sra`: 187 MB/s
  with 8 streams (job 789933), but 11 MB/s (1 stream) and 28-56 MB/s in other windows when other lab download jobs shared the egress (jobs 789839,
  789920). Network speed is noisy; parallel ranged streams to local scratch are worth 3-10x.
- **VDB streaming is the slow part of today's fetcher.** `singlet download SRR21102187` (whole run, 61.09 M spots; job 789920): wall 463.6 s,
  361 core-s (5.9 core-s per M spots), 77% of one CPU on average; the profile says VDB read 397 s (86.6%), convert 31 s, writer 25 s (sort 17 s).
  The same tool on a **local** Lite file of a same-size run (SRR21102188, 60.68 M spots; job 789934): fetch 16.1 s (135 MB/s), decode 52.7 s wall,
  136 core-s (2.24 core-s per M spots); VDB read 9.3 s, and now the `.1fq` writer is 69% (sort + pack + compress 35 s).
- **FASTQ route cost.** Inflating real reads re-gzipped at level 6 costs 1.47 core-s per M pairs (264 B of text per pair; job 789903) before any
  parsing, i.e. in the same range as the local `.sra` decode (2.24). It is not cheaper in practice once ENA's gaps are counted.
- **BAM submissions** (original 10x BAMs on SRA / ENA `submitted_ftp`) were not measured: no accession in the sample set has one (`submitted_bytes`
  empty for all six runs). They are the slowest route (bam-stream workstream) and carry aligned reads, not raw barcodes; keep them as a per-study
  exception handled by the BAM path.

**Recommendation.** Per run: SDL query (with alternates) -> Lite on the NCBI host, else full `.sra` on S3, else ENA -> parallel ranged GET into job
`/tmp` (8 streams, retries per range, verify against the SDL `md5`/size) -> local decode -> transform -> `.1fg` -> upload. Never stream through VDB.

## 3. The `.1fg` format

Definition: `bench/gpu/ingest/ingest_fmt.h`. Little-endian, 4 KiB-aligned chunks.

```
FileHeader (256 B)  magic "1FG\0", version 1, codec (0 raw, 1 zstd), level, flags (CRC present), protocol_id, chunk_pairs (32768),
                    n_pairs, n_pairs_in, n_dropped, n_chunks, r2_slot (B), r2_max_len, bc_len, umi_len, bc_start, r1_len,
                    max_exc_bytes, meta offset/size, chunks_offset, index_offset, source_bytes
Metadata            one zstd-compressed JSON (provenance, encodings; the source .1fq metadata block is copied into it when present: it was EMPTY in the PBMC 1k file, the only one checked)
Chunk i (4 KiB aligned):  ChunkHeader (64 B): n_pairs, n_exc, flags (bit s: section s stored raw),
                          3 x {comp_size, raw_size, crc32c of the stored payload, offset}
                          section 0  CBUMI : n x 8 B   CB bits 0-31 (2 bit/base, first base high), UMI bits 32-55 (<= 12 nt, 2 bit/base),
                                             R2 length bits 56-62, bit 63 = read has an exception entry
                          section 1  R2    : n x r2_slot B   2 bit/base, first base in the top bits, zero padded
                          section 2  EXC   : u32 list sorted by pair: bits 0-19 pair index in chunk, 20-21 kind, 22-29 position
                                             kind 0 N in R2, 1 N in UMI, 3 barcode not in whitelist (CB word keeps the raw barcode)
Chunk index         {offset, bytes, n_pairs} per chunk; Footer {magic, n_chunks, index_offset, crc32c(index)}
```

Design points and why:

- **Batch layout = M8 layout.** Decoded sections of consecutive chunks land contiguously, so a batch of B chunks is exactly the M8 SoA batch
  (CBUMI n x 8, R2 n x 24; quality slot dropped, 32 B/pair on the device instead of M8's 56).
- **Section frames are plain zstd frames** with content size, no dictionary and no zstd checksum: what `nvcompBatchedZstdDecompressAsync` takes.
  Our own CRC32C (SSE4.2 on CPU) covers the stored bytes so the host can verify without decoding (0.004 core-s per M pairs). nvCOMP 6 also ships a
  GPU CRC32 (`nvcomp/crc32.h`), not used here.
- **Chunk = 32,768 pairs** (R2 section 0.8 MB). Smaller chunks cost almost nothing in size (16K: +0.5%, 256K: -0.4% vs 32K) and give the GPU decoder
  many independent frames (section 5). A batch is one contiguous file range (one `pread`), so the reader needs no per-chunk I/O.
- **Per-section raw flag**: a section stays raw if compression does not help (auto) or by option. `R2 raw + CBUMI/EXC zstd` gives 27.6 B/pair on PBMC 1k
  with 0.09 core-s per M cheaper encode than zstd 1; not recommended (section 5).
- **Exceptions, not an N alphabet.** N bases are rare (170 K in 66.6 M PBMC 1k pairs, 0.86 M in 121.8 M GSM6470208 pairs), so bases are 2-bit and N positions are a list. The
  barcode-not-in-whitelist flag is also an entry (5.1% and 18.8% of pairs). An earlier draft also listed N positions inside unknown barcodes: in
  GSM6470208 the unknown barcodes are all N (17 entries per unknown pair, 12.8 B/pair of raw exception data; inferred from the entry count); dropped (kind 2 reserved). A per-chunk bitmap would shrink this further
  (0.5 B/pair compressed for GSM6470208); left for a later version.
- **Limits (encoder refuses or drops, never truncates):** R2 > 127 nt (7-bit length) -> exit 3; `R1 != barcode + UMI` bases, deduplicated or tier-2
  `.1fq`, no barcode dictionary -> exit 4. Reads with R1 or R2 length 0 or R2 > 1024 are dropped and counted (`n_dropped`), as `singlet decode`
  does. The R2 slot is 24 B up to 96 nt and grows by 8 B steps (measured only at 90-91 nt).

## 4. CPU-node transform (measured)

`enc_1fg` (`bench/gpu/ingest/enc_1fg.cpp`): N worker threads each take one `.1fq` block (500 K reads), decompress it, parse straight into the layout
(no intermediate byte-numeric form), cut it into 32K-pair chunks, compress the three sections, CRC them, and an ordered writer appends. Output goes to
node-local `/tmp`. Source read from the shared filesystem with the page cache dropped. The 1-thread and 4-thread outputs are byte-identical (checked with `cmp` in job 789859).

Core-s per M pairs, 1 thread (job 789930 for PBMC 1k, 789904 for GSM6470208; the zstd -3 row from the level-ladder jobs 789931/789932, 4 threads):

| sample, variant | out B/pair | out / `.1fq` | total core-s per M | read + decompress `.1fq` | parse | zstd compress | copy + CRC |
|---|---|---|---|---|---|---|---|
| PBMC 1k (66.6 M pairs, `.1fq` 27.6 B/pair): zstd 1 | 22.94 | 0.83 | 0.308 | 0.106 | 0.024 | 0.136 | 0.008 |
| zstd 3 (4 threads) | 20.81 | 0.75 | 0.422 | 0.109 | 0.025 | 0.248 | 0.007 |
| zstd -3 (4 threads) | 24.42 | 0.89 | 0.222 | 0.110 | 0.022 | 0.060 | 0.008 |
| R2 raw, rest zstd 1 | 27.60 | 1.00 | 0.218 | 0.106 | 0.024 | 0.035 | 0.010 |
| raw | 32.33 | 1.17 | 0.193 | 0.106 | 0.024 | 0 | 0.011 |
| GSM6470208 (121.8 M pairs, `.1fq` 14.4 B/pair): zstd 1 | 18.76 | 1.30 | 0.242 | 0.052 | 0.026 | 0.129 | 0.006 |
| zstd 3 (4 threads) | 16.91 | 1.17 | 0.334 | 0.054 | 0.028 | 0.218 | 0.006 |
| raw | 32.90 | 2.28 | 0.145 | 0.052 | 0.026 | 0 | 0.011 |

- Rate: 1 thread 3.5 (PBMC) / 4.4 (GSM6470208) M pairs/s = 78 MB/s of `.1fq` in; 4 threads 10.9 / 13.4; 8 threads 17.3 / 22.0 (about 400 MB/s in). Repeats of the same configuration vary by up to 10% (zstd 3, 4 threads: 0.387, 0.416, 0.422 core-s per M on PBMC 1k in three jobs).
  Compression is 44% of the cost at zstd 1 and 59-65% at zstd 3. Chunk size does not matter (16K-256K: same cost).
- **Where the bytes go** (zstd 1, B/pair; raw in brackets): CBUMI 3.1-3.4 (8), R2 15.0-19.3 (24), EXC 0.14-0.52 (0.22-0.78). The CB is almost free because the
  `.1fq` is barcode-sorted within each 500 K-read block; the UMI (24 bits) is incompressible. **The barcode sort matters, measured** (same transform, pairs of
  every block in random order; jobs 789944, 789945; B/pair, CBUMI + R2 sections in brackets):

  | | zstd 1 sorted | zstd 1 shuffled | zstd 3 sorted | zstd 3 shuffled |
  |---|---|---|---|---|
  | PBMC 1k | 22.94 (3.40 + 19.33) | 28.66 (7.55 + 20.90) | 20.81 (3.39 + 17.20) | 24.27 (5.65 + 18.40) |
  | GSM6470208 | 18.76 (3.14 + 15.03) | 25.71 (6.88 + 18.23) | 16.91 (3.23 + 13.09) | 22.08 (6.27 + 15.22) |

  Unsorted input costs +17-37% bytes (and 0.04-0.1 more encode core-s per M). **A fetcher that writes `.1fg` directly must sort each block by barcode**
  (the `.1fq` writer's sort cost 17 s for 61 M spots = 0.28 core-s per M, job 789920) or accept the larger files.
- GSM6470208's `.1fq` is smaller than any `.1fg`: it was written with zstd level 4 over 500 K-read column blocks, and this sample's R2 is highly
  repetitive (R2 15.0 B at zstd 1, 13.1 at zstd 3, 12.6 at zstd 5 vs 24 raw). `.1fg` is 17% larger at zstd 3 and 30% at zstd 1; on PBMC 1k it is
  25% smaller at zstd 3.
- **Level choice.** In the real CPU-node job the source decode costs 2.24 (local) to 5.9 (VDB streaming) core-s per M spots, so the transform
  without the `.1fq` read is 0.15-0.27 core-s per M at zstd 1-3: 3-12% of the node's work. Recommendation: **zstd 3** (-9-10% bytes vs zstd 1 for
  +0.1 core-s per M); zstd 5 gains another 3-5% for +0.25 and is not worth it.
- **Whole CPU-node stage on one public run** (SRR21102188, 60.68 M spots, 8 cores, job 789934): fetch Lite 16.1 s -> `singlet download` local 52.7 s
  (136 core-s) -> `enc_1fg` zstd 3, 4 threads, 6.4 s (22 core-s, of which 4 core-s re-read and decompress the `.1fq`) -> 1.066 GB `.1fg` (17.56 B/pair vs `.1fq`
  15.54) -> upload dry run (16 parts). ~75 s wall and ~160 core-s for the run, 2.7 core-s per M pairs. Writing `.1fg` inside the fetcher instead of
  going through `.1fq` would drop the `.1fq` writer (35 s of the 53 s; not measured).

## 5. GPU node (measured)

`read_1fg` (`bench/gpu/ingest/read_1fg.cu`): R reader threads `pread` a batch (B consecutive chunks = one file range) into a pinned ring slot, optionally
CRC32C it; the GPU thread copies the compressed bytes H2D, builds the frame lists (3B entries) and launches
`nvcompBatchedZstdDecompressAsync` with output pointers straight into the batch SoA arrays (raw sections: device-to-device copy), then a consume
kernel reads every decoded byte. nvCOMP 6.0.0.12 comes from the pip wheel `nvidia-libnvcomp-cu12` (C headers + `libnvcomp.so.6` copied to
`$W/jobs/w11/ingest/nvcomp`); it runs on the CUDA 12.4.1 image with driver 610 as is. No custom codec was needed.

**GPU decode, data resident on the device** (median of 30 repeats, 2 batches; `status_ok` and output sizes checked for every frame):

| frames | pairs per batch | PBMC 1k M pairs/s (GB/s decoded) | GSM6470208 M pairs/s |
|---|---|---|---|
| 256K-pair chunks, 16 per batch | 4.2 M | 56 (1.8) | 58 |
| 64K, 64 per batch | 4.2 M | 205 (6.6) | 193 |
| 32K, 128 per batch (default) | 4.2 M | 320 (10.3) | 303 |
| 16K, 256 per batch | 4.2 M | 373 (12.0) | 408 |
| 32K, 256 per batch | 8.4 M | **600 (19.3)** | 576 |
| 32K zstd 3, 128 per batch | 4.2 M | 276 (8.9) | 258 |
| 32K R2 raw, 128 per batch | 4.2 M | 395 (12.7) | 315 |
| 32K raw (copies only) | 4.2 M | 3590 | 4350 |

The rate is proportional to the number of frames in the batch until the 132 SMs are full (an early test with 8 x 256K frames gave 28 M pairs/s).
At the 14-18 M pairs/s a GPU must absorb (exploration section 12.11), decode takes 3-6% of one GPU (zstd 1, 128-256 frames per batch).

**Host cost and end-to-end rate into GPU memory** (cold node-local NVMe, `fadvise DONTNEED` before each run; 1 / 2 / 3 reader threads; M pairs/s;
host core-s per M pairs of the whole process, `getrusage`; batch = 4.2 M pairs):

| file | B/pair | PBMC 1k M/s (1/2/3 threads) | host core-s per M | GSM6470208 M/s | host core-s per M |
|---|---|---|---|---|---|
| zstd 1, 32K | 22.9 / 18.8 | 42.9 / 49.0 / 81.2 | 0.0132-0.0142 | 52.6 / 60.1 / 100.0 | 0.0108-0.0115 |
| zstd 3, 32K | 20.8 / 16.9 | 47.1 / 53.5 / 87.5 | 0.0121-0.0129 | 58.1 / 67.4 / 107.3 | 0.0097-0.0105 |
| R2 raw | 27.6 / 27.7 | 35.6 / 40.9 / 68.4 | 0.0160-0.0171 | 35.6 / 40.8 / 70.2 | 0.0161-0.0172 |
| raw | 32.3 / 32.9 | 30.7 / 35.1 / 61.0 | 0.0187-0.0201 | 30.0 / 34.9 / 61.1 | 0.0192-0.0204 |
| zstd 1 + CRC32C check, 2 threads | | 44.9 | 0.0176 | 56.0 | 0.0145 |
| raw + CRC32C check, 2 threads | | 32.4 | 0.0254 | 32.1 | 0.0253 |
| file read + H2D only (no decode), zstd 1, 1 thread | | 42.8 | 0.0131 | 52.8 | 0.0104 |
| **M8 `.1fq` baseline** (decode + pack + H2D, 3 threads) | 27.6 / 14.4 | 21.6 | 0.126-0.136 | 40.7 | 0.067-0.073 |

- The reader's host work is `pread` + page-locked copy submission: 0.010-0.014 core-s per M pairs, **6-10x less than the `.1fq` reader** (0.067-0.136); the GPU thread uses 0.0003-0.0012 core-s per M. At 27-36 M pairs/s per node that is **0.3-0.5 cores for
  ingest instead of 2.3-4.7** (exploration section 12.10/12.11 numbers, at 27 M pairs/s), CRC included 0.5.
- The e2e rate here is bounded by the disk (1 thread = 1 GB/s of file reads; 3 threads = 1.9 GB/s), not by decode or PCIe: compressed zstd files are read
  faster per pair, so the compressed variants win whenever storage, not the GPU, is the limit. PCIe (12.3 GB/s measured in M8) is not limiting:
  22.9 B/pair -> 540 M pairs/s link bound (380 for raw), shared by both GPUs of a node (6.2 GB/s each when both copy).
- Raw wins nothing on the host (0.019-0.020 vs 0.010-0.014 core-s per M) and costs 40-75% more bytes; its only advantage is no GPU decode work (a 2 ms
  device copy per batch) and a simpler reader. Use it for node-local staging if GPU time is the scarce resource and the disk is idle.
- Each configuration ran once per sample (reader runs last 0.6-2.5 s, so ramp-up and tail show). The shared filesystem was not used as the reader's source here (M8 found no penalty for reading it directly).

## 6. Identity with the `.1fq` decode, and the CPU engine

- **GPU decode vs `singlet decode`** (`read_1fg --verify`): after GPU decode, every batch is copied back and each pair is rebuilt as text (R1 = UMI prefix +
  barcode + UMI suffix at the protocol's offset, unknown barcode and exceptions as N; R2 from the 2-bit slot and the stored length) and compared, line by
  line, with the FASTQ that `singlet-w3 decode` writes from the `.1fq`: record names `@r<N>`, R1 and R2 sequence lines, and the quality line (the files
  have no qualities: `F` x length, compared by length). PBMC 1k: **66,601,887 pairs, 0 mismatches**, 3,501,977 pairs with exceptions; GSM6470208:
  **121,774,155 pairs, 0 mismatches**, 23,603,076 with exceptions; for `zstd 1`, `raw` and `R2 raw` files each (jobs 789930, 789904). (`singlet decode`
  writes all of R1 of a batch before R2, so the check decodes to job-local files first; a FIFO version deadlocks.)
- **CPU engine bytes.** `dec_1fg` decodes a `.1fg` on the host to the byte-numeric R1/R2 that `lib1fq` hands the engine (A,C,G,T = 0-3, N = 4) and
  hashes every read; `lib1fq` decodes the source `.1fq`; the two hashes are equal for PBMC 1k (66.6 M pairs), GSM6470208 (121.8 M) and the freshly
  fetched SRR21102188 (60.7 M) (jobs 789931, 789932, 789934). Host decode cost, 1 thread, no hashing: `.1fg` 0.150 / 0.181 core-s per M pairs vs `.1fq`
  library 0.308 / 0.199 (PBMC / GSM6470208): equal or up to 2x cheaper.
- **What changes for the CPU engine.** A `lib1fq` reader class for `.1fg` that fills `DecodedBlock` (r1/r2 byte arrays and lengths) is the only
  required change; the engine and STAR input stay as they are. Differences from `.1fq` to carry deliberately: (1) no barcode dictionary, so
  `bc_indices` must be derived from the whitelist when needed (the CB word is explicit 2-bit); (2) unknown barcodes keep the raw barcode in the CB word
  with a flag (the `.1fq` library decode returns N for them: both are available, and the identity tests use the N form); (3) no dup counts, tier-2
  reference compression, I2 stream, trim lengths or qualities (the atlas files use none of them; the encoder refuses deduplicated and tier-2 sources);
  (4) records stay in source order (the `.1fq` from `singlet download` is barcode-sorted, the `.1fg` inherits it); (5) the `.1fg` header carries `protocol_id` and the read geometry (bc/UMI lengths, offset, R1 length), the same fields the engine takes from the `.1fq` header; the
  `.1fq` metadata block (STAR parameters, autodetect) was empty in the PBMC 1k file (job 789959; the other two were not checked), so carrying it is untested. `.1fq` remains the archival/CPU format; `.1fg` is the GPU-node transport.

## 7. Upload (dry-run interface only)

`bench/gpu/ingest/upload_dryrun.py FILE.1fg --accession SRR... --bucket NAME [--part-mib 64]` reads only the file header, chunk index and footer and prints
the manifest the real `singlet ingest upload` would act on; it makes no network call, uses no credentials, writes nothing but stdout/`--out`, and the
manifest says `"dry_run": true, "network_calls": 0`. Key scheme `ingest/v1/<accession>/<accession>.1fg`; **multipart parts end on chunk boundaries**
(64 MiB by default: 23 parts for the 1.53 GB PBMC 1k file, 16 for the 1.07 GB SRR21102188 file), so every part is a valid run of chunks and a failed part
is re-sent alone; checksum algorithm CRC32C per part (S3 `x-amz-checksum-crc32c`) on top of the per-section CRCs; the registered row is (accession,
key, bytes, pairs, n_chunks, index sha256). Example manifests: `bench/gpu/ingest/results_raw/manifest_dryrun_*.json`. Nothing was written to
Cloudflare in this study; the real uploader and its destination (scratch on the shared filesystem vs object storage) are W4 decisions.

## 8. How this fits W4 (download workstream)

W4 owns the fetcher. Proposed split, none of it requires changing W4's scope:

1. **Acquire (W4).** SDL lookup with alternates -> Lite (NCBI host) / full `.sra` (S3) / ENA fallback; 8 parallel ranged streams to job `/tmp`; verify
   size and md5 from the SDL answer; local decode with the existing encoder (`singlet download FILE.sra`). Drop VDB streaming: measured 6.7x less wall time
   including the fetch (8.8x for the decode alone) and 2.6x fewer core-s per M spots on comparable runs. Keep the `.1fq` as the archival output.
2. **Transform (new module, `libfg`, hand-over from this prototype).** `.1fq` -> `.1fg` (measured, 0.3-0.4 core-s per M pairs) first; later a direct
   writer in the encoder's block loop, keeping the per-block barcode sort (section 4: without it files are 17-37% larger), to skip the `.1fq` writer.
3. **Upload (W4).** The dry-run manifest is the contract: object key, parts on chunk boundaries, CRC32C per part, registry row.
4. **Consume (W11).** `read_1fg` (host reader + nvCOMP) replaces `.1fq` decode + pack in the M8 pipeline: 0.010-0.014 core-s per M pairs instead of 0.085-0.175 (M8, exploration section 12.10; 0.067-0.136 measured here on the same two samples).

Budget at campaign scale (extrapolation from the measured per-M figures, not a measurement; 20.6 T pairs from exploration section 11): the CPU-node stage at
2.7 core-s per M pairs is ~15,000 core-hours (33,800 at today's VDB streaming rate of 5.9); download volume with Lite at ~36 B/pair is ~740 TB.

## 9. Assumptions and decisions made without asking

- Samples: PBMC 1k v3 (3' v3, variable-length trimmed R2, 91 nt max) and GSM6470208 (5' v2, fixed 90 nt), both real `.1fq` from the Clipper data tree; the freshly fetched
  SRR21102188 for the CPU-node demonstration. Two samples, one node type, each configuration run once (the node-to-node spread was not measured).
- Chunk 32,768 pairs, batch 128-256 chunks, zstd 3 recommended (levels 1 and 3 and others measured), CRC32C of the stored payload (not of the decoded data), no zstd
  frame checksum (nvCOMP).
- Length field 7 bits, UMI <= 12 nt, CB <= 16 nt (as M8): protocols outside these are refused by the encoder, not truncated.
- The format keeps the source order and the whitelist-membership flag of `.1fq`; whitelist matching on the GPU would make the flag redundant but is not
  designed here.
- GPUDirect Storage, nvCOMP's GPU CRC32, shared-filesystem reads for the reader and both GPUs of a node at once were not tested. Unsorted input was tested for size only (transform), not through the reader.
- No Cloudflare write, no Anvil, no `sync.sh`; reads only in job `/tmp`; the shared bridge was used as found.

## 10. Reproduce

`bench/gpu/ingest/README.md` lists jobs, scripts and result files. In short: `jobs/build.sbatch` (g++/nvcc in the CUDA 12.4.1 image), `jobs/run_sample.sbatch`
(transform variants, reader, GPU decode, identity, M8 baseline), `jobs/cpu_extra.sbatch` (level ladder, CPU compatibility, dry run), `jobs/download.sbatch`,
`jobs/lite_compare.sbatch`, `jobs/lite_vs_full_cpu.sbatch`, `jobs/cpu_node_pipeline.sbatch`; `summarize.py` builds `bench/gpu/results/ingest.json`.
