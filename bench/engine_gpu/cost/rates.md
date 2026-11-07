# Cost rates for the GPU vs CPU alignment comparison

Accessed 2026-10-27. Full detail, sources and caveats in `rates.json`. Excludes tax, storage, egress.

## Recommended central values

| Quantity | Central | Range | Basis |
|---|---|---|---|
| $/H100-GPU-hour, on-demand | **$3.99** | 2.59 (RunPod NVL community) to 6.88 (AWS) | Lambda 8x H100 SXM, primary page. Hyperscalers are 1.7x to 3x higher |
| $/physical-core-hour, on-demand | **$0.05** | 0.039 to 0.089 | No-SMT HPC/compute instances (AWS c7a/hpc7a, Azure HBv4) are 0.041 to 0.051 |
| W per GPU node (2x H100 NVL, wall) | **~1,400 (estimate)** | 1,200 to 1,600 | 2x400 W GPU plus dual-socket host. No measured figure found |
| W per CPU node (dual-socket 64+ core, wall) | **~650** | 630 to 830 | SPECpower, dual EPYC 9654 at 100% load |
| Electricity | **14.47 c/kWh commercial** (9.64 industrial) | | EIA, US, Aug 2026 |
| PUE | **1.56** | | Uptime Institute 2024 global average (secondary summary) |

## Cloud GPU, on-demand (per GPU-hour)

| Provider / instance | Region | $/GPU-h | Instance $/h | vCPU, RAM per GPU | Source |
|---|---|---|---|---|---|
| Lambda 8x H100 SXM | US | 3.99 | 31.92 | 26, 225 GiB | lambda.ai/pricing (primary) |
| RunPod H100 NVL 94 GB (secure) | n/s | 3.19 (community 2.59) | per GPU | 16, 94 GiB | runpod.io/pricing (primary) |
| CoreWeave HGX H100 | N. America | 6.16 (spot 2.46) | 49.24 | 16, 256 GB | coreweave.com/pricing (primary) |
| AWS p5.48xlarge | us-east-1 | 6.88 (1-yr SP 5.43, 3-yr 2.97, spot ~3.55) | 55.04 | 24, 256 GiB | Holori calculator (aggregator) |
| GCP a3-highgpu-8g | us-central1 | 11.06 | 88.49 | 26, 234 GiB | Holori and Google localized page |
| Azure ND96isr H100 v5 | eastus | 12.29 (spot 2.27) | 98.32 | 12, 237.5 GiB | Azure Retail Prices API (primary) |

H100 NVL is sold specifically only by RunPod among those checked. AWS p5e/p5en are H200, not H100.

## Cloud CPU, on-demand

| Instance | Region | $/h | vCPU : cores | $/vCPU-h | $/core-h | RAM | Source |
|---|---|---|---|---|---|---|---|
| AWS c7i.48xlarge | us-east-1 | 8.568 | 192 : 96 (SMT) | 0.0446 | 0.0893 | 384 GiB | Vantage (aggregator) |
| AWS c7a.48xlarge | us-east-1 | 9.853 | 192 : 192 (no SMT) | 0.0513 | 0.0513 | 384 GiB | Vantage (aggregator) |
| AWS hpc7a.96xlarge | unconfirmed | 9.079 | 192 : 192 (no SMT) | 0.0473 | 0.0473 | 768 GiB | Vantage (unverified, may be stale) |
| Azure HB176rs_v4 | eastus | 7.20 | 176 : 176 | 0.0409 | 0.0409 | 704 GiB | Azure API (primary) |
| Azure HX176rs | eastus | 8.64 | 176 : 176 | 0.0491 | 0.0491 | 1408 GiB | Azure API (primary) |
| CoreWeave Genoa 9654 | N. America | 7.54 | 384 : 192 (SMT) | 0.0196 | 0.0393 | n/f | coreweave.com/pricing (primary) |
| GCP c3 / c3d / h3 | | | | | null | | Not found (page truncated) |

## NSF ACCESS (documentation only)

- Anvil CPU: 1 ACCESS credit = 1 core-hour (since 2024-11-19), operations.access-ci.org/node/788.
- Anvil GPU (A100): about 67 credits per GPU-hour, derived from Purdue's example (400k credits ~ 6,000 GPU-h). Not a published rate.
- Delta GPU: 1 SU = 1 A100 GPU-hour; credit rate not found. Sources disagree on the H200 factor (1x vs 3x). Delta has no H100.
- The live exchange calculator did not render rates, so these are gaps.

## Third-party claims (context only)

- Terra blog on Parabricks: 30x WGS under $2.50 on GPU vs $5 on CPU GATK. DNA germline, not RNA-seq.
- NVIDIA claims up to 50% compute-cost saving for WGS.
- No cost-per-sample for STAR/RNA-seq found.

## Gaps

GCP CPU prices, measured 2x H100 NVL wall power, ACCESS Anvil GPU and Delta credit rates, AWS p5 price from AWS's own page (aggregator used), hpc7a region and price, PUE verified against the Uptime report.
