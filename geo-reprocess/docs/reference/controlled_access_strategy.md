# Controlled-Access Data Strategy for Single-Cell Transcriptomics

**Date:** June 2025
**Status:** Comprehensive Analysis
**Purpose:** Inventory all controlled-access repositories containing single-cell RNA-seq data, assess regulatory requirements, define compute architecture, and establish that NMF model outputs (W matrix and factor summary statistics) are non-identifying.

---

## Table of Contents

1. [Context: The Public Data Baseline](#1-context-the-public-data-baseline)
2. [Controlled-Access Repository Inventory](#2-controlled-access-repository-inventory)
3. [Estimated Controlled-Access Data Volume](#3-estimated-controlled-access-data-volume)
4. [Systematic Discovery Methods](#4-systematic-discovery-methods)
5. [Access Application Process](#5-access-application-process)
6. [Regulatory & PHI Landscape](#6-regulatory--phi-landscape)
7. [Compute Architecture Decision Tree](#7-compute-architecture-decision-tree)
8. [Data Movement & Transfer Constraints](#8-data-movement--transfer-constraints)
9. [NMF PHI-Safe Architecture](#9-nmf-phi-safe-architecture)
10. [Actionable Roadmap](#10-actionable-roadmap)

**Appendices:** A (Discovery Queries), B (Cost Model), C (Key Contacts & Resources)

---

## 1. Context: The Public Data Baseline

Our current pipeline processes **publicly available** single-cell data from GEO/SRA:

| Metric | Value |
|--------|-------|
| Total human SRX samples | 206,330 |
| Total GSEs | 4,597 |
| Already processed (quant/) | 63,671 samples (2,062 GSEs) |
| Remaining to process | 142,659 samples (2,535 GSEs) |
| Pipeline | simpleaf (STAR + salmon) -> QC -> kraken2 -> cleanup |
| Compute | TAMU Launch/ACES via ACCESS BIO260157 (200K credits) |

**CZ CELLxGENE Census** (the largest curated public collection): 149.3M unique cells across 2,083 datasets from 1,163 collections across 459 unique tissues.

**Human Cell Atlas Data Portal**: 70.5M cells, 11,200 donors, 528 projects.

**The gap:** A substantial fraction of human single-cell data -- particularly from clinical studies, rare disease cohorts, pediatric populations, and pharma-sponsored trials -- is deposited in **controlled-access** repositories. These datasets are invisible to GEO/SRA queries and absent from CELLxGENE. We estimate this represents **1,100-3,000 additional studies** containing **235M-1.1B cells**, potentially doubling or tripling the public data available for training.

---

## 2. Controlled-Access Repository Inventory

### 2.1 dbGaP (Database of Genotypes and Phenotypes)

**Operator:** NCBI / NIH (United States)
**URL:** https://www.ncbi.nlm.nih.gov/gap/

| Metric | Value |
|--------|-------|
| Total studies | ~3,100 |
| Total participants | ~5.3 million |
| Registered users | ~44,800 |
| Authorized requests approved | ~12,800 |
| Single-cell relevant studies (est.) | 300-800 |

**Access model:**
- **Open-access:** Study summaries, phenotype variable lists, document sets (protocols, questionnaires), aggregate genotype data, select individual-level data
- **Controlled-access:** Individual-level phenotype data, genotype data, sequence data (FASTQ/BAM/CRAM), expression data
- **Requirements:** eRA Commons account, institutional Signing Official (SO), Data Use Certification (DUC) agreement, IRB/ethics approval or exemption
- **Timeline:** 2-8 weeks for initial approval; renewals faster
- **Download:** SRA Toolkit (prefetch, fasterq-dump), Globus, cloud delivery (AWS/GCP via AnVIL or SRA cloud)

**Key policies:**
- Data must be stored on institutional or approved systems with encryption at rest
- No redistribution; all users on a project must be listed on the DUC
- Annual renewal required; data must be destroyed when project closes
- Cloud use explicitly permitted (AnVIL, Terra, Seven Bridges)
- **University HPC explicitly permitted** with IT security plan meeting NIH Security Best Practices (SBP)

### 2.2 EGA (European Genome-Phenome Archive)

**Operator:** EMBL-EBI + Centre for Genomic Regulation (CRG), Barcelona
**URL:** https://ega-archive.org/

| Metric | Value |
|--------|-------|
| Total studies | ~4,500+ |
| Contributing institutions | ~1,000+ |
| Single-cell relevant studies (est.) | 400-1,200 |

**Access model:**
- Metadata (study/sample/experiment descriptions) is openly searchable
- Data access is governed **per-dataset** by a Data Access Committee (DAC) appointed by the data submitter
- No single application covers all EGA data -- each DAC sets its own terms
- **Data Access Agreement (DAA)** signed between requester institution and DAC

**Download:** pyega3 (Python client), Globus, CRAM/BAM streaming via htsget API

**Key policies:**
- DAC approval timelines vary widely (1 week to 6+ months)
- Some DACs require ethics approval from requester's institution
- Data may have geographic restrictions (e.g., "EU only" or "within NHS")
- University HPC permitted unless DAA specifies cloud-only or geographic restriction
- UK Biobank data (>500K participants, some sc-RNA-seq) is accessed through EGA + UK Biobank portal

### 2.3 GSA-Human (Genome Sequence Archive for Human Data)

**Operator:** National Genomics Data Center (NGDC), Beijing Institute of Genomics, Chinese Academy of Sciences
**URL:** https://ngdc.cncb.ac.cn/gsa-human/

| Metric | Value |
|--------|-------|
| Total studies | 10,919 |
| Total individuals | 1.08 million |
| Total runs | 2.01 million |
| Single-cell relevant studies (est.) | 500-2,000 |

**Access model:**
- Open-access metadata browsing
- Controlled data requires application through NGDC Data Access Committee
- Chinese data sovereignty regulations may restrict bulk export of individual-level data outside China
- Some datasets have been dual-deposited in EGA or SRA (check cross-references)

**Key considerations:**
- Largest single repository by study count for human genomics data
- Transfer speeds from Chinese data centers to US may be slow
- Export restrictions may require in-country compute or use of dual-deposited copies
- Growing rapidly -- doubles every ~2 years

### 2.4 JGAS (Japanese Genotype-phenotype Archive)

**Operator:** NBDC (National Bioscience Database Center) + DDBJ, Japan
**URL:** https://humandbs.dbcls.jp/en/

| Metric | Value |
|--------|-------|
| Total projects (controlled) | ~500+ |
| Access tiers | Type I (unrestricted) and Type II (controlled) |
| Single-cell relevant (est.) | 50-150 |

**Access model:**
- Type I: Data available after user registration agreement
- Type II: Requires institutional application, ethics review, data handling plan
- Off-premise server policy exists (can use non-DDBJ systems with approval)
- Application form in Japanese (English documentation improving)

### 2.5 Federated / Disease-Specific Portals

These portals aggregate controlled-access data, often with their own access layers on top of dbGaP:

| Portal | Focus | Data Scale | Access Via |
|--------|-------|-----------|------------|
| **HTAN** (Human Tumor Atlas Network) | Cancer atlases | 26+ atlases, ~2M cells | dbGaP + Synapse + ISB-CGC |
| **HuBMAP** | Healthy human tissue maps | 7,000+ tissue blocks | dbGaP + Globus |
| **KPMP** (Kidney Precision Medicine) | Kidney disease | ~200K cells | dbGaP + KPMP data portal |
| **LungMAP** | Lung development/disease | ~500K cells | dbGaP + LungMAP portal |
| **GDC** (Genomic Data Commons) | Cancer (TCGA, TARGET) | 84,000+ cases | dbGaP + GDC API |
| **ENCODE** | Functional elements | Select sc-RNA-seq | Open + dbGaP |
| **Brain Initiative** (BICCN/BICAN) | Brain cell atlases | ~10M cells | NeMO Archive + dbGaP |
| **GTEx** | Multi-tissue expression | 948 donors, some sc-RNA-seq | dbGaP + AnVIL |

### 2.6 Cloud Compute Platforms (Not Repositories)

These platforms **host** controlled-access data and provide compute co-located with storage:

| Platform | Operator | Backend | Key Data |
|----------|----------|---------|----------|
| **AnVIL** | NHGRI / Broad | Terra (Google Cloud) | GTEx, CCDG, CMG, AnVIL datasets |
| **BioData Catalyst** | NHLBI | Terra + Seven Bridges + Gen3 | TOPMed, COPDGene, heart/lung/blood |
| **Cancer Genomics Cloud** | NCI | Seven Bridges (AWS) | TCGA, TARGET, CPTAC |
| **Kids First DRC** | NIH Common Fund | CAVATICA (Seven Bridges) | Pediatric cancers, structural defects |
| **UK Biobank RAP** | UK Biobank | DNAnexus (AWS) | 500K participants, WGS + sc-RNA-seq |

---

## 3. Estimated Controlled-Access Data Volume

### 3.1 Bottom-Up Estimate

| Source | Est. Studies with sc-RNA-seq | Est. Cells (M) | Confidence |
|--------|------------------------------|-----------------|------------|
| dbGaP (direct) | 300-800 | 100-400 | Medium |
| EGA (direct) | 400-1,200 | 150-500 | Medium |
| GSA-Human | 500-2,000 | 100-400 | Low (export uncertain) |
| JGAS | 50-150 | 10-50 | Low |
| Disease portals (HTAN, HuBMAP, etc.) | 100-300 | 25-100 | Medium-High |
| UK Biobank | 1 (pilot + expansion) | 5-50 | Medium |
| **Total** | **1,100-3,000** | **235M-1.1B** | -- |

### 3.2 Cross-Reference with Public Data

| Dataset | Cells | Relationship to Controlled |
|---------|-------|---------------------------|
| Our GEO/SRA pipeline | ~206K samples | No overlap (public only) |
| CELLxGENE Census | 149.3M cells | Mostly public, some controlled re-deposited |
| HCA Data Portal | 70.5M cells | Mix of public + controlled |

**Key insight:** Many controlled-access studies also deposit **processed** count matrices in GEO or CELLxGENE, while keeping raw FASTQ in dbGaP/EGA. For NMF training, we need consistent reprocessing from FASTQ (uniform pipeline = simpleaf), so we must access the raw controlled data even if processed matrices are publicly available.

### 3.3 Why This Matters for CPM Training

The Conditional Program Model architecture requires:
- **W matrix** (genes x factors): Learned from NMF on the full atlas. More diverse training data leads to a better program dictionary
- **Conditional encoder** f_theta: Maps metadata to program activities. Clinical metadata (disease stage, treatment response, drug exposure) is **almost exclusively** in controlled-access datasets
- **Scale target:** k = 1,024-2,048 programs require ~500M-1B diverse cells for stable estimation

Controlled-access data provides exactly what public data lacks: **clinical depth** (longitudinal samples, treatment arms, matched controls) and **phenotypic richness** (detailed disease staging, drug responses, survival outcomes).

---

## 4. Systematic Discovery Methods

### 4.1 dbGaP Discovery

**Primary approach:** Query dbGaP Advanced Search for studies containing single-cell RNA-seq:

```
# dbGaP Advanced Search URL parameters
# Molecular Data Type: "RNA Sequencing"
# Study Disease/Focus: (iterate by disease)
# Study Type: filter for "Case-Control", "Cohort", "Longitudinal"
```

**Programmatic approach:** dbGaP has a RESTful API for study metadata:
```python
import requests

# Search dbGaP studies via Entrez
base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
params = {
    "db": "gap",
    "term": "single cell RNA-seq[Molecular Data Type]",
    "retmax": 5000,
    "retmode": "json"
}
resp = requests.get(base, params=params)
study_ids = resp.json()["esearchresult"]["idlist"]
```

**Cross-reference with SRA:** Many dbGaP studies register their runs in SRA with restricted access flags. Query SRA for access_type:controlled:
```python
# SRA query for controlled-access single-cell studies
# Uses the SRA Run Selector or Entrez:
term = '("single cell"[Selection]) AND "controlled access"'
```

### 4.2 EGA Discovery

**Primary approach:** EGA search API + Dataset Browser:
```python
# EGA metadata search
import requests

url = "https://ega-archive.org/metadata/v2/datasets"
# Filter by technology type, experiment type
# Cross-reference with publications via DOI/PMID links
```

**Publication mining:** Many EGA datasets are only discoverable through the associated publication. Strategy:
1. Query PubMed for single-cell papers citing EGA accessions (EGAD*)
2. Parse supplementary materials for EGA dataset IDs
3. Maintain a lookup table of PMID -> EGAD -> DAC contact

### 4.3 GSA-Human Discovery

```python
# GSA-Human search API
url = "https://ngdc.cncb.ac.cn/gsa-human/browse"
# Filter by: Experiment Type = "RNA-Seq", Library Strategy = "scRNA-seq"
# Cross-check for dual deposits in SRA/EGA via BioProject/BioSample links
```

### 4.4 Unified Discovery Pipeline (Recommended)

Build a **controlled-access catalog** mirroring our public stage7_multimodal_catalog.parquet:

1. **Harvest metadata** from dbGaP, EGA, GSA-Human, JGAS APIs
2. **Link to SRA** via BioProject/BioSample cross-references
3. **Filter for single-cell** using same criteria as public pipeline (library strategy, platform, selection method)
4. **Deduplicate** across repositories (same study in dbGaP + EGA)
5. **Annotate** with: access tier, DAC contact, estimated approval timeline, geographic restrictions
6. **Priority score** based on: cell count estimate, clinical metadata richness, assay type, disease relevance

Output: controlled_access_catalog.parquet with columns:
```
study_id | repository | accession | est_cells | disease_focus | assay |
access_tier | dac_contact | geo_restriction | dual_deposit_public_id |
priority_score | application_status
```

---

## 5. Access Application Process

### 5.1 dbGaP Application Workflow

```
1. PI creates eRA Commons account (if not existing)
   -- Requires NIH-recognized institution

2. Institutional Signing Official (SO) registered
   -- University Office of Research / Sponsored Programs

3. Create Data Access Request (DAR) on dbGaP
   |-- Select datasets (can batch multiple phs* studies per request)
   |-- Write Research Use Statement (RUS) -- 1-2 paragraphs
   |-- List all project personnel with eRA Commons IDs
   '-- Upload IRB approval or exemption letter

4. SO reviews and co-signs the DAR
   -- Certifies institutional security measures

5. NIH Data Access Committee (DAC) reviews
   |-- Checks RUS against data use limitations (DULs)
   |-- Typical turnaround: 2-4 weeks (first request)
   '-- Automated for datasets with General Research Use (GRU) consent

6. Access granted -> SRA Toolkit / cloud delivery
   -- Annual renewal required; 3-year max, then re-apply
```

**Batch strategy:** dbGaP allows **multiple phs studies in a single DAR** if they share the same consent group. Group requests by DUL compatibility to minimize applications.

### 5.2 EGA Application Workflow

```
1. Register EGA account
   -- Institutional email required

2. Identify dataset(s) of interest (EGAD*)
   -- Each EGAD has a linked Data Access Committee (DAC)

3. Contact DAC (often the PI of the original study)
   |-- Submit Data Access Agreement (DAA)
   |-- Describe research purpose
   |-- Some DACs use standard forms; others have custom processes
   '-- May require ethics approval from your institution

4. DAC approval (timeline varies: 1 week to 6+ months)
   -- Expedited if: same field, same institution as depositor, or standard form

5. EGA grants download credentials
   -- pyega3 or Globus transfer
```

**Key difference from dbGaP:** No centralized committee -- each dataset has its own DAC with its own rules, timelines, and requirements. This makes EGA applications inherently slower to scale.

### 5.3 Batch Application Strategy

Given 1,100-3,000 potential studies across repositories:

| Phase | Action | Studies | Timeline |
|-------|--------|---------|----------|
| 1 | dbGaP GRU-consent studies (auto-approve batch) | ~100-300 | 2-4 weeks |
| 2 | dbGaP disease-specific (manual DAC review) | ~200-500 | 4-8 weeks |
| 3 | EGA high-priority (large studies, responsive DACs) | ~100-200 | 2-4 months |
| 4 | EGA long-tail + HTAN/HuBMAP portal requests | ~200-500 | 4-12 months |
| 5 | GSA-Human (assess export feasibility) | TBD | 6+ months |

**Parallelization:** Applications to different repositories and DACs are independent -- submit all Phase 1-3 simultaneously while building the controlled-access catalog.

---

## 6. Regulatory & PHI Landscape

### 6.1 What Constitutes PHI in Genomics?

Under HIPAA, Protected Health Information (PHI) includes 18 identifiers. In single-cell genomics context:

| Data Type | PHI Status | Rationale |
|-----------|-----------|-----------|
| Raw FASTQ/BAM | **Yes (potential)** | Contains germline variants that are unique identifiers |
| Count matrix (genes x cells) | **Gray area** | No direct identifiers, but gene expression profiles may be fingerprinted with sufficient auxiliary data |
| Clinical metadata (age, sex, disease, treatment) | **Yes** when linked to individual | De-identified when not linkable to a specific person |
| Cell-level metadata (cell type, cluster) | **No** | Derived computationally, not linked to individual identity |
| NMF W matrix (genes x factors) | **No** | Population-level gene program dictionary; no individual information |
| NMF H matrix (factors x cells) | **Controlled** | Linked to individual cells, which may link to donors |
| Factor summary statistics (mean, variance per factor per condition) | **No** | Aggregated across cells/donors; non-identifying |

### 6.2 NIH Genomic Data Sharing (GDS) Policy

The NIH GDS Policy (effective January 2023 update) governs all NIH-funded genomic data:

**Key provisions:**
- All human genomic data generated with NIH funds must be shared via approved repositories
- Data requiring controlled access must be submitted to dbGaP or equivalent
- Informed consent must address future research use and broad data sharing
- Institutions must have a Genomic Data Sharing Plan in grant applications

**Data Use Limitations (DULs):** Each dataset in dbGaP has coded consent restrictions:
- **GRU** (General Research Use): Broadest -- data can be used for any research
- **HMB** (Health/Medical/Biomedical): Research must be health-related
- **DS-{disease}** (Disease-Specific): Only for research on specified disease
- **POA** (Population Origins/Ancestry): Includes population structure research
- **NPU** (Not-for-Profit Use): Commercial use prohibited

**For our use case:** Training a general-purpose transcriptomics model (CPM) is most compatible with **GRU** and **HMB** consent codes. DS-coded datasets may require argument that a general model benefits the specified disease. NPU restricts commercial deployment.

### 6.3 NIH Security Best Practices for Controlled-Access Data

The NIH SBP document defines minimum security requirements for institutions handling controlled-access data:

| Requirement | Satisfied by University HPC? | Notes |
|-------------|------------------------------|-------|
| Encrypted storage at rest | Yes (LUKS, dm-crypt) | Most HPC centers offer encrypted scratch |
| Encrypted data in transit | Yes (SSH, TLS) | Standard HPC access patterns |
| Multi-factor authentication | Yes (Duo, etc.) | University identity systems |
| Access limited to approved users | Yes (Unix permissions, ACLs) | PI controls group membership |
| Audit logging | Yes (auditd, SLURM accounting) | HPC centers log all access |
| Data destruction upon project end | Yes (scrub + unlink) | Document procedure |
| Firewall / network segmentation | Depends | Some HPC centers have dedicated secure enclaves |
| Physical security | Yes | Data centers have badge access |
| Incident response plan | Yes | University IT security office |

**Conclusion: University HPC IS permitted** for dbGaP and most EGA data, provided the IT security plan documents compliance with all SBP requirements. This is explicitly supported in NIH guidance -- controlled-access data does NOT require cloud-only processing.

### 6.4 Regulatory Requirements by Repository

| Repository | US University HPC | NSF ACCESS | Cloud (AnVIL/Terra) | Geographic Restriction |
|-----------|-------------------|------------|---------------------|----------------------|
| dbGaP | Yes, with SBP plan | Case-by-case | Yes, preferred by NIH | None |
| EGA | Yes, unless DAA restricts | Unlikely for EU-restricted | Yes, via EGA-EBI cloud | Some datasets EU-only |
| GSA-Human | Export uncertain | No | No, may require China compute | Chinese sovereignty rules |
| JGAS | With off-premise approval | No | Maybe | Prefer Japan-based |
| HTAN/HuBMAP | Yes, via dbGaP DUC | Case-by-case | Yes | None |
| UK Biobank | No, RAP (DNAnexus) only | No | Yes, UK Biobank RAP only | UK cloud only |

### 6.5 NSF ACCESS Considerations

Our allocation (BIO260157) is on TAMU Launch/ACES. Using ACCESS resources for controlled-access data requires:

1. **Supplemental security plan** submitted to ACCESS and the data repository
2. **TAMU HPRC security team** approval for controlled data on Launch
3. **Dedicated project directory** with restricted ACLs (not shared scratch)
4. **Encryption at rest** -- verify TAMU Launch scratch filesystem supports this
5. **No data persistence** after allocation ends -- document destruction procedure

**Recommendation:** Process controlled-access data on **university-owned HPC** (not ACCESS) for simplicity. Use ACCESS credits for the public data pipeline where there are no security complications. If controlled-access volume is large enough to justify the overhead, negotiate a separate ACCESS allocation with security provisions.

---

## 7. Compute Architecture Decision Tree

### 7.1 Decision Framework

```
Is the data from dbGaP or US-based portal?
  |
  +-- YES --> Does your university HPC have an SBP-compliant security plan?
  |             |
  |             +-- YES --> Process on university HPC (cheapest, fastest)
  |             |
  |             +-- NO --> Use AnVIL/Terra (cloud, higher cost)
  |
  +-- NO --> Is the data from EGA?
               |
               +-- YES --> Does the DAA allow off-EU processing?
               |             |
               |             +-- YES --> University HPC (with DAA documentation)
               |             |
               |             +-- NO --> Use EGA Federated node or EU cloud
               |
               +-- NO --> GSA-Human / JGAS / other
                            |
                            +-- Check export restrictions per dataset
                            +-- May require in-country compute or dual-deposit fallback
```

### 7.2 University HPC (Recommended Primary)

**Advantages:**
- $0 marginal cost (existing allocation or departmental resources)
- Full control over pipeline (simpleaf, kraken2, custom NMF code)
- No cloud vendor lock-in
- Familiar SLURM workflow identical to public data pipeline
- No egress charges for moving count matrices out

**Requirements:**
- IT security plan documented and approved by data repository
- Encrypted scratch or dedicated secure partition
- Access restricted to DUC-listed personnel only
- Audit logging enabled

**Estimated capacity needed:** For 100K controlled-access samples at ~8 hrs/sample median (from our v5b benchmarks): ~800K core-hours, or ~42 node-days on Launch at 192 cores/node. This is modest.

### 7.3 Cloud Options (When Required)

| Platform | Cost Model | Best For |
|----------|-----------|----------|
| AnVIL (Terra/GCP) | Pay-per-compute + storage | dbGaP data already staged in AnVIL |
| BioData Catalyst (AWS/GCP) | NIH cloud credits available | NHLBI/TOPMed data |
| UK Biobank RAP (DNAnexus) | UKB-specific billing | UK Biobank data (mandatory) |
| EGA Federated (EMBL-EBI) | Institutional agreement | EU-restricted EGA data |

**Cloud cost estimate for 100K samples:**
- Compute: ~$0.50-0.80/sample (n2-highmem-16 or similar, 8 hrs) = $50K-80K
- Storage: ~$0.02/GB/month, 50 TB FASTQ = $1K/month during processing
- Egress: Count matrices (~1-5 GB total) = negligible
- **Total: ~$55K-85K** vs ~$0 on university HPC

### 7.4 Recommended Hybrid Architecture

```
PUBLIC DATA (206K samples)                CONTROLLED DATA (est. 100K-500K samples)
         |                                            |
    ACCESS / TAMU Launch                    University HPC (secure partition)
    (BIO260157 credits)                     (SBP-compliant, DUC-restricted)
         |                                            |
    simpleaf pipeline                           simpleaf pipeline
    (identical code)                            (identical code)
         |                                            |
    count matrices                              count matrices
         |                                            |
         +-------- MERGE on secure HPC --------+
                          |
                   Combined atlas
                          |
                   NMF training (W matrix)
                          |
              W matrix is NON-PHI --> publishable
              H matrix is CONTROLLED --> stays on secure HPC
              Factor summaries are NON-PHI --> publishable
```

---

## 8. Data Movement & Transfer Constraints

### 8.1 What Can Leave the Secure Environment?

| Artifact | Can Leave? | Rationale |
|----------|-----------|-----------|
| Raw FASTQ/BAM | NO | Individual-identifiable; must stay in secure storage |
| Count matrices (genes x cells) | GRAY AREA | Not directly identifiable but could be fingerprinted; treat as controlled |
| Cell-level QC metrics | YES | Technical metrics, not individual-linked |
| Per-sample QC summaries | YES | Aggregate; no individual data |
| NMF W matrix | YES | Population-level gene programs; no individual data |
| Factor summary statistics | YES | Aggregated across donors; non-identifying |
| Trained CPM model weights | YES | Neural network parameters; no individual data recoverable |
| Individual H vectors | NO | Linked to specific cells/donors; controlled |

### 8.2 Transfer Protocol

1. **Inbound:** FASTQ downloaded from dbGaP/EGA to secure HPC partition via SRA Toolkit/pyega3
2. **Processing:** simpleaf runs entirely within secure partition; count matrices stay local
3. **NMF training:** Runs on secure partition; W matrix extracted, H matrix stays
4. **Outbound:** Only W matrix, factor summaries, and model weights leave secure environment
5. **Deletion:** Raw FASTQ, count matrices, and H matrices destroyed per DUC timeline

### 8.3 Can Count Matrices Leave the Cloud/Secure Environment?

**Conservative position (recommended):** Treat count matrices as controlled data. While a single-cell count matrix does not contain direct identifiers (no names, no SSNs), the combination of gene expression profile + metadata could theoretically re-identify donors if an adversary had access to a reference database of the same individuals.

**Practical implications:**
- Count matrices should be processed on the same secure system where FASTQ resides
- NMF training runs co-located with count matrices
- Only model outputs (W, summaries) leave the secure environment
- This is the most defensible position and avoids any gray-area disputes

---

## 9. NMF PHI-Safe Architecture

### 9.1 De-Identification at the Model Level

The NMF decomposition X = WH provides a natural de-identification boundary:

**W matrix (genes x factors, e.g., 30,000 x 2,048):**
- Each column is a gene program (e.g., "oxidative phosphorylation", "T cell activation")
- Learned from millions of cells across thousands of donors
- No single donor contributes meaningfully to any column of W
- Analogous to a dictionary -- the words exist independent of who spoke them
- **SAFE TO PUBLISH AND DISTRIBUTE**

**H matrix (factors x cells, e.g., 2,048 x 500M):**
- Each column is one cell's program activity vector
- Cells can be grouped by donor; H columns of a donor form that donor's expression fingerprint
- **MUST REMAIN IN CONTROLLED ENVIRONMENT**

**Factor summary statistics (per condition):**
- Mean and variance of each factor across all cells of a given condition
- Example: "Factor 47 (lipid metabolism) has mean=2.3, sd=0.8 in NASH hepatocytes"
- Aggregated across many donors -- no individual contributor is recoverable
- Minimum aggregation: report statistics only for groups of >= 10 donors (k-anonymity)
- **SAFE TO PUBLISH AND DISTRIBUTE**

### 9.2 Formal Privacy Argument

The W matrix satisfies the definition of a **sufficient statistic** for the data-generating distribution that is independent of individual observations:

1. **W is trained on N >> 1M cells from D >> 1K donors.** Removing any single donor's cells changes W by epsilon << noise floor.
2. **W has fixed dimensionality (g x k)** independent of N. It does not grow with more data and cannot memorize individuals.
3. **No reconstruction path:** Given W alone, there is no function f(W) that returns any individual cell's expression profile or any donor's metadata. You need H for that, and H stays controlled.
4. **Factor summaries** satisfy k-anonymity (k >= 10 donors per group) and l-diversity (factors represent orthogonal biological processes, not individual traits).

### 9.3 What We Would Provide to Users (via singlet API / singletdb)

| Output | Description | PHI Status |
|--------|-------------|------------|
| Gene programs (W columns) | Top genes per factor, loadings | Non-PHI |
| Factor annotations | Biological labels per factor | Non-PHI |
| Condition-factor summaries | Mean/sd of each factor per tissue/disease/treatment | Non-PHI (if k >= 10 donors) |
| Predicted expression | W * h_predicted for unseen conditions via CPM | Non-PHI (synthetic) |
| Cell type proportions | Fraction of cells in each cluster per condition | Non-PHI (aggregated) |

**None of these outputs expose individual-level data.** The controlled-access data is consumed during training and its individual-level information is destroyed when H and count matrices are deleted per DUC requirements.

---

## 10. Actionable Roadmap

### Phase 1: Discovery & Catalog (Month 1-2)

- [ ] Build controlled-access discovery pipeline (Section 4.4)
- [ ] Query dbGaP, EGA, GSA-Human for single-cell studies
- [ ] Cross-reference with existing public catalog to identify unique controlled-access studies
- [ ] Produce controlled_access_catalog.parquet with priority scores
- [ ] Estimate total cell count and assay distribution

### Phase 2: Institutional Preparation (Month 1-3, parallel)

- [ ] Engage university Office of Research for SO designation
- [ ] Obtain or verify eRA Commons accounts for all project personnel
- [ ] Prepare IRB exemption or approval for secondary data analysis
- [ ] Draft institutional IT Security Plan for dbGaP/EGA data
- [ ] Document HPC secure partition specifications (encryption, ACLs, logging)

### Phase 3: Access Applications (Month 2-6)

- [ ] Submit batch dbGaP DAR for GRU-consent studies (Phase 1 in Section 5.3)
- [ ] Submit dbGaP DARs for disease-specific studies (Phase 2)
- [ ] Contact top-priority EGA DACs with DAA requests (Phase 3)
- [ ] Track application status in controlled_access_catalog.parquet

### Phase 4: Reprocessing (Month 4-12, rolling)

- [ ] Set up secure HPC partition with SBP compliance
- [ ] Download approved FASTQ to secure partition
- [ ] Run identical simpleaf pipeline on controlled-access data
- [ ] QC and merge count matrices with public atlas
- [ ] Progressive NMF training as batches arrive

### Phase 5: Model Training & Deployment (Month 8-14)

- [ ] Train NMF on combined public + controlled-access atlas
- [ ] Extract W matrix and factor summaries
- [ ] Validate factor stability (compare W from public-only vs combined)
- [ ] Deploy W matrix and summaries via singlet API / singletdb
- [ ] Destroy controlled-access FASTQ, count matrices, and H per DUC requirements

---

## Appendix A: Discovery Query Templates

### dbGaP Entrez Query
```python
import requests

def search_dbgap_sc(disease=None):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    term = '"single cell"[Study Description] OR "scRNA-seq"[Study Description]'
    if disease:
        term += f' AND "{disease}"[Study Disease/Focus]'
    params = {"db": "gap", "term": term, "retmax": 10000, "retmode": "json"}
    resp = requests.get(base, params=params)
    return resp.json()["esearchresult"]["idlist"]
```

### SRA Controlled-Access Query
```python
def search_sra_controlled():
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    term = (
        '("single cell"[Selection] OR "10x"[Platform])'
        ' AND "controlled access"[Access]'
        ' AND "rna seq"[Strategy]'
    )
    params = {"db": "sra", "term": term, "retmax": 50000, "retmode": "json"}
    resp = requests.get(base, params=params)
    return resp.json()["esearchresult"]["idlist"]
```

---

## Appendix B: Cost Model Comparison

| Scenario | Compute | Storage | Egress | Total |
|----------|---------|---------|--------|-------|
| 100K samples on university HPC | $0 (existing) | $0 (existing) | $0 | **$0** |
| 100K samples on AnVIL (GCP) | ~$60K | ~$2K | <$1K | **~$62K** |
| 100K samples on BioData Catalyst | ~$55K | ~$2K | <$1K | **~$57K** |
| UK Biobank (RAP, mandatory) | ~$5K (small scale) | ~$500 | <$100 | **~$6K** |

**University HPC is 60-100x cheaper** than cloud for the same workload. Cloud is only justified when mandated by the data repository (e.g., UK Biobank RAP) or when geographic restrictions prevent download.

---

## Appendix C: Key Contacts & Resources

| Resource | URL / Contact |
|----------|--------------|
| dbGaP Help Desk | dbgap-help@ncbi.nlm.nih.gov |
| EGA Helpdesk | helpdesk@ega-archive.org |
| NIH GDS Policy | https://sharing.nih.gov/genomic-data-sharing-policy |
| NIH Security Best Practices | https://osp.od.nih.gov/scientific-sharing/security-best-practices/ |
| AnVIL Portal | https://anvilproject.org/ |
| BioData Catalyst | https://biodatacatalyst.nhlbi.nih.gov/ |
| HTAN Data Portal | https://data.humantumoratlas.org/ |
| HuBMAP Portal | https://portal.hubmapconsortium.org/ |
| CELLxGENE Census | https://cellxgene.cziscience.com/ |
| GSA-Human | https://ngdc.cncb.ac.cn/gsa-human/ |
| JGAS/NBDC | https://humandbs.dbcls.jp/en/ |
| ACCESS Allocations | https://access-ci.org/ |
