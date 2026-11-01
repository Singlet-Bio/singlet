# V(D)J germline references

| file | use | CDR3 start |
|---|---|---|
| `vdj_human.tsv.gz` | human; matches Cell Ranger 3.x / 4.x calls | second of two adjacent Cys codons |
| `vdj_human_imgtcys.tsv.gz` | human; matches Cell Ranger >= 5 calls | first of two adjacent Cys codons |
| `vdj_mouse.tsv.gz` | mouse | |

The first line of each file (`# ...`) records its provenance.

## Sources

- V / D / J / C segments: GRCh38 genome sequence with the GENCODE v44 annotation, primary assembly plus
  patches / haplotypes, and RefSeq GRCh38. Built by `singlet.vdj.reference`.
- Added IG V genes (human files only, since 2026-10-31). These are genes the GRCh38 annotations lack: IGHV1-8*01,
  IGHV3-30-3*01, IGHV3-38-3*01, IGHV3-43D*03, IGHV3-9*01, IGHV4-30-4*01 and IGHV4-38-2*01.
  - Source: the AIRR Community (AIRR-C) human reference germline sets, from OGRDB (https://ogrdb.airr-community.org).
    The published versions are IGH_VDJ r10, IGKappa_VJ r5 and IGLambda_VJ r4. The Zenodo DOIs are
    10.5281/zenodo.20409587, 10.5281/zenodo.20409594 and 10.5281/zenodo.20409602.
  - Download URL: `/download_germline_set/Homo%20sapiens/<set>/published/ungapped_ex`, fetched on 2026-10-31.
  - FASTA record counts and sha256 prefixes:
    - IGH: 240 records, 5beb005256e702a4
    - IGK: 84 records, 78f24cd89d3a9ff2
    - IGL: 97 records, d3a9949dac9b48be
  - Licence: CC0 1.0, from the OGRDB licensing statement
    (https://ogrdb.airr-community.org/render_page/licensing_statement.html), which says the information provided on the
    site is published under a CC0 licence.
  - No IMGT data is used.
  - Built with `bench/expression_sota/vdj2/add_airrc.py` (`airrc.sbatch`).
  - Rule: a gene is added only when it is absent by name. Genes whose names contain "NL" (not localised), alleles with
    non-ACGT bases, and genes with a stop codon before Cys104 or with no anchor are skipped. IGHV1-38-4 and IGKV2-29
    were skipped this way.
  - These segments carry `gene_id` `AIRR-C:<allele>` and `contig` `AIRR-C`.

sha256 of the files:
- `vdj_human.tsv.gz`: 2108ecffe1775dc4
- `vdj_human_imgtcys.tsv.gz`: deb8e60355277f66
