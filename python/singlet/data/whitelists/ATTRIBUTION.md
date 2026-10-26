# Barcode lists rebuilt from 10x Genomics public datasets

Three files in this directory were rebuilt only from **10x Genomics public datasets**, which 10x publishes under the
**Creative Commons Attribution 4.0 International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/)**.
Nothing was taken from Space Ranger, Cell Ranger ARC, Cell Ranger ATAC or any other 10x software. Source: 10x Genomics,
https://www.10xgenomics.com/datasets. 10x Genomics does not endorse Singlet. The code is `bench/atac_spatial/make_lists.py`
(Singlet repository), run 2026-10-25 on the files listed below, downloaded by `bench/atac_spatial/download.sbatch`.

| file | sha256 | rows | how it was rebuilt |
|---|---|---|---|
| `visium-v1.txt` | `87f515caefd47fed9388ae038607e12b0bf42d156cbb70da3111fd98780caa0f` | 4,992 | barcodes of `spatial/tissue_positions_list.csv` of the three Visium datasets below (identical in all three), "-1" suffix removed, ordered by array row and column |
| `visium-v1_coordinates.tsv` | `c9ead69cfa064e7ec3d405a66c1951496641f63fd1b147b1e4e8d25b21ac96cd` | 4,992 + header | the same rows with `array_row`, `array_col` (no pixel coordinates: those depend on each slide's image) |
| `737K-arc-v1-atac-gex.tsv.gz` | `dde9b07932aa097219122770c2c207ffe05e405eb13cfbed387dab871a848638` | 736,198 | `atac_barcode` <TAB> `gex_barcode` pairs from the `per_barcode_metrics.csv` of the seven Multiome datasets below (union, "-1" removed; no barcode pairs with two partners). It covers 736,198 of the 736,320 barcodes of `gex_737K-arc-v1.txt`. Leaving any one dataset out, the other six still contain the pair of 100% of that dataset's called cells |
| `737K-cratac-v1.txt` | `53cb87ae2aa8e38ab7efb9ef625a725359a3235ded866bb1e0d4157c308edcc9` | 736,795 | union of the barcodes in the `singlecell.csv` of the twelve Cell Ranger ATAC datasets below ("NO_BARCODE" and "-1" removed). Leaving any one dataset out, the others contain ≥ 99.95% of that dataset's called cells |

**The previous `737K-cratac-v1.txt` was wrong.** It was byte-identical to `737K-august-2016.txt` (the 10x 3' v2 GEX list,
sha256 of that file in the old checkout `0bd69cc671d7ad3c569c71606bb78443eef52ead3b5819bf0985ec9827eb76ec`). It shares
10,801 barcodes with the rebuilt list and matched ~1.2% of the barcodes of a real scATAC library. The rebuilt list matches
94.6% exactly (98.4% after one-mismatch correction) on 10x scATAC PBMC 5k Next GEM.

## Source datasets (10x Genomics, CC BY 4.0)

Visium (Space Ranger 1.1.0 outputs), `https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/<name>/<name>_spatial.tar.gz`:

| dataset | bytes | sha256 of `_spatial.tar.gz` |
|---|---|---|
| V1_Mouse_Brain_Sagittal_Anterior | 9,233,573 | `5f41a803e2bd69fa4dfca6abc8fa2d4e0d76aeb6c72d7038a5fdcf9cc50a36f8` |
| V1_Human_Lymph_Node | 8,246,238 | `812808883366ff9623dc8354847a7211b0d922b2bfc4c9359d6e12e993ea6a73` |
| V1_Breast_Cancer_Block_A_Section_1 | 9,960,960 | `415530a9d31172f63fd358d5d1cefb0798698ef9c0e614f52825894cd45c61bf` |

Multiome (Cell Ranger ARC 2.0.0 outputs), `https://cf.10xgenomics.com/samples/cell-arc/2.0.0/<name>/<name>_per_barcode_metrics.csv`:

| dataset | bytes | sha256 |
|---|---|---|
| pbmc_granulocyte_sorted_3k | 77,721,645 | `d8361253c64a67fd31ed9a226623d1d824b9acbc1bfe18c7c4b897de9d02c4f7` |
| pbmc_granulocyte_sorted_10k | 88,822,764 | `fd3e069b83e152145af234667b419c982968aca0df322a92adb71284d0d902cd` |
| pbmc_unsorted_3k | 72,092,653 | `588bfb89e9ba20090615f94c0d95067bdebace72e42e102e06077ad590f4ff14` |
| pbmc_unsorted_10k | 89,487,185 | `998138c74d8a4fda9e15b55017310b40258f1421dcd29c1204120e7ceea062c4` |
| human_brain_3k | 81,812,727 | `c77cf9873a40e1def44ce092bfecea4f57815c421231c4b0ccae9b4c2c1b812e` |
| e18_mouse_brain_fresh_5k | 84,337,849 | `fc961f16dd93d86f840db12af11740328615e57984072017969aaa59eed8b38d` |
| lymph_node_lymphoma_14k | 84,924,021 | `2e31d48a687ddc991bba5b33c6c4d5bfd03ab52c3b1c4f94b76a91c764e93c2b` |

scATAC (Cell Ranger ATAC 1.2.0 outputs), `https://cf.10xgenomics.com/samples/cell-atac/1.2.0/<name>/<name>_singlecell.csv`:

| dataset | bytes | sha256 |
|---|---|---|
| atac_pbmc_5k_nextgem | 29,678,090 | `dcacebb4c0fc00c096a6c3d965385087639cc8244b7aee7b35b002565915d583` |
| atac_pbmc_10k_nextgem | 36,486,486 | `5b4bced8fa10f52fad4dd4619c6a208423c5d468fbc91bcb8890931523f6457c` |
| atac_pbmc_1k_nextgem | 16,965,589 | `7612a0612ea2924b640aeeab02412141e3899071fef7b758ea63f7b390db5067` |
| atac_pbmc_500_nextgem | 14,393,663 | `974fa4ad9d4f959b1fec631b0ad5aaf5f707c26ede26c7410657420774809709` |
| atac_pbmc_10k_v1 | 30,871,355 | `9c8aa80f5d379b1e6327852888a5193c4ee5f749675169f2c4ff0a187863de18` |
| atac_pbmc_5k_v1 | 23,201,033 | `0d4b9a294b66e008a1258098910ad930c21a8aa0174cc8775281bceac727c882` |
| atac_pbmc_1k_v1 | 13,184,798 | `a6ef633ae2dd4a9b8df13bbf1b8eed284b10769a8e45e2824ec409b4024fa057` |
| atac_v1_pbmc_10k | 37,116,270 | `aabac449d3706d13d303b70e1d372c533708bebbe84fe97b9fe5adcb04ffe88a` |
| atac_v1_pbmc_5k | 32,182,575 | `5b1e92a21c9f7554b780ed1947c994697fa9067c469fe1a06767a96a3c3bbd23` |
| atac_hgmm_10k_nextgem | 43,140,358 | `90738a593dc98c0c9b57d394192af207f4d4925cdadf3dedca8f555cca312580` |
| atac_hgmm_5k_v1 | 39,951,457 | `1eb8eb8a655642eb5cf601e77b43e290fad617942d8c4ac81988bebb40d0f615` |
| atac_v1_E18_brain_fresh_5k | 29,034,595 | `fade603d33e94a6e5fd9d93b7bb85fb329eaa2bc3e56af8364367314503aaa97` |
