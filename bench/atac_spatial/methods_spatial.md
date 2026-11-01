**Data.** 10x Visium mouse brain, sagittal anterior section 1 (CC BY 4.0), processed from its published FASTQs with 10x
GRCm39-2024-A. Truth: the Space Ranger 1.1.0 published spot matrix (exonic counts) and tissue positions. Space Ranger
detects tissue from the H&E image.

**Singlet 1.0.0.** `singlet encode --protocol 10x-visium`, then the standard pipeline. Spot barcodes are matched to the
visium-v1 spot list (4,992 spots, rebuilt from three 10x public datasets; exact or unique Hamming-1). Tissue is detected
from counts alone, without the image:
- a two-component mixture on log10 UMIs per spot;
- mean-field smoothing on the hexagonal array, with per-spot evidence capped so that six concordant neighbours decide.

Tissue spots are the sample's called set. Exon counts (spliced + ambiguous molecules) are compared, because Space Ranger
1.1 counts exonic reads only.

**Competitors.**
- Counts: STARsolo 2.7.11b with the same spot list, Cell Ranger-like settings (CB 1 mismatch with pseudocounts, UMI
  1MM_CR, MultiGeneUMI_CR, CellRanger4 adapter clipping), same FASTQs, reference and threads.
- Tissue: an Otsu threshold on log10 UMIs per spot, the usual count-based fallback when no image is available.

**Metrics.** Genes are matched by Ensembl id. The first three metrics are over Space Ranger's tissue spots:
- Spearman of UMIs per spot and of genes per spot with Space Ranger;
- median per-spot Pearson of log-CPM expression profiles with Space Ranger;
- Spearman of molecules per gene summed over the tissue spots;
- Cohen's kappa of the tissue call with Space Ranger over all 4,992 spots.

All are match-or-beat against the competitor, with no tolerance.
