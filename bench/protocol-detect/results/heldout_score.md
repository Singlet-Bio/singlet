# Protocol detection score: heldout.tsv

Rows 109. Family recall 96/109 = 0.881. Family precision 96/109 = 0.881. Version recall 63/66 = 0.955.
Relabeled to no_cell_barcode by the deposit rule: 6. Decided by trial mapping: 2; trial pending: 0.

| family | rows | recall | calls | precision |
|---|---|---|---|---|
| 10x_3p | 60 | 59 | 59 | 59 |
| 10x_5p | 3 | 3 | 5 | 3 |
| 10x_arc_gex | 5 | 4 | 5 | 4 |
| atac | 4 | 4 | 4 | 4 |
| feature_barcode | 4 | 4 | 5 | 4 |
| no_cell_barcode | 13 | 13 | 21 | 13 |
| other_barcoded | 13 | 6 | 7 | 6 |
| vdj | 2 | 0 | 0 | 0 |
| visium | 4 | 3 | 3 | 3 |
| visium_probe | 1 | 0 | 0 | 0 |

Protocol tag right within other_barcoded: 2/6

## Misses

- `GSM5534708` truth 10x_arc_gex/10x-arc-gex; called feature_barcode/10x-feature-barcode (detector). candidates: feature_barcode/10x-feature-barcode 1.0; 10x_3p/10x-3p-v3 0.255; 10x_5p/10x-5p-v3 0.09. evidence: v3 list 95.7% at slot 1 offset 0 strand + (slot length 28) | cDNA slot 2: top-100 15-mers 98%, capture/scaffold constant 0% (tag library)  
- `GSM8618582` truth 10x_3p/10x-3p-v4; called 10x_arc_gex/10x-arc-gex (detector). candidates: 10x_arc_gex/10x-arc-gex 1.0. evidence: arc_gex list 91.6% at slot 3 offset 0 strand + (slot length 28)  
- `mereu_SCRBseq_N702` truth other_barcoded/scrbseq; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.24%) | slots: 16, 50 b | Tn5 mosaic end 0%  
- `mereu_SCRBseq_N703` truth other_barcoded/scrbseq; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.24%) | slots: 16, 50 b | Tn5 mosaic end 0%  
- `mereu_ddSEQ_Sample2` truth other_barcoded/ddseq; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.16%) | slots: 52, 75 b | Tn5 mosaic end 3%  
- `mereu_ddSEQ_Sample3` truth other_barcoded/ddseq; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.94%) | slots: 52, 75 b | Tn5 mosaic end 2%  
- `SRR17209008` truth visium/10x-visium; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.00%) | slots: 90 b | Tn5 mosaic end 0%  
- `SRR29202305` truth visium_probe/10x-visium-ffpe-probe; called other_barcoded/dropseq (detector). candidates: other_barcoded/dropseq 1.0. evidence: slot 3 (28 b): first 12 bases repeat >= 5x in 94% (bead barcodes), poly(T) at 18  
- `SRR1853178` truth other_barcoded/dropseq; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.33%) | slots: 20, 60 b | Tn5 mosaic end 0%  
- `SRR19645614` truth other_barcoded/celseq2; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.21%) | slots: 16, 51 b | Tn5 mosaic end 0%  
- `SRR21226943` truth other_barcoded/scrbseq; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.24%) | slots: 8, 15, 69 b | Tn5 mosaic end 0%  
- `SRR15178153` truth vdj/10x-5p-vdj; called 10x_5p/10x-5p-v2 (detector). candidates: 10x_5p/10x-5p-v2 1.0; 10x_3p/10x-3p-v2 0.1. evidence: v2 list 91.2% at slot 2 offset 0 strand + (slot length 150) | TSO at 26 in 86% of barcode reads  
- `SRR15178156` truth vdj/10x-5p-vdj; called 10x_5p/10x-5p-v2 (detector). candidates: 10x_5p/10x-5p-v2 1.0; 10x_3p/10x-3p-v2 0.1. evidence: v2 list 89.8% at slot 2 offset 0 strand + (slot length 150) | TSO at 26 in 83% of barcode reads  
