# Protocol detection score: dev.tsv

Rows 114. Family recall 105/114 = 0.921. Family precision 105/114 = 0.921. Version recall 65/68 = 0.956.
Relabeled to no_cell_barcode by the deposit rule: 9. Decided by trial mapping: 4; trial pending: 0.

| family | rows | recall | calls | precision |
|---|---|---|---|---|
| 10x_3p | 59 | 57 | 58 | 57 |
| 10x_5p | 5 | 5 | 6 | 5 |
| 10x_arc_gex | 3 | 2 | 2 | 2 |
| 10x_flex | 1 | 0 | 0 | 0 |
| atac | 5 | 2 | 2 | 2 |
| feature_barcode | 5 | 5 | 5 | 5 |
| no_cell_barcode | 17 | 17 | 23 | 17 |
| other_barcoded | 12 | 12 | 13 | 12 |
| vdj | 1 | 0 | 0 | 0 |
| visium | 5 | 5 | 5 | 5 |
| visium_probe | 1 | 0 | 0 | 0 |

Protocol tag right within other_barcoded: 7/12

## Misses

- `SRR27536879` truth visium_probe/10x-visium-cytassist-probe; called other_barcoded/barcoded-unknown (detector). candidates: other_barcoded/barcoded-unknown 0.6. evidence: slot 1 (28 b): repeated 12-base prefixes in 90%  
- `SRR18070426` truth 10x_arc_gex/10x-arc-gex; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.00%) | slots: 90 b | Tn5 mosaic end 0%  
- `SRR18070627` truth atac/10x-arc-atac; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.21%) | slots: 51, 51 b | Tn5 mosaic end 0%  
- `SRR18070626` truth atac/10x-arc-atac; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.00%) | slots: 51, 51 b | Tn5 mosaic end 0%  
- `SRR18070625` truth atac/10x-arc-atac; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.00%) | slots: 51, 51 b | Tn5 mosaic end 0%  
- `SRR7722018` truth 10x_3p/10x-3p-v2; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.00%) | slots: 99 b | Tn5 mosaic end 0%  
- `SRR7788880` truth 10x_3p/10x-3p-v2; called no_cell_barcode/rna-no-barcode (detector). candidates: no_cell_barcode/rna-no-barcode 0.9. evidence: no 10x list above 5% in any slot at any offset or strand (best 0.06%) | slots: 98 b | Tn5 mosaic end 0%  
- `SRR31116722` truth 10x_flex/10x-flex; called 10x_3p/10x-3p-v4 (detector). candidates: 10x_3p/10x-3p-v4 0.5; atac/10x-arc-atac 0.308. evidence: v4_3p list 9.4% at slot 1 offset 0 strand + (slot length 28) | barcode read too short to see poly(T) or TSO after barcode+UMI; cDNA read-through poly(A) 0.0% / TSO 0.0%: undecided  
- `king_tonsil_bcr` truth vdj/10x-5p-vdj-bcr; called 10x_5p/10x-5p-v2 (trial). candidates: 10x_3p/10x-3p-v2 0.5; 10x_5p/10x-5p-v2 0.5. evidence: v2 list 95.1% at slot 2 offset 0 strand + (slot length 155) | poly(T) 0% / TSO 0% at 26: undecided  
