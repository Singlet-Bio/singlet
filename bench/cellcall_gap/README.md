# Cell calling: recall at a matched number of called cells (development libraries, v1.3.4)

`matched_count.py` scores the registered counted metric on the gate libraries: each tool contributes its top N barcodes, N being the number of
truth cells, with a paired bootstrap 95% interval of Singlet minus each tool. Competitors carry no score in the gate tables, so each is ordered
by its called set first and total UMIs second (a stated approximation). Result file: `matched_count_v1.3.4.json` (job 803557 on the v1.3.4 runs).

At matched count Singlet is at or above CellBender and STARsolo EmptyDrops_CR on all eight libraries (differences from -0.0010 to +0.040; the only
negative ones, kang2018_ctrl against CellBender -0.0005 [-0.0012, 0.0002] and hgmm1k_v3 against EmptyDrops_CR -0.0010 [-0.0050, 0.0029], include zero).
The four failing gate rows (neuron1k_v3 recall, kang2018_ctrl recall and Jaccard, nuclei2k_v2 precision) come from the operating point:
CellBender calls 1,470 barcodes where Cell Ranger has 1,301 (neuron) and 16,440 where demuxlet has 14,619 (Kang), which buys recall that
no single operating point can match while also holding precision. No engine change is proposed.
