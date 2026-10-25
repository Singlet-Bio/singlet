## datasets

```
        dataset           study  cells  doublets  rate species  genes  median_umi  synthetic
 HEK-HMEC-MULTI    mcginnis2019  10641       489 0.046   human  23982     17424.0      False
HMEC-orig-MULTI    mcginnis2019  26426      3568 0.135   human  24199     23502.0      False
 HMEC-rep-MULTI    mcginnis2019  10580      3282 0.310   human  17473      1187.5      False
       J293t-dm   kang2018_293t    500        42 0.084   human  16374     14134.5      False
       cline-ch   stoeckius2018   7954      1465 0.184   human  25221      4824.5      False
   hgmm1k_synth        10x_hgmm    924        45 0.049   mixed 116904     21661.0       True
         hm-12k        10x_hgmm  12820       730 0.057   human  15106     12424.0      False
          hm-6k        10x_hgmm   6806       171 0.025   human  15080     21301.0      False
         kang21        kang2018  13737      1598 0.116   human  38606      1594.0      False
  kang21_homsim        kang2018  13351      1212 0.091   human  38606      1586.0       True
     mkidney-ch mkidney_hashing  21179      7901 0.373   mouse  18940      3929.0      False
      nuc-MULTI    mcginnis2019   5578       475 0.085   human  21490      1021.0      False
     pbmc-1A-dm        kang2018   3298       120 0.036   human  15170       973.0      False
     pbmc-1B-dm        kang2018   3790       130 0.034   human  15143       862.0      False
     pbmc-1C-dm        kang2018   5270       316 0.060   human  15865       829.0      False
  pbmc-2ctrl-dm        kang2018  13913      1598 0.115   human  17584      1276.0      False
  pbmc-2stim-dm        kang2018  13916      1631 0.117   human  17315      1360.5      False
        pbmc-ch   stoeckius2018  15272      2545 0.167   human  21639       556.0      False
      pdx-MULTI    mcginnis2019  10296      1317 0.128   human  14025      2242.5      False
```

## mean / median over real-label datasets

```
                             auroc               auprc              recall_at_rate             
                              mean median count   mean median count           mean median count
method                                                                                         
atlas_clf                    0.798  0.802    17  0.484  0.534    17          0.501  0.531    17
atlas_knn                    0.795  0.812    17  0.501  0.523    17          0.520  0.547    17
atlas_nnls                   0.617  0.609    17  0.160  0.159    17          0.172  0.180    17
combo_all                    0.803  0.812    17  0.572  0.549    17          0.564  0.542    17
combo_atlas                  0.799  0.794    17  0.536  0.548    17          0.535  0.517    17
combo_no_atlas               0.798  0.795    17  0.547  0.490    17          0.547  0.530    17
combo_prod_atlas             0.801  0.793    17  0.557  0.550    17          0.554  0.525    17
ct_entropy                   0.625  0.617    17  0.248  0.192    17          0.267  0.238    17
ct_maxprob                   0.603  0.578    17  0.224  0.168    17          0.228  0.181    17
nnls                         0.704  0.687    17  0.309  0.256    17          0.330  0.295    17
prod                         0.787  0.794    17  0.501  0.457    17          0.516  0.508    17
prod_geno                    0.969  0.969     1  0.897  0.897     1          0.864  0.864     1
rankavg_sdf_solo_atlas_prod  0.804  0.809    17  0.561  0.522    17          0.554  0.551    17
scdblfinder                  0.798  0.800    17  0.559  0.534    17          0.567  0.558    17
scrublet                     0.761  0.734    17  0.466  0.460    17          0.475  0.462    17
solo                         0.753  0.729    17  0.412  0.428    17          0.427  0.448    17
umi                          0.720  0.767    17  0.269  0.264    17          0.295  0.294    17
vireo_geno                   0.950  0.950     1  0.842  0.842     1          0.840  0.840     1
```

## AUROC

```
method           atlas_clf  atlas_knn  atlas_nnls  combo_all  combo_atlas  combo_no_atlas  combo_prod_atlas  ct_entropy  ct_maxprob   nnls   prod  prod_geno  rankavg_sdf_solo_atlas_prod  scdblfinder  scrublet   solo    umi  vireo_geno
dataset                                                                                                                                                                                                                                   
HEK-HMEC-MULTI       0.802      0.812       0.634      0.813        0.794           0.795             0.793       0.416       0.432  0.726  0.794        NaN                        0.809        0.800     0.762  0.751  0.767         NaN
HMEC-orig-MULTI      0.784      0.792       0.636      0.779        0.776           0.774             0.777       0.479       0.519  0.687  0.754        NaN                        0.790        0.777     0.732  0.755  0.701         NaN
HMEC-rep-MULTI       0.708      0.706       0.582      0.709        0.685           0.717             0.699       0.439       0.482  0.636  0.689        NaN                        0.725        0.713     0.647  0.660  0.644         NaN
J293t-dm             0.512      0.472       0.494      0.486        0.527           0.481             0.489       0.474       0.459  0.573  0.459        NaN                        0.475        0.538     0.561  0.500  0.420         NaN
cline-ch             0.631      0.619       0.515      0.634        0.627           0.624             0.629       0.510       0.509  0.555  0.604        NaN                        0.625        0.615     0.601  0.623  0.544         NaN
hm-12k               0.966      0.982       0.731      0.996        0.989           0.994             0.992       0.960       0.948  0.955  0.974        NaN                        0.992        1.000     0.992  0.957  0.881         NaN
hm-6k                0.985      0.993       0.768      0.999        0.997           0.998             0.998       0.970       0.969  0.920  0.992        NaN                        0.997        1.000     0.994  0.968  0.888         NaN
kang21               0.918      0.923       0.665      0.934        0.930           0.924             0.935       0.723       0.639  0.738  0.919      0.969                        0.929        0.927     0.880  0.876  0.820        0.95
mkidney-ch           0.697      0.685       0.529      0.707        0.683           0.697             0.696       0.556       0.545  0.649  0.700        NaN                        0.716        0.700     0.659  0.673  0.603         NaN
nuc-MULTI            0.785      0.776       0.666      0.789        0.774           0.784             0.792       0.522       0.510  0.704  0.785        NaN                        0.795        0.778     0.734  0.693  0.720         NaN
pbmc-1A-dm           0.841      0.813       0.603      0.812        0.826           0.805             0.823       0.710       0.648  0.671  0.805        NaN                        0.809        0.800     0.732  0.720  0.781         NaN
pbmc-1B-dm           0.748      0.772       0.606      0.789        0.782           0.790             0.780       0.703       0.635  0.678  0.748        NaN                        0.774        0.763     0.729  0.709  0.689         NaN
pbmc-1C-dm           0.827      0.833       0.644      0.825        0.835           0.820             0.831       0.679       0.600  0.687  0.815        NaN                        0.828        0.824     0.754  0.729  0.771         NaN
pbmc-2ctrl-dm        0.908      0.912       0.602      0.926        0.917           0.917             0.923       0.710       0.659  0.711  0.916        NaN                        0.925        0.922     0.873  0.855  0.800         NaN
pbmc-2stim-dm        0.908      0.914       0.609      0.907        0.913           0.904             0.921       0.617       0.594  0.712  0.917        NaN                        0.926        0.918     0.862  0.849  0.797         NaN
pbmc-ch              0.834      0.827       0.618      0.841        0.834           0.834             0.839       0.628       0.578  0.708  0.820        NaN                        0.834        0.822     0.777  0.766  0.774         NaN
pdx-MULTI            0.707      0.685       0.589      0.709        0.689           0.708             0.695       0.524       0.525  0.657  0.687        NaN                        0.714        0.673     0.644  0.717  0.640         NaN
```

## AUPRC

```
method           atlas_clf  atlas_knn  atlas_nnls  combo_all  combo_atlas  combo_no_atlas  combo_prod_atlas  ct_entropy  ct_maxprob   nnls   prod  prod_geno  rankavg_sdf_solo_atlas_prod  scdblfinder  scrublet   solo    umi  vireo_geno
dataset                                                                                                                                                                                                                                   
HEK-HMEC-MULTI       0.344      0.384       0.066      0.485        0.275           0.484             0.384       0.036       0.037  0.245  0.425        NaN                        0.486        0.534     0.460  0.322  0.187         NaN
HMEC-orig-MULTI      0.478      0.511       0.193      0.503        0.471           0.490             0.478       0.121       0.133  0.308  0.457        NaN                        0.517        0.503     0.400  0.428  0.359         NaN
HMEC-rep-MULTI       0.586      0.565       0.364      0.598        0.548           0.604             0.578       0.268       0.289  0.441  0.579        NaN                        0.620        0.603     0.481  0.480  0.501         NaN
J293t-dm             0.125      0.180       0.088      0.227        0.196           0.198             0.214       0.077       0.074  0.231  0.190        NaN                        0.187        0.213     0.240  0.137  0.070         NaN
cline-ch             0.404      0.409       0.191      0.427        0.392           0.416             0.408       0.195       0.194  0.240  0.400        NaN                        0.428        0.413     0.377  0.406  0.232         NaN
hm-12k               0.569      0.674       0.110      0.952        0.862           0.930             0.892       0.604       0.628  0.803  0.761        NaN                        0.908        0.997     0.917  0.709  0.275         NaN
hm-6k                0.609      0.680       0.061      0.979        0.923           0.964             0.954       0.543       0.585  0.528  0.855        NaN                        0.926        1.000     0.964  0.724  0.145         NaN
kang21               0.595      0.649       0.175      0.688        0.684           0.645             0.698       0.368       0.228  0.256  0.632      0.897                        0.667        0.642     0.582  0.566  0.350       0.842
mkidney-ch           0.598      0.587       0.403      0.610        0.590           0.599             0.598       0.413       0.409  0.534  0.596        NaN                        0.616        0.609     0.545  0.564  0.476         NaN
nuc-MULTI            0.409      0.356       0.147      0.477        0.455           0.467             0.464       0.108       0.095  0.306  0.399        NaN                        0.464        0.432     0.336  0.247  0.218         NaN
pbmc-1A-dm           0.426      0.408       0.048      0.484        0.420           0.450             0.451       0.154       0.100  0.092  0.352        NaN                        0.451        0.460     0.240  0.172  0.137         NaN
pbmc-1B-dm           0.275      0.327       0.063      0.390        0.384           0.350             0.399       0.156       0.096  0.086  0.260        NaN                        0.372        0.361     0.201  0.113  0.112         NaN
pbmc-1C-dm           0.534      0.523       0.092      0.549        0.550           0.472             0.550       0.192       0.127  0.141  0.393        NaN                        0.522        0.515     0.298  0.239  0.202         NaN
pbmc-2ctrl-dm        0.623      0.651       0.159      0.681        0.682           0.630             0.695       0.330       0.263  0.214  0.635        NaN                        0.661        0.641     0.567  0.533  0.312         NaN
pbmc-2stim-dm        0.632      0.639       0.169      0.649        0.673           0.613             0.690       0.189       0.168  0.235  0.621        NaN                        0.659        0.643     0.540  0.510  0.300         NaN
pbmc-ch              0.626      0.616       0.224      0.657        0.636           0.645             0.651       0.344       0.257  0.338  0.600        NaN                        0.647        0.628     0.526  0.506  0.438         NaN
pdx-MULTI            0.395      0.353       0.159      0.358        0.364           0.341             0.369       0.126       0.127  0.258  0.366        NaN                        0.400        0.304     0.251  0.354  0.264         NaN
```

## vs production (real datasets)

```
                     method  datasets  auroc_delta_mean  auprc_delta_mean  auprc_better
                  atlas_clf        17            0.0107           -0.0171            11
                  atlas_knn        17            0.0081           -0.0004             9
                 atlas_nnls        17           -0.1698           -0.3415             0
                 ct_entropy        17           -0.1623           -0.2527             0
                 ct_maxprob        17           -0.1840           -0.2771             0
                       nnls        17           -0.0831           -0.1919             2
                       prod        17            0.0000            0.0000             0
                scdblfinder        17            0.0113            0.0576            16
                   scrublet        17           -0.0263           -0.0349             4
                       solo        17           -0.0339           -0.0888             1
                        umi        17           -0.0670           -0.2318             0
                 vireo_geno         1            0.0314            0.2108             1
                  prod_geno         1            0.0501            0.2652             1
                  combo_all        17            0.0162            0.0704            16
             combo_no_atlas        17            0.0109            0.0458            14
                combo_atlas        17            0.0117            0.0344            12
           combo_prod_atlas        17            0.0139            0.0561            15
rankavg_sdf_solo_atlas_prod        17            0.0168            0.0597            16
```

## cost, median seconds per 10k cells

```
method
atlas_clf       16.83
atlas_knn        1.27
atlas_nnls       1.01
ct_entropy       1.15
ct_maxprob       1.15
nnls            23.43
prod             2.00
scdblfinder     43.55
scrublet         7.70
solo           166.99
```

## per-study means

```
                             auroc_mean_over_studies  auprc_mean_over_studies
method                                                                       
atlas_clf                                      0.755                    0.464
atlas_knn                                      0.747                    0.487
atlas_nnls                                     0.597                    0.181
combo_all                                      0.759                    0.567
combo_atlas                                    0.757                    0.530
combo_no_atlas                                 0.753                    0.546
combo_prod_atlas                               0.756                    0.550
ct_entropy                                     0.622                    0.283
ct_maxprob                                     0.605                    0.269
nnls                                           0.695                    0.367
prod                                           0.742                    0.504
prod_geno                                      0.969                    0.897
rankavg_sdf_solo_atlas_prod                    0.758                    0.552
scdblfinder                                    0.761                    0.560
scrublet                                       0.735                    0.495
solo                                           0.723                    0.432
umi                                            0.673                    0.272
vireo_geno                                     0.950                    0.842
```

## paired AUPRC vs production

```
                     method  n  auprc_delta_median  better  wilcoxon_p
                  atlas_clf 17              0.0069      11      1.0000
                  atlas_knn 17              0.0092       9      0.6777
                 atlas_nnls 17             -0.3007       0      0.0000
                 ct_entropy 17             -0.2562       0      0.0000
                 ct_maxprob 17             -0.2700       0      0.0000
                       nnls 17             -0.1737       2      0.0001
                scdblfinder 17              0.0274      16      0.0008
                   scrublet 17             -0.0588       4      0.0638
                       solo 17             -0.0988       1      0.0000
                        umi 17             -0.1808       0      0.0000
                  combo_all 17              0.0562      16      0.0000
             combo_no_atlas 17              0.0335      14      0.0013
                combo_atlas 17              0.0470      12      0.0267
           combo_prod_atlas 17              0.0603      15      0.0004
rankavg_sdf_solo_atlas_prod 17              0.0472      16      0.0000
```

## real doublets by atlas lineage (PBMC pools)

```
      dataset  doublets  one_lineage  two_lineages  singlets_one_lineage  expected_same_lineage_pairs
       kang21      1598        0.257         0.410                 0.398                        0.862
   pbmc-1A-dm       120        0.000         0.192                 0.010                        1.000
   pbmc-1B-dm       130        0.015         0.238                 0.040                        0.960
   pbmc-1C-dm       316        0.019         0.203                 0.024                        0.966
pbmc-2ctrl-dm      1598        0.043         0.213                 0.070                        0.977
pbmc-2stim-dm      1631        0.018         0.379                 0.021                        0.992
      pbmc-ch      2545        0.223         0.172                 0.298                        0.971
```

```
          method  kang21:one_lineage  kang21:two_lineages  pbmc-ch:one_lineage  pbmc-ch:two_lineages
       atlas_clf               0.543                0.679                0.636                 0.769
       atlas_knn               0.472                0.794                0.623                 0.812
      atlas_nnls               0.051                0.296                0.181                 0.233
       combo_all               0.348                0.832                0.651                 0.824
     combo_atlas               0.460                0.763                0.600                 0.826
  combo_no_atlas               0.333                0.832                0.614                 0.805
combo_prod_atlas               0.501                0.766                0.658                 0.842
      ct_entropy               0.046                0.557                0.000                 0.675
      ct_maxprob               0.146                0.351                0.033                 0.579
            nnls               0.039                0.438                0.033                 0.554
            prod               0.526                0.774                0.694                 0.767
       prod_geno               0.893                0.885                  NaN                   NaN
     scdblfinder               0.586                0.759                0.623                 0.757
        scrublet               0.299                0.748                0.273                 0.744
            solo               0.175                0.777                0.301                 0.721
             umi               0.151                0.515                0.567                 0.529
      vireo_geno               0.888                0.834                  NaN                   NaN
```
