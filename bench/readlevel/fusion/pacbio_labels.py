# PacBio genotype labels for the B6 held-out verdict (ORACLES_B6_B8.md, "Label amendment")
import csv, gzip, sys, collections
rows = csv.DictReader(gzip.open(sys.argv[1], "rt"))
n = collections.Counter()
with open(sys.argv[2], "w") as o:
    o.write("barcode\tline\n")
    for r in rows:
        ba = float(r["BCR_ABL1"] or 0); km = float(r["KIT_c2447_MT"] or 0) + float(r["KIT_c1679_MT"] or 0)
        lab = "Ku812" if ba > 0 and km == 0 else "HMC1.2" if km > 0 and ba == 0 else None
        if lab:
            o.write(r["Cell_barcode"].split("-")[0] + "\t" + lab + "\n"); n[lab] += 1
print(dict(n))
