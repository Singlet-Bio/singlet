# PacBio genotype labels for the B6 held-out verdict (ORACLES_B6_B8.md, "Label amendment")
import csv, gzip, sys, collections
kit = len(sys.argv) > 3 and sys.argv[3] == "--kit"   # fusion-independent: Ku812 = KIT wild-type reads only, HMC1.2 = any KIT mutant read
rows = csv.DictReader(gzip.open(sys.argv[1], "rt"))
n = collections.Counter()
with open(sys.argv[2], "w") as o:
    o.write("barcode\tline\n")
    for r in rows:
        ba = float(r["BCR_ABL1"] or 0); km = float(r["KIT_c2447_MT"] or 0) + float(r["KIT_c1679_MT"] or 0)
        kw = float(r["KIT_c2447_WT"] or 0) + float(r["KIT_c1679_WT"] or 0)
        if kit: lab = "Ku812" if kw > 0 and km == 0 else "HMC1.2" if km > 0 else None
        else: lab = "Ku812" if ba > 0 and km == 0 else "HMC1.2" if km > 0 and ba == 0 else None
        if lab:
            o.write(r["Cell_barcode"].split("-")[0] + "\t" + lab + "\n"); n[lab] += 1
print(dict(n))
