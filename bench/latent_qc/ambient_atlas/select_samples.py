"""Pick the atlas ambient-study sample list from sample_meta.csv (metadata only). Usage: select_samples.py sample_meta.csv out.tsv"""
import re
import sys

import numpy as np
import pandas as pd

d = pd.read_csv(sys.argv[1]).fillna("")
d = d[d.sample_kind.isin(["", "primary"])]
RULES = [
    ("blood", r"blood|pbmc|mononuclear|plasma|buffy"), ("bone_marrow", r"bone marrow|hematopoietic"),
    ("lung", r"lung|respiratory|bronch|trachea|alveol|airway|pleura"), ("liver", r"liver|hepat"),
    ("pancreas", r"pancrea|islet"), ("kidney", r"kidney|renal|nephr|urine|bladder"),
    ("brain", r"brain|cortex|brodmann|hippocamp|cerebell|neocortex|frontal|cerebr|striatum|midbrain|hypothal|thalam|amygdal|spinal|ganglion|retina|meninx|leptomen|glia"),
    ("heart", r"heart|cardiac|atrium|ventric|aort|carotid|artery|vascul"),
    ("gut", r"colon|ileum|intestin|rectum|stomach|duoden|jejun|digestive|gastro|esophag|cecum|caecum|gut|appendix|gastric"),
    ("skin", r"skin|derm|epiderm|keratin|scalp|hair"), ("lymphoid", r"lymph|spleen|tonsil|thymus|adenoid"),
    ("breast", r"breast|mammary"), ("prostate", r"prostat"),
    ("reproductive", r"ovary|testis|uter|endometr|placent|decidua|cervi|fallopian|vagina|embry|fetal membrane|umbilical|oviduct"),
    ("adipose", r"adipose|omentum|fat"), ("muscle", r"muscle|skeletal|tendon|limb|bone|cartilage|joint|synov"),
    ("eye", r"eye|cornea|sclera|choroid"), ("oral_nasal", r"nasal|nose|oral|tongue|gingiv|saliv|pharyn|larynx|sinus|olfactory|ear|cochlea"),
    ("thyroid_endocrine", r"thyroid|adrenal|pituitar|parathyroid"),
]


def coarse(x):
    x = x.lower()
    for n, r in RULES:
        if re.search(r, x):
            return n
    return "other" if x else ""


d["tissue_coarse"] = d.tissue_label.map(coarse)
d = d[d.tissue_coarse != ""]
d["org"] = np.where(d.organism.str.contains("sapiens"), "human", np.where(d.organism.str.contains("musculus"), "mouse", ""))
d = d[d.org != ""]
out = []
for org, n_target, cap_t in (("human", 460, 22), ("mouse", 80, 8)):  # oversample ~1.5x: some fetches fail / wrong strand / empty
    x = d[d.org == org].sample(frac=1, random_state=1)
    picked, per_gse, per_t = [], {}, {}
    groups = {t: g.copy() for t, g in x.groupby("tissue_coarse")}
    while len(picked) < n_target and any(len(g) for g in groups.values()):
        for t in list(groups):
            g = groups[t]
            while len(g):
                r = g.iloc[0]
                g = g.iloc[1:]
                if per_gse.get(r.gse, 0) >= 3 or per_t.get(t, 0) >= cap_t:
                    continue
                picked.append(r)
                per_gse[r.gse] = per_gse.get(r.gse, 0) + 1
                per_t[t] = per_t.get(t, 0) + 1
                break
            groups[t] = g
            if len(picked) >= n_target:
                break
    out.append(pd.DataFrame(picked))
o = pd.concat(out)
o[["gsm", "gse", "org", "tissue_coarse", "tissue_label", "tissue", "disease_label", "sample_kind"]].to_csv(sys.argv[2], sep="\t", index=False)
print(o.groupby(["org", "tissue_coarse"]).size().unstack(0))
print(o.gse.nunique(), len(o))
