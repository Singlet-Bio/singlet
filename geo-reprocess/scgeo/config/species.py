"""Species reference configuration.

Maps NCBI taxonomy IDs to reference genome information for
all supported species in sc-geo.
"""
from typing import Dict, Any

# All 39 supported species with reference genome information
SPECIES_REF: Dict[int, Dict[str, Any]] = {
    # ── Original 15 (indices already built) ───────────────────────────────
    9606:  {"name": "Homo sapiens",              "assembly": "GRCh38",              "annotation": "GENCODE v45"},
    10090: {"name": "Mus musculus",               "assembly": "GRCm39",              "annotation": "GENCODE vM34"},
    7955:  {"name": "Danio rerio",                "assembly": "GRCz11",              "annotation": "Ensembl 111"},
    10116: {"name": "Rattus norvegicus",           "assembly": "mRatBN7.2",           "annotation": "Ensembl 111"},
    9031:  {"name": "Gallus gallus",               "assembly": "GRCg7b",              "annotation": "Ensembl 111"},
    9823:  {"name": "Sus scrofa",                  "assembly": "Sscrofa11.1",         "annotation": "Ensembl 111"},
    9913:  {"name": "Bos taurus",                  "assembly": "ARS-UCD1.3",          "annotation": "Ensembl 111"},
    9615:  {"name": "Canis lupus familiaris",       "assembly": "ROS_Cfam_1.0",        "annotation": "Ensembl 111"},
    9541:  {"name": "Macaca fascicularis",          "assembly": "Macaca_fascicularis_6.0", "annotation": "Ensembl 111"},
    9544:  {"name": "Macaca mulatta",              "assembly": "Mmul_10",             "annotation": "Ensembl 111"},
    9796:  {"name": "Equus caballus",              "assembly": "EquCab3.0",           "annotation": "Ensembl 111"},
    9685:  {"name": "Felis catus",                 "assembly": "Felis_catus_9.0",     "annotation": "Ensembl 111"},
    8364:  {"name": "Xenopus tropicalis",          "assembly": "UCB_Xtro_10.0",       "annotation": "Ensembl 111"},
    28377: {"name": "Anolis carolinensis",         "assembly": "AnoCar2.0",           "annotation": "Ensembl 111"},
    13616: {"name": "Monodelphis domestica",       "assembly": "ASM229v1",            "annotation": "Ensembl 111"},
    # ── New species (indices to be built) ─────────────────────────────────
    7227:  {"name": "Drosophila melanogaster",     "assembly": "BDGP6.46",            "annotation": "Ensembl 111"},
    6239:  {"name": "Caenorhabditis elegans",      "assembly": "WBcel235",            "annotation": "Ensembl 111"},
    3702:  {"name": "Arabidopsis thaliana",        "assembly": "TAIR10",              "annotation": "Ensembl Plants 59"},
    4932:  {"name": "Saccharomyces cerevisiae",    "assembly": "R64-1-1",             "annotation": "Ensembl 111"},
    9986:  {"name": "Oryctolagus cuniculus",       "assembly": "OryCun2.0",           "annotation": "Ensembl 111"},
    9940:  {"name": "Ovis aries",                  "assembly": "ARS-UI_Ramb_v2.0",    "annotation": "Ensembl 111", "ensembl_name": "Ovis aries rambouillet", "ensembl_dir": "ovis_aries"},
    8090:  {"name": "Oryzias latipes",             "assembly": "ASM223467v1",         "annotation": "Ensembl 111"},
    7719:  {"name": "Ciona intestinalis",          "assembly": "KH",                  "annotation": "Ensembl 111"},
    9669:  {"name": "Mustela putorius furo",       "assembly": "MusPutFur1.0",        "annotation": "Ensembl 111"},
    # ── High-volume unmatched species ─────────────────────────────────────
    8296:  {"name": "Ambystoma mexicanum",         "assembly": "AmexG_v6.0-DD",       "annotation": "Ensembl 111"},
    4577:  {"name": "Zea mays",                    "assembly": "Zm-B73-REFERENCE-NAM-5.0", "annotation": "Ensembl Plants 59"},
    8319:  {"name": "Pleurodeles waltl",           "assembly": "pwalHiC1",            "annotation": "Ensembl 111"},
    # ── Exotic/diverse species ────────────────────────────────────────────
    5833:  {"name": "Plasmodium falciparum",       "assembly": "ASM276v2",           "annotation": "Ensembl Protists 59"},
    79327: {"name": "Schmidtea mediterranea",      "assembly": "ASM2634887v1",       "annotation": "Ensembl Metazoa 59"},
    6183:  {"name": "Schistosoma mansoni",         "assembly": "Smansoni_v7",        "annotation": "Ensembl Metazoa 59"},
    105023:{"name": "Nothobranchius furzeri",      "assembly": "Nfu_20140520",       "annotation": "Ensembl 111"},
    44689: {"name": "Dictyostelium discoideum",    "assembly": "dicty_2.7",          "annotation": "Ensembl Protists 59"},
    10141: {"name": "Cavia porcellus",             "assembly": "Cavpor3.0",          "annotation": "Ensembl 111"},
    7652:  {"name": "Lytechinus variegatus",       "assembly": "Lvar_3.0",           "annotation": "Ensembl Metazoa 59", "ensembl_name": "Lytechinus variegatus gca018143015v1"},
    5691:  {"name": "Trypanosoma brucei brucei",   "assembly": "TryBru_Apr2005_chr11", "annotation": "Ensembl Protists 59", "ensembl_name": "Trypanosoma brucei"},
    10036: {"name": "Mesocricetus auratus",        "assembly": "MesAur1.0",          "annotation": "Ensembl 111"},
    6500:  {"name": "Aplysia californica",         "assembly": "AplCal3.0",          "annotation": "Ensembl Metazoa 59", "ensembl_name": "Aplysia californica gca000002075v2"},
    9598:  {"name": "Pan troglodytes",             "assembly": "Pan_tro_3.0",        "annotation": "Ensembl 111"},
    9483:  {"name": "Callithrix jacchus",          "assembly": "mCalJac1.pat.X",     "annotation": "Ensembl 111"},
    # ── Vertebrates (Ensembl 113) ─────────────────────────────────────────
    7994:  {"name": "Astyanax mexicanus",          "assembly": "Astyanax_mexicanus-2.0", "annotation": "Ensembl 113"},
    10029: {"name": "Cricetulus griseus",           "assembly": "CriGri-PICRH-1.0",   "annotation": "Ensembl 113", "ensembl_name": "Cricetulus griseus picr"},
    9925:  {"name": "Capra hircus",                "assembly": "ARS1",                "annotation": "Ensembl 113"},
    51511: {"name": "Ciona savignyi",              "assembly": "CSAV2.0",             "annotation": "Ensembl 113"},
    37347: {"name": "Tupaia belangeri",             "assembly": "TREESHREW",            "annotation": "Ensembl 113"},
    52904: {"name": "Scophthalmus maximus",         "assembly": "ASM1334776v1",        "annotation": "Ensembl 113"},
    79684: {"name": "Microtus ochrogaster",         "assembly": "MicOch1.0",           "annotation": "Ensembl 113"},
    # ── Non-vertebrates (Ensembl Genomes 59) ──────────────────────────────
    7460:  {"name": "Apis mellifera",              "assembly": "Amel_HAv3.1",         "annotation": "Ensembl Metazoa 59"},
    45351: {"name": "Nematostella vectensis",      "assembly": "ASM20922v1",          "annotation": "Ensembl Metazoa 59"},
    3847:  {"name": "Glycine max",                 "assembly": "Glycine_max_v2.1",    "annotation": "Ensembl Plants 59"},
    4530:  {"name": "Oryza sativa",                "assembly": "IRGSP-1.0",           "annotation": "Ensembl Plants 59", "ensembl_name": "Oryza sativa japonica group"},
    4558:  {"name": "Sorghum bicolor",             "assembly": "Sorghum_bicolor_NCBIv3", "annotation": "Ensembl Plants 59"},
    7668:  {"name": "Strongylocentrotus purpuratus", "assembly": "Spur_5.0",           "annotation": "Ensembl Metazoa 59"},
    5811:  {"name": "Toxoplasma gondii",           "assembly": "TGME49",              "annotation": "Ensembl Protists 59"},
    3880:  {"name": "Medicago truncatula",          "assembly": "MedtrA17_4.0",        "annotation": "Ensembl Plants 59"},
    3694:  {"name": "Populus trichocarpa",          "assembly": "Pop_tri_v3",          "annotation": "Ensembl Plants 59"},
    6253:  {"name": "Ascaris suum",                "assembly": "ASM18702v3",          "annotation": "Ensembl Metazoa 59"},
    27461: {"name": "Mnemiopsis leidyi",           "assembly": "MneLei_Aug2011",      "annotation": "Ensembl Metazoa 59"},
}

# Reverse lookup: lowercase organism name → taxon_id
ORGANISM_TO_TAXON: Dict[str, int] = {
    info["name"].lower(): tid for tid, info in SPECIES_REF.items()
}

# Add common aliases and alternate names
ORGANISM_TO_TAXON.update({
    "ciona robusta": 7719,
    "saccharomyces cerevisiae by4741": 4932,
    "plasmodium falciparum nf54": 5833,
    "trypanosoma brucei": 5691,
    "pan troglodytes verus": 9598,
    "pan troglodytes schweinfurthii": 9598,
    "pan troglodytes troglodytes": 9598,
    "golden hamster": 10036,
    "sus scrofa domesticus": 9823,
    "mus": 10090,
    "mus musculus musculus x mus musculus castaneus": 10090,
    "musculus": 10090,
    "dictyostelium discoideum ax4": 44689,
    "mustela putorius furo; mustela putorius": 9669,
    "xenopus laevis": 8364,
    "chlorocebus aethiops": 9541,
    "macaca nemestrina": 9544,
    # New species aliases
    "chinese hamster": 10029,
    "cricetulus griseus chok1gshd": 10029,
    "goat": 9925,
    "tree shrew": 37347,
    "tupaia belangeri chinensis": 37347,
    "turbot": 52904,
    "prairie vole": 79684,
    "cave fish": 7994,
    "honeybee": 7460,
    "honey bee": 7460,
    "soybean": 3847,
    "rice": 4530,
    "oryza sativa japonica group": 4530,
    "oryza sativa indica group": 4530,
    "sorghum": 4558,
    "sea urchin": 7668,
    "barrel clover": 3880,
    "poplar": 3694,
    "comb jelly": 27461,
    "sea anemone": 45351,
    "starlet sea anemone": 45351,
})


def get_taxon_id(organism_name: str) -> int:
    """Get taxon ID from organism name (case-insensitive lookup).
    
    Handles semicolon-separated multi-species strings by trying each
    component (e.g. 'blank sample; Homo sapiens; Mus musculus').
    Returns the first matching taxon ID.
    """
    # Try exact match first
    result = ORGANISM_TO_TAXON.get(organism_name.lower().strip())
    if result is not None:
        return result
    # Try each semicolon-separated component
    if ';' in organism_name:
        for part in organism_name.split(';'):
            part = part.strip().lower()
            if part and part not in ('blank sample', 'blank', 'unknown', 'other', 'na', 'n/a'):
                result = ORGANISM_TO_TAXON.get(part)
                if result is not None:
                    return result
    return None


def get_species_info(taxon_id: int) -> Dict[str, Any]:
    """Get species reference information by taxon ID."""
    return SPECIES_REF.get(taxon_id)


def list_supported_species() -> list:
    """Return list of all supported species."""
    return [
        {
            "taxon_id": tid,
            "name": info["name"],
            "assembly": info["assembly"],
            "annotation": info["annotation"],
        }
        for tid, info in sorted(SPECIES_REF.items(), key=lambda x: x[1]["name"])
        if not info.get("skip", False)
    ]
