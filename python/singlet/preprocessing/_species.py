"""Species reference configuration."""

from __future__ import annotations

from typing import Any, Dict

# NCBI taxonomy ID → reference genome info
SPECIES_REF: Dict[int, Dict[str, Any]] = {
    9606: {"name": "human", "assembly": "GRCh38", "ensembl": 110},
    10090: {"name": "mouse", "assembly": "GRCm39", "ensembl": 110},
    7955: {"name": "zebrafish", "assembly": "GRCz11", "ensembl": 110},
    10116: {"name": "rat", "assembly": "mRatBN7.2", "ensembl": 110},
    9031: {"name": "chicken", "assembly": "GRCg7b", "ensembl": 110},
    9823: {"name": "pig", "assembly": "Sscrofa11.1", "ensembl": 110},
    9913: {"name": "cow", "assembly": "ARS-UCD1.2", "ensembl": 110},
    9615: {"name": "dog", "assembly": "ROS_Cfam_1.0", "ensembl": 110},
    9541: {"name": "crab-eating macaque", "assembly": "Macaca_fascicularis_6.0", "ensembl": 110},
    9544: {"name": "rhesus macaque", "assembly": "Mmul_10", "ensembl": 110},
    9796: {"name": "horse", "assembly": "EquCab3.0", "ensembl": 110},
    9685: {"name": "cat", "assembly": "Felis_catus_9.0", "ensembl": 110},
    8364: {"name": "xenopus", "assembly": "XENLA_10.1", "ensembl": 110},
    28377: {"name": "anole", "assembly": "AnoCar2.0v2", "ensembl": 110},
    13616: {"name": "opossum", "assembly": "ASM229v1", "ensembl": 110},
    7227: {"name": "drosophila", "assembly": "BDGP6.46", "ensembl": 110},
    6239: {"name": "c. elegans", "assembly": "WBcel235", "ensembl": 110},
    3702: {"name": "arabidopsis", "assembly": "TAIR10", "ensembl": 57},
    4932: {"name": "yeast", "assembly": "R64-1-1", "ensembl": 110},
    9986: {"name": "rabbit", "assembly": "OryCun2.0", "ensembl": 110},
    9940: {"name": "sheep", "assembly": "Oar_rambouillet_v1.0", "ensembl": 110},
    8090: {"name": "medaka", "assembly": "ASM223467v1", "ensembl": 110},
    7719: {"name": "ciona", "assembly": "KH", "ensembl": 110},
    9669: {"name": "ferret", "assembly": "MusPutFur1.0", "ensembl": 110},
}

# Organism name → taxonomy ID (with aliases)
ORGANISM_TO_TAXON: Dict[str, int] = {}
for _txid, _info in SPECIES_REF.items():
    ORGANISM_TO_TAXON[_info["name"]] = _txid
    ORGANISM_TO_TAXON[_info["name"].lower()] = _txid

# Common aliases
ORGANISM_TO_TAXON.update(
    {
        "homo sapiens": 9606,
        "mus musculus": 10090,
        "danio rerio": 7955,
        "rattus norvegicus": 10116,
        "gallus gallus": 9031,
        "sus scrofa": 9823,
        "bos taurus": 9913,
        "canis lupus familiaris": 9615,
        "macaca fascicularis": 9541,
        "macaca mulatta": 9544,
        "equus caballus": 9796,
        "felis catus": 9685,
        "xenopus laevis": 8364,
        "anolis carolinensis": 28377,
        "monodelphis domestica": 13616,
        "drosophila melanogaster": 7227,
        "caenorhabditis elegans": 6239,
        "arabidopsis thaliana": 3702,
        "saccharomyces cerevisiae": 4932,
        "oryctolagus cuniculus": 9986,
        "ovis aries": 9940,
        "oryzias latipes": 8090,
        "ciona intestinalis": 7719,
        "mustela putorius furo": 9669,
    }
)


def get_taxon_id(organism_name: str) -> int:
    """Look up NCBI taxonomy ID from organism name.

    Parameters
    ----------
    organism_name : str
        Common or scientific name (case-insensitive).

    Returns
    -------
    int
        NCBI taxonomy ID.

    Raises
    ------
    KeyError
        If organism not found.
    """
    key = organism_name.strip().lower()
    if key in ORGANISM_TO_TAXON:
        return ORGANISM_TO_TAXON[key]
    raise KeyError(
        f"Unknown organism: {organism_name!r}. Use list_supported_species() for available species."
    )


def get_species_info(taxon_id: int) -> Dict[str, Any]:
    """Get reference genome info for a taxonomy ID.

    Parameters
    ----------
    taxon_id : int

    Returns
    -------
    dict
        Keys: name, assembly, ensembl, (optionally index_path).

    Raises
    ------
    KeyError
        If taxonomy ID not found.
    """
    if taxon_id not in SPECIES_REF:
        raise KeyError(f"Unsupported taxonomy ID: {taxon_id}")
    return dict(SPECIES_REF[taxon_id])


def list_supported_species() -> list[dict]:
    """List all supported species.

    Returns
    -------
    list of dict
        Each dict has: taxon_id, name, assembly.
    """
    return [
        {"taxon_id": txid, "name": info["name"], "assembly": info["assembly"]}
        for txid, info in sorted(SPECIES_REF.items())
    ]
