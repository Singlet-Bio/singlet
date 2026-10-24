# SPDX-License-Identifier: MIT
"""The cell guide: a marker-gene hierarchy of cell types keyed to the Cell Ontology, compiled against one sample's genes.

The guide (513 nodes, human and mouse) and the scoring/gating method come from the cell-phenotype engine in
Singlet-Bio/singlet-agents (branch ``feat/cell-phenotype``, ``engine/cellguide`` and ``engine/python/singlet_engine/phenotype``),
where they were benchmarked on 15 labelled CELLxGENE datasets and 5 held-out ones. ``data/cellguide.json.gz`` is that guide's
compiled node list (ids, names, CL ids, parents, tissues, markers; notes and references dropped), ``data/cl.json.gz`` the Cell
Ontology release it is keyed to (names and ``is_a`` edges of the non-obsolete terms) and ``data/human_mouse.tsv.gz`` the MGI
human-mouse orthologs of the marker genes.

A compiled :class:`Guide` holds the tree (parent index per node, children, depth) and gene x node weight matrices restricted to
the genes of the sample: positive markers and negative markers, each column normalised to sum 1. Scoring every node of every
cell is then one sparse product (see :mod:`singlet.annotate.celltype`).
"""

from __future__ import annotations

import gzip
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import scipy.sparse as sps

DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=None)
def _guide_doc() -> dict:
    with gzip.open(DATA / "cellguide.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def load_nodes() -> List[dict]:
    """The guide's nodes (a fresh list of dicts), each with ``_compartment`` = its depth-0 ancestor."""
    nodes = [dict(n) for n in _guide_doc()["nodes"]]
    byid = {n["id"]: n for n in nodes}
    for n in nodes:
        c = n
        while c["parent"] != "root":
            c = byid[c["parent"]]
        n["_compartment"] = c["id"]
    return nodes


def guide_version() -> str:
    return str(_guide_doc()["guide_version"])


def cl_version() -> str:
    return str(_guide_doc()["cl_version"])


@lru_cache(maxsize=None)
def _orthologs() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with gzip.open(DATA / "human_mouse.tsv.gz", "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out.setdefault(parts[0].upper(), []).append(parts[1])
    return out


def human_to_mouse(sym: str) -> List[str]:
    """Mouse symbols for a human symbol: MGI orthologs, else the symbol in mouse capitalisation."""
    return _orthologs().get(sym.upper()) or [sym[:1].upper() + sym[1:].lower()]


# --------------------------------------------------------------------------------------------------------- tissue context
# Free-text tissue words -> coarse tissue groups. A node is allowed in a sample when it is ubiquitous ('any'), when one of its
# groups matches the sample's, when it is a circulating immune node and the sample names any tissue, or when one of its
# descendants is allowed. Unknown words fail open (every node allowed).
TISSUE_GROUPS: Dict[str, List[str]] = {
    "blood": ["blood", "cord blood", "pbmc", "peripheral blood"],
    "marrow": ["bone marrow", "marrow", "fetal liver", "yolk sac"],
    "lymphoid_organ": [
        "spleen",
        "lymph node",
        "tonsil",
        "thymus",
        "peyer",
        "appendix",
        "marginal zone",
    ],
    "lung": [
        "lung",
        "bronch",
        "trachea",
        "nose",
        "nasal",
        "airway",
        "pleura",
        "alveol",
        "respiratory",
    ],
    "kidney": [
        "kidney",
        "renal",
        "nephron",
        "glomerul",
        "ureter",
        "cortex of kidney",
        "medulla of kidney",
        "papilla",
    ],
    "bladder": ["bladder", "urethra", "urothel"],
    "liver": ["liver", "hepat", "bile duct", "gallbladder", "biliary"],
    "pancreas": ["pancrea", "islet"],
    "gut": [
        "colon",
        "intestin",
        "gut",
        "ileum",
        "jejunum",
        "duodenum",
        "rectum",
        "cecum",
        "caecum",
        "colorectal",
        "bowel",
        "mucosa of",
    ],
    "stomach": ["stomach", "gastric"],
    "esophagus": ["esophag", "oesophag"],
    "oral": ["tongue", "oral", "gingiva", "salivary", "parotid", "submandibular", "sublingual"],
    "skin": [
        "skin",
        "scalp",
        "epiderm",
        "dermis",
        "hair follicle",
        "palm",
        "sole",
        "axilla",
        "foreskin",
        "wound",
    ],
    "brain": [
        "brain",
        "cortex",
        "cortical",
        "hippocamp",
        "hypothalam",
        "cerebell",
        "striatum",
        "white matter",
        "grey matter",
        "gray matter",
        "gyrus",
        "nucleus",
        "thalam",
        "midbrain",
        "brainstem",
        "pons",
        "medulla oblongata",
        "amygdala",
        "olfactory bulb",
        "ventricle wall",
        "subventricular",
        "ventricular zone",
        "forebrain",
        "substantia nigra",
        "tegmental",
        "locus coeruleus",
        "raphe",
        "habenula",
        "choroid plexus",
        "meninges",
        "pituitary",
        "neocortex",
        "isocortex",
        "putamen",
        "caudate",
        "accumbens",
        "claustrum",
        "subiculum",
        "ganglionic eminence",
        "neural tube",
        "subpallium",
        "subplate",
        "median eminence",
        "zona incerta",
        "ca1",
        "ca2",
        "ca3",
    ],
    "spinal_cord": ["spinal cord"],
    "eye": [
        "retina",
        "eye",
        "fovea",
        "macula",
        "cornea",
        "limbus",
        "lens",
        "choroid",
        "uvea",
        "optic",
        "conjunctiva",
        "rpe",
    ],
    "pns": [
        "ganglion",
        "ganglia",
        "nerve",
        "enteric nervous",
        "myenteric",
        "submucosal plexus",
        "neural crest",
    ],
    "heart": [
        "heart",
        "cardiac",
        "atrium",
        "atrial",
        "ventricle",
        "myocard",
        "pericard",
        "aortic valve",
        "valve",
    ],
    "muscle": ["skeletal muscle", "muscle", "diaphragm", "myotendinous", "limb", "somite"],
    "adipose": ["adipose", "fat", "omentum"],
    "breast": ["breast", "mammary"],
    "prostate": ["prostat"],
    "testis": ["testis", "testes", "seminiferous", "epididym"],
    "female_reproductive": [
        "uterus",
        "uterine",
        "endometri",
        "ovary",
        "ovarian",
        "cervix",
        "ectocervix",
        "endocervix",
        "fallopian",
        "oviduct",
        "vagina",
        "decidua",
        "myometri",
        "follicle",
    ],
    "placenta": ["placenta", "trophoblast", "chorion", "decidua"],
    "thyroid": ["thyroid"],
    "adrenal": ["adrenal"],
    "bone_joint": [
        "bone",
        "cartilage",
        "synovi",
        "joint",
        "tendon",
        "ligament",
        "growth plate",
        "calvaria",
        "periosteum",
        "enthesis",
        "rib",
    ],
    "vessel": ["aorta", "artery", "arterial", "vein", "vessel", "atherosclerotic", "vascular"],
    "serosa": ["peritone", "pleural cavity", "ascites", "mesentery", "serosa", "mesotheli"],
    "embryo": ["embryo", "fetal", "foetal", "developing", "organoid"],
    "tumor": [
        "tumor",
        "tumour",
        "carcinoma",
        "cancer",
        "melanoma",
        "glioma",
        "lymphoma",
        "leukemia",
        "sarcoma",
        "neoplas",
    ],
}
# keywords match at a word start ('renal' not in 'adrenal', 'oral' not in 'temporal')
_PAT = [
    (g, re.compile("|".join(r"\b" + re.escape(k) for k in ks))) for g, ks in TISSUE_GROUPS.items()
]
_BRAIN_STRONG = re.compile(
    r"brain|cerebr|hippocamp|hypothalam|cerebell|striatum|thalam|neocortex|isocortex|gyrus|forebrain|midbrain"
)


def tissue_groups(text: Optional[str]) -> Set[str]:
    """Tissue groups named in a free-text tissue string (empty when none is recognised)."""
    t = str(text or "").lower()
    if not t or t in ("any", "unknown", "na", "nan", "none"):
        return set()
    out = {g for g, p in _PAT if p.search(t)}
    # generic anatomy words ('cortex', 'nucleus', 'ventricle') mean brain only when no other organ is named
    if "brain" in out and (out - {"brain", "embryo", "tumor"}) and not _BRAIN_STRONG.search(t):
        out.discard("brain")
    return out


def _node_groups(node: dict) -> Optional[Set[str]]:
    ts = node.get("tissues") or ["any"]
    if any(str(t).lower() == "any" for t in ts):
        return None
    out: Set[str] = set()
    for t in ts:
        out |= tissue_groups(t)
    return out or None


def allowed_nodes(
    nodes: Sequence[dict], tissue: Optional[str], disease: Optional[str] = None
) -> Set[str]:
    """Ids of the nodes allowed in a sample of ``tissue`` (free text) with ``disease`` (free text, only 'tumour' words count)."""
    sg = tissue_groups(tissue) | ({"tumor"} if "tumor" in tissue_groups(disease) else set())
    if not sg - {
        "tumor",
        "embryo",
    }:  # no organ named ('tumour', 'fetal' alone): every node stays possible
        return {n["id"] for n in nodes}
    byid = {n["id"]: n for n in nodes}
    ok: Set[str] = set()
    for n in nodes:
        ng = _node_groups(n)
        if ng is None:
            ok.add(n["id"])
            continue
        if "embryo" in ng and len(ng) > 1:
            ng = ng - {"embryo"}  # 'fetal liver' etc.: judge by the organ
        if ng & sg or ("blood" in ng and n.get("_compartment") == "immune"):
            ok.add(n["id"])
    for i in list(ok):
        p = byid[i]["parent"]
        while p != "root" and p not in ok:
            ok.add(p)
            p = byid[p]["parent"]
    return ok


# --------------------------------------------------------------------------------------------------------- compiled guide
def _colnorm(W: sps.spmatrix) -> sps.csc_matrix:
    W = sps.csc_matrix(W, dtype=np.float32, copy=True)
    s = np.asarray(W.sum(0)).ravel()
    s[s == 0] = 1
    return sps.csc_matrix(W @ sps.diags((1.0 / s).astype(np.float32)))


class Guide:
    """The guide compiled for one sample's genes. Node 0 is the virtual root (CL:0000000, 'cell')."""

    def __init__(
        self,
        gene_names: Sequence[str],
        species: str = "human",
        tissue: Optional[str] = None,
        disease: Optional[str] = None,
        nodes: Optional[Sequence[dict]] = None,
        strip_ancestor_markers: bool = True,
    ):
        self.species = species
        all_nodes = list(nodes) if nodes is not None else load_nodes()
        kept = [n for n in all_nodes if species in n.get("species", ["human", "mouse"])]
        if tissue or disease:
            ok = allowed_nodes(kept, tissue, disease)
            kept = [n for n in kept if n["id"] in ok]
        ids = {n["id"] for n in kept}
        while True:  # drop nodes whose parent was filtered out
            k = len(kept)
            kept = [n for n in kept if n["parent"] == "root" or n["parent"] in ids]
            ids = {n["id"] for n in kept}
            if len(kept) == k:
                break
        self.nodes: List[dict] = [
            dict(id="root", name="cell", cl="CL:0000000", parent=None, level=-1)
        ] + kept
        self.ids: List[str] = [n["id"] for n in self.nodes]
        self.index: Dict[str, int] = {i: j for j, i in enumerate(self.ids)}
        n_nodes = len(self.nodes)
        self.parent = np.array(
            [-1] + [self.index[n["parent"]] if n["parent"] != "root" else 0 for n in kept], np.int32
        )
        self.children: List[List[int]] = [[] for _ in self.nodes]
        for j in range(1, n_nodes):
            self.children[self.parent[j]].append(j)
        self.depth = np.zeros(n_nodes, np.int32)
        for j in self._topo():
            if j:
                self.depth[j] = self.depth[self.parent[j]] + 1
        self.gene_names = [str(g) for g in gene_names]
        gi: Dict[str, int] = {}
        for j, g in enumerate(self.gene_names):
            gi.setdefault(g.upper(), j)
        self._gi = gi
        G = len(self.gene_names)
        rows_p: List[int] = []
        cols_p: List[int] = []
        w_p: List[float] = []
        rows_n: List[int] = []
        cols_n: List[int] = []
        # A positive marker an ancestor already lists is shared by all siblings and cannot separate them.
        anc_pos: List[Set[str]] = [set() for _ in self.nodes]
        for j in self._topo():
            if j == 0:
                continue
            pj = int(self.parent[j])
            anc_pos[j] = anc_pos[pj] | (
                {g.upper() for g in self._markers(self.nodes[pj])[0]} if pj > 0 else set()
            )
        self.n_markers_found = np.zeros(n_nodes, np.int32)
        for j, n in enumerate(self.nodes[1:], start=1):
            pos, neg = self._markers(n)
            if strip_ancestor_markers:
                pos = {g: w for g, w in pos.items() if g.upper() not in anc_pos[j]}
            if n.get("gate") is False:
                pos = {}
            found = 0
            for g, w in pos.items():
                k = gi.get(g.upper())
                if k is None:
                    continue
                rows_p.append(k)
                cols_p.append(j)
                w_p.append(float(w))
                found += 1
            for g in neg:
                k = gi.get(g.upper())
                if k is not None:
                    rows_n.append(k)
                    cols_n.append(j)
            self.n_markers_found[j] = found
        self.Wpos = _colnorm(sps.csc_matrix((w_p, (rows_p, cols_p)), shape=(G, n_nodes)))
        self.Wneg = _colnorm(
            sps.csc_matrix((np.ones(len(rows_n)), (rows_n, cols_n)), shape=(G, n_nodes))
        )
        self.cl = np.array([n.get("cl") or "" for n in self.nodes], dtype=object)
        self.names = np.array([n.get("name") or n["id"] for n in self.nodes], dtype=object)

    def _markers(self, n: dict):
        m = n.get("markers") or {}
        pos = dict(m.get("pos") or {})
        neg = list(m.get("neg") or [])
        if self.species == "mouse":
            mo = n.get("mouse") or {}
            mpos: Dict[str, float] = {}
            for g, w in pos.items():
                for mg in human_to_mouse(g)[:2]:
                    mpos[mg] = w
            mpos.update(mo.get("pos") or {})
            pos = mpos
            neg = [mg for g in neg for mg in human_to_mouse(g)[:2]] + list(mo.get("neg") or [])
        return pos, neg

    def _topo(self) -> List[int]:
        order, st = [], [0]
        while st:
            j = st.pop()
            order.append(j)
            st += self.children[j]
        return order

    def gateable(self, j: int) -> List[int]:
        """Children of ``j`` that have at least one marker present in the sample (the only ones gating can enter)."""
        return [c for c in self.children[j] if self.n_markers_found[c] > 0]

    def path(self, j: int) -> List[int]:
        out = []
        while j > 0:
            out.append(int(j))
            j = int(self.parent[j])
        return out[::-1]

    def ancestor_at_depth(self, nodes: Iterable[int], d: int) -> np.ndarray:
        """Ancestor at guide depth ``d`` of each node (the node itself when it is shallower)."""
        out = []
        for j in nodes:
            p = self.path(int(j))
            out.append(p[d - 1] if len(p) >= d else int(j))
        return np.array(out, np.int32)


class Ontology:
    """Cell Ontology names and ``is_a`` ancestry (the release the guide is keyed to)."""

    def __init__(self) -> None:
        with gzip.open(DATA / "cl.json.gz", "rt", encoding="utf-8") as f:
            d = json.load(f)
        self.version: str = d["version"]
        self.terms: Dict[str, dict] = d["terms"]
        self._anc: Dict[str, Set[str]] = {}

    def name(self, t: str) -> str:
        return self.terms.get(t, {}).get("name", t)

    def ancestors(self, t: str) -> Set[str]:
        """``t`` and all its ``is_a`` ancestors."""
        hit = self._anc.get(t)
        if hit is not None:
            return hit
        out, stack = {t}, [t]
        while stack:
            for p in self.terms.get(stack.pop(), {}).get("is_a", []):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        self._anc[t] = out
        return out


@lru_cache(maxsize=1)
def ontology() -> Ontology:
    return Ontology()
