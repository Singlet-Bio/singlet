# SPDX-License-Identifier: MIT
"""Cell type identity probabilities: a network over Cell Ontology terms, trained on CELLxGENE Census author labels and an
ensemble of annotators, for every cell of the atlas.

::

    from singlet.celltype_nn import CellTypePredictor, write_sidecar
    pred = CellTypePredictor.load("model.pt")
    P = pred.predict(X, gene_ids, "human")             # cells x CL terms, softmax (rows sum to 1), calibrated
    write_sidecar(pred.table(P, gsm, col), f"{gsm}.celltype.parquet")

Modules: :mod:`~singlet.celltype_nn.ontology` (output terms, credit, roll-up), :mod:`~singlet.celltype_nn.features` (gene
vocabulary, transform, augmentation), :mod:`~singlet.celltype_nn.model` (network, loss, temperature; needs PyTorch),
:mod:`~singlet.celltype_nn.ensemble` (the annotators it learns from), :mod:`~singlet.celltype_nn.predict`,
:mod:`~singlet.celltype_nn.output` (the sidecar schema and its guard). ``docs/celltype_nn.md`` documents the method and its
evaluation.
"""

from .output import COLUMNS, PublicOutputError, assert_public_safe, write_sidecar
from .predict import CellTypePredictor

__version__ = "0.1.0"

__all__ = ["COLUMNS", "CellTypePredictor", "PublicOutputError", "assert_public_safe", "write_sidecar", "__version__"]
