# SPDX-License-Identifier: MIT
"""The network, its loss and its calibration (PyTorch; imported only where a model is trained or run).

An MLP over the transformed vocabulary counts (:mod:`singlet.celltype_nn.features`) with a softmax over the output terms
(:class:`singlet.celltype_nn.ontology.TermSpace`). The loss of a cell with target term ``t`` is ``-log(p . credit(t))``
(expected ontology credit); a cell may carry a second, weaker target from the annotator ensemble (a refinement below the
author's term). A single temperature fitted on held-out collections calibrates the softmax.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CellTypeNet(nn.Module):
    def __init__(
        self, n_genes: int, n_terms: int, hidden=(1024, 512), dropout: float = 0.2, input_dropout: float = 0.1, transform: str = "log",
        n_context: int = 0,
    ):
        super().__init__()
        self.n_context = n_context
        self.in_drop = nn.Dropout(input_dropout)
        layers, d = [], n_genes + n_context
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_terms)
        self.register_buffer("temperature", torch.ones(()))
        self.transform = transform
        self.config = dict(n_genes=n_genes, n_terms=n_terms, hidden=list(hidden), dropout=dropout, input_dropout=input_dropout,
                           transform=transform, n_context=n_context)

    def _inputs(self, x: torch.Tensor, ctx: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.in_drop(x)
        if self.n_context:
            if ctx is None:
                ctx = torch.zeros(x.shape[0], self.n_context, device=x.device, dtype=x.dtype)
            x = torch.cat([x, ctx], 1)
        return x

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Logits (uncalibrated). ``ctx``: tissue-group multi-hot (cells x n_context); None = no context."""
        return self.head(self.body(self._inputs(x, ctx)))

    def embed(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.body(self._inputs(x, ctx))

    def proba(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.softmax(self.forward(x, ctx) / self.temperature, dim=1)


def credit_loss(logits: torch.Tensor, credit: torch.Tensor, weight: Optional[torch.Tensor] = None, eps: float = 1e-6):
    """Mean of ``-log(softmax(logits) . credit)`` over cells; ``credit`` is cells x terms (rows from TermSpace.credit)."""
    logp = F.log_softmax(logits, dim=1)
    # log sum_v p_v c_v, computed in log space for stability
    lc = torch.log(credit.clamp_min(0) + 1e-30)
    val = torch.logsumexp(logp + lc, dim=1)
    loss = -torch.clamp(val, min=np.log(eps))
    if weight is not None:
        return (loss * weight).sum() / weight.sum().clamp_min(1e-6)
    return loss.mean()


def fit_temperature(logits: torch.Tensor, credit: torch.Tensor, iters: int = 200) -> float:
    """Temperature minimising ``credit_loss`` on held-out cells (one scalar; logits are not refitted). Pass the closure
    indicator (1 on the target term and its descendants, 0 elsewhere): the loss is then the likelihood of the label's subtree,
    a proper scoring rule; partial credit for coarser terms is not, and would sharpen the softmax."""
    t = torch.zeros((), requires_grad=True, device=logits.device)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = credit_loss(logits / t.exp(), credit)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.exp().detach())


def save(model: CellTypeNet, path, extra: dict) -> None:
    torch.save(dict(state=model.state_dict(), config=model.config, **extra), path)


def load(path, map_location="cpu"):
    d = torch.load(path, map_location=map_location, weights_only=False)
    m = CellTypeNet(**d["config"])
    state = dict(d["state"])
    # models saved before the context input kept the input dropout as body.0 (no parameters), so their layers sit one higher
    if any(k.startswith("body.1.") for k in state) and not any(k.startswith("body.0.") for k in state):
        state = {
            (f"body.{int(k.split('.')[1]) - 1}." + k.split(".", 2)[2]) if k.startswith("body.") else k: v for k, v in state.items()
        }
    m.load_state_dict(state)
    m.eval()
    return m, d
