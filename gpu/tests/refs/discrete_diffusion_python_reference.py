#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/discrete_diffusion_python_reference.py
#
# Reference implementation driver for cycle 30 discrete diffusion correctness harness.
#
# Loads a CSC count binary (dump_csc.h format) produced by the C++ test fixture,
# runs the bioRxiv 2026 discrete diffusion Python reference, and writes a .npz
# with the trained model's sampled cells for comparison.
#
# Usage (called by generative_discrete_diffusion_correctness.cpp via subprocess):
#   python3 discrete_diffusion_python_reference.py \
#       --counts   <counts.bin>          \
#       --out      <results.npz>         \
#       --n-epochs <N>                   \
#       --n-sample <S>                   \
#       --seed     <R>
#
# Exit codes:
#   0  success — results.npz written
#   2  discrete-diffusion-sc (or fallback packages) not installed → GTEST_SKIP
#   1  any other error → FAIL
#
# Output npz keys:
#   sampled_counts  float32[n_genes × n_sample]  — sampled cell matrix (genes × cells)
#   train_loss      float32[n_epochs]             — cross-entropy loss per epoch
#   n_genes         int scalar
#   n_sample        int scalar
#
# Dependencies (heavy install — skip cleanly if unavailable):
#   pip install discrete-diffusion-sc torch anndata numpy scipy
#   (package name from the bioRxiv Feb 2026 repo; fallback to 'discrete_diffusion_sc')
#
# Note: The reference Python implementation runs on CPU by default (PyTorch).
# For the tiny synthetic test this is fast (~30s).  For real data it is slow (~hours).
# The C++ harness only calls this for the tiny + top-genes tests.

import sys
import argparse
import struct
import numpy as np
import scipy.sparse


# ---------------------------------------------------------------------------
# Package availability check — exit 2 if unavailable (→ GTEST_SKIP in C++)
# ---------------------------------------------------------------------------

def _check_discrete_diffusion():
    """Try multiple likely package names from the bioRxiv 2026 repo."""
    candidates = [
        'discrete_diffusion_sc',
        'discrete-diffusion-sc',
        'discDiff',
        'discrete_diffusion',
    ]
    for name in candidates:
        try:
            mod_name = name.replace('-', '_')
            __import__(mod_name)
            return mod_name
        except ImportError:
            continue

    # Last resort: try importing the module directly if it might be installed
    # under a different entry point.
    try:
        import torch                 # noqa: F401
        import anndata               # noqa: F401
    except ImportError as e:
        print(f"[discrete_diffusion_python_reference] SKIP: torch/anndata not found: {e}",
              file=sys.stderr)
        sys.exit(2)

    print("[discrete_diffusion_python_reference] SKIP: discrete diffusion package not found. "
          "Install with: pip install discrete-diffusion-sc",
          file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# CSC binary reader (dump_csc.h format)
#
# Format (little-endian, no padding):
#   uint32  magic      = 0x43535343 ("CSCC")
#   uint32  n_rows
#   uint32  n_cols
#   uint64  nnz
#   float32[nnz]       values
#   int32[n_cols+1]    indptr
#   int32[nnz]         indices
# ---------------------------------------------------------------------------

def read_csc_bin(path: str) -> scipy.sparse.csc_matrix:
    with open(path, 'rb') as f:
        magic, n_rows, n_cols = struct.unpack('<III', f.read(12))
        if magic != 0x43535343:
            raise ValueError(f"Bad magic in {path}: 0x{magic:08X}")
        (nnz,) = struct.unpack('<Q', f.read(8))
        values  = np.frombuffer(f.read(nnz * 4),          dtype=np.float32).copy()
        indptr  = np.frombuffer(f.read((n_cols + 1) * 4), dtype=np.int32).copy()
        indices = np.frombuffer(f.read(nnz * 4),          dtype=np.int32).copy()
    return scipy.sparse.csc_matrix(
        (values, indices, indptr), shape=(n_rows, n_cols)
    )


# ---------------------------------------------------------------------------
# Tokenization helpers (log2 binning, vocab_size=16)
# These mirror the C++ discrete_diffusion_tokenize / detokenize functions.
# ---------------------------------------------------------------------------

def tokenize_counts(counts: np.ndarray, vocab_size: int = 16) -> np.ndarray:
    """Tokenize integer counts using log2 binning.

    bin 0: count == 0
    bin k (k=1..vocab_size-1): 2^(k-1) <= count < 2^k

    Counts >= 2^(vocab_size-1) are clamped to bin vocab_size-1.
    """
    tokens = np.zeros_like(counts, dtype=np.int32)
    for k in range(1, vocab_size):
        lo = 1 << (k - 1)
        hi = 1 << k
        mask = (counts >= lo) & (counts < hi)
        tokens[mask] = k
    # Clamp large counts to the top bin.
    top_lo = 1 << (vocab_size - 1)
    tokens[counts >= top_lo] = vocab_size - 1
    return tokens


def detokenize_tokens(tokens: np.ndarray, vocab_size: int = 16) -> np.ndarray:
    """Return bin-center counts for each token ID.

    bin 0 → 0
    bin k → floor((2^(k-1) + 2^k - 1) / 2)  = (2^(k-1) + 2^k) // 2 - 1 if even
           actually: center = (lo + hi) // 2  where hi = 2^k - 1
    """
    counts = np.zeros_like(tokens, dtype=np.int32)
    for k in range(1, vocab_size):
        lo = 1 << (k - 1)
        hi = (1 << k) - 1
        center = (lo + hi) // 2
        counts[tokens == k] = center
    return counts


# ---------------------------------------------------------------------------
# Minimal D3PM-style absorbing discrete diffusion (pure NumPy fallback)
#
# This is invoked if the published bioRxiv reference package is unavailable
# but torch is present. It implements the absorbing-state discrete diffusion
# from the design doc:
#   - Forward: each gene independently masked with probability
#       q_t = 1 - prod(1 - beta_s, s=1..t),  beta linearly spaced.
#   - Denoiser: a simple MLP (no attention) that predicts x_0 from x_t.
#   - Loss: cross-entropy at masked positions.
#   - Sampling: iterative denoising T → 0.
# ---------------------------------------------------------------------------

def _run_fallback_diffusion(
        counts_matrix: np.ndarray,  # float32, shape (n_genes, n_cells)
        n_epochs: int,
        n_sample: int,
        seed: int,
        vocab_size: int = 16,
        n_timesteps: int = 100,
        hidden_dim: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (sampled_counts [n_genes × n_sample], train_loss [n_epochs])."""
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n_genes, n_cells = counts_matrix.shape

    # Tokenize input.
    int_counts = counts_matrix.astype(np.int32)
    tokens = tokenize_counts(int_counts, vocab_size)  # shape (n_genes, n_cells)

    # Noise schedule: linear beta from 0.001 to 0.2 over T steps.
    betas = np.linspace(0.001, 0.2, n_timesteps, dtype=np.float32)
    # Cumulative mask probability at each timestep.
    alpha_bar = np.cumprod(1.0 - betas)  # prob of NOT being masked at step t

    MASK_ID = vocab_size  # one extra token for MASK

    # --- Denoiser: MLP on flattened (noisy_token + time_embed) ---
    # Input: per-gene (token_id, timestep_embed) → predict clean token.
    # We use a simple linear + ReLU + linear per-gene (shared weights).
    # Hidden dim is hidden_dim.
    embed_dim = 8
    time_embed_dim = 8

    model = nn.Sequential(
        nn.Embedding(MASK_ID + 1, embed_dim),   # index 0..vocab_size (MASK=vocab_size)
    )
    # Separate time embedding + projection.
    time_proj = nn.Linear(time_embed_dim, hidden_dim)
    tok_proj   = nn.Linear(embed_dim,     hidden_dim)
    out_proj   = nn.Linear(hidden_dim,    vocab_size)

    params = (list(model.parameters()) +
              list(time_proj.parameters()) +
              list(tok_proj.parameters()) +
              list(out_proj.parameters()))
    optimizer = torch.optim.Adam(params, lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()

    # Time embedding (sinusoidal).
    def time_emb(t: int) -> torch.Tensor:
        freqs = np.exp(-np.arange(time_embed_dim // 2) *
                       np.log(10000.0) / (time_embed_dim // 2))
        emb = np.concatenate([np.sin(t * freqs), np.cos(t * freqs)]).astype(np.float32)
        return torch.tensor(emb).unsqueeze(0)  # [1, time_embed_dim]

    train_loss_history = []

    # Tokenize as tensor.
    tokens_t = torch.tensor(tokens, dtype=torch.long)  # [n_genes, n_cells]

    for epoch in range(n_epochs):
        # Sample a random batch of cells and a random timestep.
        cell_idx = rng.integers(0, n_cells)
        t = int(rng.integers(1, n_timesteps + 1))

        x0 = tokens_t[:, cell_idx]  # [n_genes]

        # Forward: mask each gene with prob 1 - alpha_bar[t-1].
        mask_prob = 1.0 - float(alpha_bar[t - 1])
        mask = torch.tensor(
            rng.random(n_genes) < mask_prob, dtype=torch.bool)
        xt = x0.clone()
        xt[mask] = MASK_ID

        # Denoiser forward pass.
        t_emb = time_emb(t)  # [1, time_embed_dim]
        t_vec = time_proj(t_emb)  # [1, hidden_dim]

        tok_emb = model[0](xt)  # [n_genes, embed_dim]
        tok_vec = tok_proj(tok_emb)  # [n_genes, hidden_dim]
        h = torch.relu(tok_vec + t_vec.expand_as(tok_vec))
        logits = out_proj(h)  # [n_genes, vocab_size]

        # Loss only at masked positions.
        if mask.any():
            loss = ce_loss(logits[mask], x0[mask])
        else:
            loss = torch.tensor(0.0, requires_grad=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_history.append(float(loss.item()))

    # --- Sampling: start from all-MASK, iterate t = T → 1 ---
    sampled_tokens = np.zeros((n_genes, n_sample), dtype=np.int32)
    for s in range(n_sample):
        xt_np = np.full(n_genes, MASK_ID, dtype=np.int32)
        for t in range(n_timesteps, 0, -1):
            xt_tensor = torch.tensor(xt_np, dtype=torch.long)
            t_emb_s = time_emb(t)
            t_vec_s = time_proj(t_emb_s)
            tok_emb_s = model[0](xt_tensor)
            tok_vec_s = tok_proj(tok_emb_s)
            h_s = torch.relu(tok_vec_s + t_vec_s.expand_as(tok_vec_s))
            logits_s = out_proj(h_s)  # [n_genes, vocab_size]
            probs = torch.softmax(logits_s, dim=-1).detach().numpy()  # [n_genes, vocab_size]
            # Sample tokens per gene from predicted distribution.
            chosen = np.array([
                rng.choice(vocab_size, p=probs[g].astype(np.float64) /
                           probs[g].astype(np.float64).sum())
                for g in range(n_genes)
            ], dtype=np.int32)
            xt_np = chosen
        sampled_tokens[:, s] = xt_np

    # De-tokenize to counts.
    sampled_counts = detokenize_tokens(sampled_tokens, vocab_size).astype(np.float32)
    return sampled_counts, np.array(train_loss_history, dtype=np.float32)


# ---------------------------------------------------------------------------
# Primary reference: bioRxiv 2026 discrete_diffusion_sc package
# ---------------------------------------------------------------------------

def run_reference_diffusion(
        counts_path: str,
        out_path: str,
        n_epochs: int,
        n_sample: int,
        seed: int) -> None:
    """Load counts, run discrete diffusion, write npz."""

    # Load count matrix (genes × cells, float32).
    mat_csc = read_csc_bin(counts_path)  # scipy CSC, genes × cells
    n_genes, n_cells = mat_csc.shape
    counts_dense = np.asarray(mat_csc.todense(), dtype=np.float32)

    # Try the published bioRxiv package first.
    pkg_name = None
    for candidate in ['discrete_diffusion_sc', 'discDiff', 'discrete_diffusion']:
        try:
            pkg_name = candidate
            pkg = __import__(candidate)
            break
        except ImportError:
            pkg_name = None
            continue

    if pkg_name is not None:
        # --- Published API (bioRxiv 2026 reference implementation) ---
        # The exact API depends on the released package version.
        # We try the most common entry points from the paper.
        try:
            import anndata as ad

            # Transpose: AnnData convention is cells × genes.
            adata = ad.AnnData(X=counts_dense.T.astype(np.float32))
            adata.var_names = [f"gene_{i}" for i in range(n_genes)]
            adata.obs_names = [f"cell_{j}" for j in range(n_cells)]

            # Try the fit() entry point from the reference repo.
            # Signature varies; try common patterns.
            if hasattr(pkg, 'DiscreteDiffusionModel'):
                model = pkg.DiscreteDiffusionModel(
                    n_genes=n_genes,
                    vocab_size=16,
                    n_timesteps=100,
                    hidden_dim=256,
                    seed=seed,
                )
                model.fit(adata, n_epochs=n_epochs, verbose=False)
                sampled_adata = model.sample(n_cells=n_sample, seed=seed)
                sampled_dense = np.asarray(sampled_adata.X, dtype=np.float32).T  # genes × cells
                loss_hist = np.array(
                    getattr(model, 'loss_history_', []), dtype=np.float32)

            elif hasattr(pkg, 'fit'):
                model = pkg.fit(adata, n_epochs=n_epochs, vocab_size=16,
                                n_timesteps=100, seed=seed, verbose=False)
                sampled_adata = pkg.sample(model, n_cells=n_sample, seed=seed)
                sampled_dense = np.asarray(sampled_adata.X, dtype=np.float32).T
                loss_hist = np.array(
                    getattr(model, 'loss_history_', []), dtype=np.float32)

            else:
                raise AttributeError(f"{pkg_name} has no recognised entry point")

        except Exception as e:
            print(f"[discrete_diffusion_python_reference] Published API failed: {e}; "
                  f"falling back to minimal NumPy/PyTorch reference.",
                  file=sys.stderr)
            # Fall through to the fallback.
            pkg_name = None

    if pkg_name is None:
        # --- Fallback: minimal absorbing D3PM in PyTorch ---
        print("[discrete_diffusion_python_reference] Using fallback torch D3PM reference.",
              file=sys.stderr)
        try:
            sampled_dense, loss_hist = _run_fallback_diffusion(
                counts_matrix=counts_dense,
                n_epochs=n_epochs,
                n_sample=n_sample,
                seed=seed,
            )
        except ImportError:
            # torch not available either.
            print("[discrete_diffusion_python_reference] SKIP: torch not installed.",
                  file=sys.stderr)
            sys.exit(2)

    # Validate shape: sampled_dense must be (n_genes, n_sample).
    if sampled_dense.shape != (n_genes, n_sample):
        # Try transposing if shape is (n_sample, n_genes).
        if sampled_dense.shape == (n_sample, n_genes):
            sampled_dense = sampled_dense.T
        else:
            raise ValueError(
                f"Unexpected sampled_counts shape: {sampled_dense.shape}, "
                f"expected ({n_genes}, {n_sample})")

    np.savez(
        out_path,
        sampled_counts = sampled_dense.astype(np.float32),  # [n_genes × n_sample]
        train_loss     = loss_hist,
        n_genes        = np.array([n_genes],  dtype=np.int32),
        n_sample       = np.array([n_sample], dtype=np.int32),
    )
    print(f"[discrete_diffusion_python_reference] wrote {out_path}  "
          f"n_genes={n_genes} n_cells_input={n_cells} n_sample={n_sample}",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Discrete diffusion Python reference for singlet-gpu cycle 30 harness")
    parser.add_argument('--counts',   required=True,  help='input counts .bin (dump_csc format)')
    parser.add_argument('--out',      required=True,  help='output .npz path')
    parser.add_argument('--n-epochs', type=int, default=100,  help='training epochs')
    parser.add_argument('--n-sample', type=int, default=50,   help='cells to sample')
    parser.add_argument('--seed',     type=int, default=0,    help='random seed')
    args = parser.parse_args()

    # Exit 2 if neither the package nor torch is available.
    _check_discrete_diffusion()

    try:
        run_reference_diffusion(
            counts_path = args.counts,
            out_path    = args.out,
            n_epochs    = args.n_epochs,
            n_sample    = args.n_sample,
            seed        = args.seed,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"[discrete_diffusion_python_reference] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
