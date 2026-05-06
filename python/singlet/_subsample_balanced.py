"""Balanced subsampling for AnnData objects.

Provides singlet.subsample_balanced() — subsample cells ensuring equal
representation from each group, useful for training classifiers or
creating balanced visualizations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def subsample_balanced(
    adata: AnnData,
    groupby: str | list[str],
    *,
    n_per_group: int | None = None,
    frac: float | None = None,
    random_state: int = 0,
    copy: bool = True,
) -> AnnData:
    """Subsample cells with balanced representation across groups.

    Ensures equal (or proportional) representation from each group,
    preventing overrepresentation of large clusters in downstream analyses.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    groupby : str or list of str
        Key(s) in adata.obs to group by. If a list, stratifies by
        the combination of all specified keys.
    n_per_group : int or None, default None
        Number of cells to sample per group. If a group has fewer cells,
        all cells from that group are kept (no upsampling).
        Mutually exclusive with frac.
    frac : float or None, default None
        Fraction of cells to keep per group (0 < frac <= 1).
        Mutually exclusive with n_per_group.
    random_state : int, default 0
        Random seed for reproducibility.
    copy : bool, default True
        If True, returns a subsampled copy. If False, subsets in place.

    Returns
    -------
    anndata.AnnData or None
        Subsampled AnnData (if copy=True), or None (if copy=False,
        modifies adata in place).

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> balanced = singlet.subsample_balanced(adata, groupby='cell_type', n_per_group=100)
    >>> balanced.obs['cell_type'].value_counts().max()  # at most 100 per type
    100
    """
    import numpy as np

    if n_per_group is not None and frac is not None:
        msg = "Specify either n_per_group or frac, not both."
        raise ValueError(msg)

    if n_per_group is None and frac is None:
        msg = "Must specify either n_per_group or frac."
        raise ValueError(msg)

    if frac is not None and not (0 < frac <= 1):
        msg = f"frac must be in (0, 1], got {frac}"
        raise ValueError(msg)

    if n_per_group is not None and n_per_group < 1:
        msg = f"n_per_group must be >= 1, got {n_per_group}"
        raise ValueError(msg)

    # Handle single or multiple groupby keys
    if isinstance(groupby, str):
        keys = [groupby]
    else:
        keys = list(groupby)

    for key in keys:
        if key not in adata.obs.columns:
            msg = f"groupby key {key!r} not found in adata.obs"
            raise KeyError(msg)

    # Create combined group labels for stratification
    if len(keys) == 1:
        groups = adata.obs[keys[0]]
    else:
        # Combine multiple keys into a single grouping
        groups = adata.obs[keys].apply(lambda row: "|".join(str(v) for v in row), axis=1)

    rng = np.random.default_rng(random_state)
    selected_indices = []

    for _group_name, group_idx in groups.groupby(groups, observed=True).groups.items():
        group_indices = np.asarray(group_idx)
        n_group = len(group_indices)

        if n_per_group is not None:
            n_take = min(n_per_group, n_group)
        else:
            n_take = max(1, int(n_group * frac))
            n_take = min(n_take, n_group)

        if n_take >= n_group:
            selected_indices.append(group_indices)
        else:
            chosen = rng.choice(group_indices, size=n_take, replace=False)
            selected_indices.append(chosen)

    # Combine and sort indices for consistent ordering
    all_indices = np.concatenate(selected_indices)
    # Convert to positional indices
    obs_index = adata.obs.index
    positional = np.array([obs_index.get_loc(idx) for idx in all_indices])
    positional.sort()

    if copy:
        return adata[adata.obs.index[positional]].copy()
    else:
        adata._inplace_subset_obs(positional)
        return None
