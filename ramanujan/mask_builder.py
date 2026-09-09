"""
mask_builder.py
---------------
Constructs biregular sparse masks via successive random perfect matchings.

Algorithm: build the (d_L, d_R)-biregular mask as d_L successive matchings.
Each pass k finds a random perfect matching in the bipartite graph:
  Left:  all out nodes (each needs exactly 1 new edge this pass)
  Right: in_ nodes with remaining capacity col_remaining[j] = d_R - (edges so far)

Adjacency lists are built with right nodes ordered by remaining capacity
descending (prevents early depletion), then shuffled within capacity tiers
for randomness. Matching via augmenting-path DFS.

Correctness: Hall's condition holds at every pass when out ≤ in, because
total remaining right capacity == out * (d_L - k) >= out.

Rectangular case (out > in): the successive-matching decomposition requires
|left| ≤ |right| per pass. When out > in we instead build the transposed
mask (in, out) with degrees (d_R, d_L) — where in ≤ out ensures Hall's
condition — then transpose the result. The returned mask has shape (out, in)
with the correct (d_L, d_R)-biregularity.

Complexity: O(d_min * max(out,in) * d_min^2) per build, where
d_min = min(d_L, d_R). Cache via MaskRegistry for repeated shapes.
"""

import numpy as np
import torch
from typing import List

from ramanujan.sparsity_grid import DegreeConfig


def build_biregular_mask(config: DegreeConfig, seed: int = 0) -> torch.Tensor:
    """
    Build a boolean mask of shape (out_features, in_features) that is
    (d_L, d_R)-biregular. Each row has exactly d_L True entries.
    Each column has exactly d_R True entries.

    Handles rectangular shapes (out ≠ in) via the transpose trick:
    when out > in, builds the transposed mask and returns mask.T.
    """
    mask_np = _build_mask_np(config, seed)
    return torch.from_numpy(mask_np)


def _build_mask_np(config: DegreeConfig, seed: int) -> np.ndarray:
    out, in_ = config.out_features, config.in_features
    d_L, d_R = config.d_L, config.d_R

    # Transpose trick: successive matchings require left_size <= right_size.
    # When out > in, build the (in, out) mask with degrees (d_R, d_L) and
    # transpose. Both masks represent the same bipartite graph.
    if out > in_:
        transposed_config = DegreeConfig(
            out_features=in_,
            in_features=out,
            d_L=d_R,
            d_R=d_L,
        )
        return _build_square_or_wide_mask(transposed_config, seed).T

    return _build_square_or_wide_mask(config, seed)


def _build_square_or_wide_mask(config: DegreeConfig, seed: int) -> np.ndarray:
    """
    Core construction: requires out_features <= in_features.
    Each of d_L passes finds a perfect matching of all out_features left nodes
    into in_features right nodes. Hall's condition holds because at pass k the
    total right capacity = out * (d_L - k) >= out.
    """
    out, in_ = config.out_features, config.in_features
    d_L, d_R = config.d_L, config.d_R

    assert out <= in_, (
        f"_build_square_or_wide_mask requires out <= in, got ({out}, {in_}). "
        "Use _build_mask_np which applies the transpose trick automatically."
    )

    rng = np.random.default_rng(seed)
    mask = np.zeros((out, in_), dtype=bool)

    for pass_k in range(d_L):
        col_remaining = d_R - mask.sum(axis=0)
        available_right = np.where(col_remaining > 0)[0]

        adj: List[List[int]] = []
        for i in range(out):
            valid = available_right[~mask[i, available_right]]
            order = np.argsort(-col_remaining[valid])
            shuffled = valid[order]
            if len(shuffled) > 0:
                caps = col_remaining[shuffled]
                for cap_val in np.unique(caps):
                    tier_idx = np.where(caps == cap_val)[0]
                    tier_slice = shuffled[tier_idx].copy()
                    rng.shuffle(tier_slice)
                    shuffled[tier_idx] = tier_slice
            adj.append(shuffled.tolist())

        assignment = _augmenting_path_matching(adj, out, in_)

        if -1 in assignment:
            unmatched = [i for i, j in enumerate(assignment) if j == -1]
            raise ValueError(
                f"Matching failed at pass {pass_k}: {len(unmatched)} left nodes "
                f"unmatched. Config: {config}. "
                f"(out={out}, in={in_}, d_L={d_L}, d_R={d_R}, pass={pass_k})"
            )

        for i, j in enumerate(assignment):
            mask[i, j] = True

    _assert_biregular(mask, d_L, d_R)
    return mask


def _augmenting_path_matching(
    adj: List[List[int]],
    n_left: int,
    n_right: int,
) -> List[int]:
    """
    Maximum bipartite matching via augmenting paths (DFS).
    Returns match_left[i] = j, or -1 if unmatched.
    """
    match_left  = [-1] * n_left
    match_right = [-1] * n_right

    def dfs(u: int, visited: set) -> bool:
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                if match_right[v] == -1 or dfs(match_right[v], visited):
                    match_left[u]  = v
                    match_right[v] = u
                    return True
        return False

    for u in range(n_left):
        dfs(u, set())

    return match_left


def _assert_biregular(mask: np.ndarray, d_L: int, d_R: int):
    row_sums = mask.sum(axis=1)
    col_sums = mask.sum(axis=0)
    if not (row_sums == d_L).all():
        bad = np.where(row_sums != d_L)[0][:5]
        raise AssertionError(
            f"Row sums not {d_L}: got {row_sums[bad]} at rows {bad}"
        )
    if not (col_sums == d_R).all():
        bad = np.where(col_sums != d_R)[0][:5]
        raise AssertionError(
            f"Col sums not {d_R}: got {col_sums[bad]} at cols {bad}"
        )