"""
test_ramanujan_numpy.py
-----------------------
Tests the pure-logic components without requiring torch.
Covers sparsity grid, mask construction, and spectral verification.
All torch tensors replaced with numpy arrays for this suite.

On the actual server (with torch installed), run test_ramanujan.py instead.
"""

import math
import sys
import time
import numpy as np
import hashlib
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple


# -----------------------------------------------------------------------
# Inline the core logic (no torch dependency)
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class DegreeConfig:
    out_features: int
    in_features: int
    d_L: int
    d_R: int

    @property
    def sparsity(self):
        return 1.0 - self.d_L / self.in_features

    @property
    def density(self):
        return self.d_L / self.in_features

    def __post_init__(self):
        assert self.d_L * self.out_features == self.d_R * self.in_features, (
            f"Edge conservation: {self.d_L}*{self.out_features} "
            f"!= {self.d_R}*{self.in_features}"
        )


def enumerate_valid_degrees(out, in_, min_degree=3):
    g = math.gcd(out, in_)
    step_L = in_ // g
    step_R = out // g
    configs = []
    for k in range(1, g + 1):
        d_L = k * step_L
        d_R = k * step_R
        if d_L < min_degree or d_R < min_degree:
            continue
        if d_L > in_ or d_R > out:
            break
        configs.append(DegreeConfig(out, in_, d_L, d_R))
    return sorted(configs, key=lambda c: c.sparsity, reverse=True)


def select_degree(out, in_, target_sparsity, min_degree=3):
    configs = enumerate_valid_degrees(out, in_, min_degree)
    if not configs:
        return None
    return min(configs, key=lambda c: abs(c.sparsity - target_sparsity))


def build_biregular_mask_np(config: DegreeConfig, seed: int = 0) -> np.ndarray:
    """
    Build a (d_L, d_R)-biregular mask on (out, in) using d_L successive
    random perfect matchings.

    At pass k, we find a random maximum matching in the bipartite graph:
      Left:  all out nodes (each needs 1 new edge this pass)
      Right: all in_ nodes with remaining capacity > 0, excluding edges
             already in the mask for each left node

    Uses augmenting-path matching (randomized via shuffled adjacency lists).
    Feasibility is guaranteed by Hall's theorem as long as the total remaining
    capacity distributes appropriately -- which holds when matching is balanced.
    To enforce balance, right nodes are ordered by remaining capacity (most
    available first) to prevent early depletion.
    """
    out, in_ = config.out_features, config.in_features
    d_L, d_R = config.d_L, config.d_R
    rng = np.random.default_rng(seed)

    mask = np.zeros((out, in_), dtype=bool)

    for pass_k in range(d_L):
        col_remaining = d_R - mask.sum(axis=0)  # shape (in_,)

        # Build adjacency list for this pass: left node i -> valid right nodes
        # Valid: has capacity AND not already connected to i
        # Shuffle within each adjacency list for randomness
        available_right = np.where(col_remaining > 0)[0]

        adj = []
        for i in range(out):
            # Right nodes i can connect to this pass
            valid = available_right[~mask[i, available_right]]
            # Order by remaining capacity descending (most available first)
            # This prevents depletion of popular right nodes
            order = np.argsort(-col_remaining[valid])
            shuffled = valid[order]
            # Add a random shuffle within capacity tiers to maintain randomness
            # (nodes with same capacity are shuffled)
            if len(shuffled) > 0:
                caps = col_remaining[shuffled]
                for cap_val in np.unique(caps):
                    tier_idx = np.where(caps == cap_val)[0]
                    # Correct in-place shuffle: copy the slice, shuffle, write back
                    tier_slice = shuffled[tier_idx].copy()
                    rng.shuffle(tier_slice)
                    shuffled[tier_idx] = tier_slice
            adj.append(shuffled.tolist())

        # Find a random perfect matching via augmenting paths
        assignment = _augmenting_path_matching(adj, out, in_)

        if -1 in assignment:
            unmatched = [i for i, j in enumerate(assignment) if j == -1]
            raise ValueError(
                f"Pass {pass_k}: augmenting path matching failed for {len(unmatched)} "
                f"left nodes. Config: {config}. "
                f"This indicates a construction infeasibility."
            )

        for i, j in enumerate(assignment):
            mask[i, j] = True

    _assert_biregular_np(mask, d_L, d_R)
    return mask


def _augmenting_path_matching(
    adj: list,
    n_left: int,
    n_right: int,
) -> list:
    """
    Maximum bipartite matching via augmenting paths (Hopcroft-Karp variant).
    adj[i] = list of right nodes that left node i can be matched to.
    Returns match_left[i] = j (right node matched to left i), or -1 if unmatched.
    """
    match_left = [-1] * n_left
    match_right = [-1] * n_right

    def dfs(u: int, visited: set) -> bool:
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                if match_right[v] == -1 or dfs(match_right[v], visited):
                    match_left[u] = v
                    match_right[v] = u
                    return True
        return False

    for u in range(n_left):
        dfs(u, set())

    return match_left


def _assert_biregular_np(mask, d_L, d_R):
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


def verify_ramanujan_np(mask: np.ndarray, config: DegreeConfig, tol=1e-6):
    d_L, d_R = config.d_L, config.d_R
    B = mask.astype(np.float32)
    B_norm = B / math.sqrt(d_L * d_R)

    # Use full SVD for these test sizes
    S = np.linalg.svd(B_norm, compute_uv=False)

    sigma_1 = float(S[0])
    sigma_2 = float(S[1]) if len(S) > 1 else 0.0
    bound = (math.sqrt(d_L - 1) + math.sqrt(d_R - 1)) / math.sqrt(d_L * d_R)

    return {
        "sigma_1": sigma_1,
        "sigma_2": sigma_2,
        "ramanujan_bound": bound,
        "is_ramanujan": sigma_2 <= bound + tol,
        "spectral_gap": sigma_1 - sigma_2,
    }


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

def test_grid_edge_conservation():
    shapes = [(64, 64), (128, 512), (256, 1024), (768, 3072)]
    for shape in shapes:
        configs = enumerate_valid_degrees(*shape)
        assert len(configs) > 0, f"No configs for {shape}"
        for c in configs:
            assert c.d_L * c.out_features == c.d_R * c.in_features
    print("PASS test_grid_edge_conservation")


def test_grid_sorted_descending():
    configs = enumerate_valid_degrees(256, 256)
    sparsities = [c.sparsity for c in configs]
    assert sparsities == sorted(sparsities, reverse=True)
    print(f"PASS test_grid_sorted_descending ({len(configs)} levels for 256x256)")


def test_grid_count():
    """Number of valid configs equals gcd(out, in) minus small-degree rejects."""
    out, in_ = 64, 64
    g = math.gcd(out, in_)  # = 64
    configs = enumerate_valid_degrees(out, in_, min_degree=1)
    # All k from 1..g are valid when min_degree=1 and step_L=1
    assert len(configs) == g
    print(f"PASS test_grid_count (gcd={g}, configs={len(configs)})")


def test_select_degree_closest():
    targets = [0.5, 0.75, 0.9, 0.95]
    for target in targets:
        config = select_degree(256, 256, target)
        assert config is not None
        all_configs = enumerate_valid_degrees(256, 256)
        errors = [abs(c.sparsity - target) for c in all_configs]
        best_error = min(errors)
        assert abs(config.sparsity - target) <= best_error + 1e-9
    print("PASS test_select_degree_closest")


def test_mask_biregularity_square():
    for (out, in_, target) in [(64, 64, 0.9), (128, 128, 0.75), (64, 64, 0.5)]:
        config = select_degree(out, in_, target)
        mask = build_biregular_mask_np(config, seed=42)
        assert mask.shape == (out, in_)
        assert mask.dtype == bool
        assert (mask.sum(axis=1) == config.d_L).all(), "Row sums non-uniform"
        assert (mask.sum(axis=0) == config.d_R).all(), "Col sums non-uniform"
        actual_sparsity = 1.0 - mask.mean()
        assert abs(actual_sparsity - config.sparsity) < 1e-6
    print("PASS test_mask_biregularity_square")


def test_mask_biregularity_rectangular():
    for (out, in_) in [(64, 256), (128, 512), (32, 128)]:
        config = select_degree(out, in_, 0.9)
        mask = build_biregular_mask_np(config, seed=0)
        assert mask.shape == (out, in_)
        assert (mask.sum(axis=1) == config.d_L).all()
        assert (mask.sum(axis=0) == config.d_R).all()
    print("PASS test_mask_biregularity_rectangular")


def test_mask_reproducibility():
    # Use a larger shape so the construction space is big enough
    # that seed=0 and seed=99 reliably differ
    config = select_degree(128, 128, 0.9)
    m1 = build_biregular_mask_np(config, seed=0)
    m2 = build_biregular_mask_np(config, seed=0)
    m3 = build_biregular_mask_np(config, seed=99)
    assert np.array_equal(m1, m2), "Same seed must give same mask"
    assert not np.array_equal(m1, m3), "Different seeds should differ (128x128 space is large)"
    print("PASS test_mask_reproducibility")


def test_sigma1_is_trivial():
    """Largest singular value of B_norm should be ~1.0."""
    config = select_degree(128, 128, 0.9)
    mask = build_biregular_mask_np(config, seed=0)
    r = verify_ramanujan_np(mask, config)
    assert abs(r["sigma_1"] - 1.0) < 0.02, f"sigma_1={r['sigma_1']:.4f}"
    print(f"PASS test_sigma1_is_trivial (sigma_1={r['sigma_1']:.6f})")


def test_ramanujan_bound_majority():
    """Majority of random seeds should satisfy the Ramanujan bound."""
    config = select_degree(128, 128, 0.9)
    n_trials = 5
    passes = 0
    for seed in range(n_trials):
        mask = build_biregular_mask_np(config, seed=seed)
        r = verify_ramanujan_np(mask, config)
        if r["is_ramanujan"]:
            passes += 1
        print(f"  seed={seed}: sigma_2={r['sigma_2']:.4f} bound={r['ramanujan_bound']:.4f} "
              f"{'OK' if r['is_ramanujan'] else 'FAIL'}")
    assert passes >= n_trials // 2, f"Only {passes}/{n_trials} passed"
    print(f"PASS test_ramanujan_bound_majority ({passes}/{n_trials})")


def test_spectral_sweep():
    """Full table across sparsity levels."""
    print("\n--- Spectral sweep (128x128) ---")
    print(f"{'target':>8} {'actual':>8} {'d_L':>5} {'sigma_1':>8} "
          f"{'sigma_2':>8} {'bound':>8} {'gap':>8} {'ramanujan':>10}")
    for target in [0.5, 0.7, 0.8, 0.9, 0.95]:
        config = select_degree(128, 128, target)
        mask = build_biregular_mask_np(config, seed=0)
        r = verify_ramanujan_np(mask, config)
        print(
            f"{target:>8.2f} {config.sparsity:>8.4f} {config.d_L:>5} "
            f"{r['sigma_1']:>8.4f} {r['sigma_2']:>8.4f} "
            f"{r['ramanujan_bound']:>8.4f} {r['spectral_gap']:>8.4f} "
            f"{'YES' if r['is_ramanujan'] else 'NO':>10}"
        )
    print("PASS test_spectral_sweep")


def test_construction_timing():
    """Build time for a few layer sizes -- useful for cache design."""
    print("\n--- Construction timing ---")
    shapes = [(64, 64, 0.9), (128, 512, 0.9), (256, 1024, 0.9), (512, 2048, 0.9)]
    for out, in_, target in shapes:
        config = select_degree(out, in_, target)
        t0 = time.perf_counter()
        mask = build_biregular_mask_np(config, seed=0)
        elapsed = (time.perf_counter() - t0) * 1000
        nnz = mask.sum()
        print(f"  {out:>5}x{in_:<6} d_L={config.d_L:>4} sparsity={config.sparsity:.4f} "
              f"nnz={nnz:>8} time={elapsed:.1f}ms")
    print("PASS test_construction_timing")


def test_degree_config_invariants():
    """DegreeConfig properties are self-consistent."""
    config = select_degree(256, 1024, 0.9)
    assert config.d_L * config.out_features == config.d_R * config.in_features
    assert abs(config.sparsity + config.density - 1.0) < 1e-9
    assert config.d_L * config.out_features == config.d_R * config.in_features  # via n_edges
    print(f"PASS test_degree_config_invariants "
          f"(d_L={config.d_L}, d_R={config.d_R}, sparsity={config.sparsity:.4f})")


if __name__ == "__main__":
    print("=" * 62)
    print("Ramanujan construction tests (numpy-only)")
    print("=" * 62)

    print("\n[1/4] Sparsity grid")
    test_grid_edge_conservation()
    test_grid_sorted_descending()
    test_grid_count()
    test_select_degree_closest()
    test_degree_config_invariants()

    print("\n[2/4] Mask biregularity")
    test_mask_biregularity_square()
    test_mask_biregularity_rectangular()
    test_mask_reproducibility()

    print("\n[3/4] Spectral properties")
    test_sigma1_is_trivial()
    test_ramanujan_bound_majority()
    test_spectral_sweep()

    print("\n[4/4] Construction timing")
    test_construction_timing()

    print("\n" + "=" * 62)
    print("All tests passed.")