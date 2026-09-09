"""
spectral_verifier.py
--------------------
Verifies that a constructed mask satisfies the Ramanujan bound on its
normalized biadjacency matrix.

For a (d_L, d_R)-biregular bipartite graph with biadjacency matrix B,
the normalized version is:

  B_norm = B / sqrt(d_L * d_R)

The singular values of B_norm are in [0, 1]. The trivial singular value
is always 1 (corresponding to the "flat" eigenvector).

Ramanujan bound for biregular bipartite graphs:
  All non-trivial singular values sigma satisfy:
  sigma <= (sqrt(d_L - 1) + sqrt(d_R - 1)) / sqrt(d_L * d_R)

This bound is tight: Ramanujan graphs are exactly those achieving it.
The bound collapses to the classical 2*sqrt(d-1)/d for d-regular (d_L==d_R==d).

Reference: Feng & Li (1996), generalization of Alon-Boppana to biregular case.
"""

import math
import torch
from dataclasses import dataclass
from typing import Optional

from ramanujan.sparsity_grid import DegreeConfig


@dataclass
class SpectralReport:
    d_L: int
    d_R: int
    out_features: int
    in_features: int

    sigma_1: float          # should be ~1.0 (trivial)
    sigma_2: float          # largest non-trivial singular value
    ramanujan_bound: float  # (sqrt(d_L-1) + sqrt(d_R-1)) / sqrt(d_L * d_R)
    is_ramanujan: bool      # sigma_2 <= ramanujan_bound

    # Additional diagnostics
    spectral_gap: float     # sigma_1 - sigma_2 (larger = better connectivity)
    random_bound: float     # expected sigma_2 for Erdos-Renyi at same density

    def summary(self) -> str:
        status = "RAMANUJAN" if self.is_ramanujan else "NOT RAMANUJAN"
        return (
            f"[{status}] ({self.out_features}x{self.in_features}) "
            f"d_L={self.d_L} d_R={self.d_R} | "
            f"sigma_2={self.sigma_2:.4f} bound={self.ramanujan_bound:.4f} "
            f"gap={self.spectral_gap:.4f} | "
            f"vs random bound={self.random_bound:.4f}"
        )


def verify_ramanujan(
    mask: torch.Tensor,
    config: DegreeConfig,
    n_singular_values: int = 10,
    tol: float = 1e-6,
) -> SpectralReport:
    """
    Verify the Ramanujan property of a biregular mask.

    Args:
        mask:             Boolean or float tensor of shape (out, in).
        config:           DegreeConfig used to construct the mask.
        n_singular_values: How many singular values to compute.
                          2 suffices for the bound check; more for diagnostics.
        tol:              Tolerance for Ramanujan bound comparison.

    Returns:
        SpectralReport with all diagnostics.

    Notes:
        Uses full SVD for small masks (<= 4096 total elements),
        randomized SVD (torch.svd_lowrank) for larger ones.
        The randomized version has approximation error but is fast.
    """
    d_L, d_R = config.d_L, config.d_R
    out, in_ = config.out_features, config.in_features

    B = mask.float()

    # Normalize: B_norm = B / sqrt(d_L * d_R)
    # sigma_1 of B_norm should be ~1.0
    B_norm = B / math.sqrt(d_L * d_R)

    n_elements = out * in_
    if n_elements <= 4096 * 4096:
        # Full SVD — accurate
        S = torch.linalg.svdvals(B_norm)
    else:
        # Randomized low-rank SVD — fast for large sparse masks
        # q: oversampling parameter; niter: power iteration steps
        _, S, _ = torch.svd_lowrank(B_norm, q=n_singular_values + 4, niter=4)

    S = S[:n_singular_values]

    sigma_1 = S[0].item()
    sigma_2 = S[1].item() if len(S) > 1 else 0.0

    # Ramanujan bound for biregular bipartite graphs
    ramanujan_bound = (math.sqrt(d_L - 1) + math.sqrt(d_R - 1)) / math.sqrt(d_L * d_R)

    # Expected sigma_2 for Erdos-Renyi graph at same density p = d_L / in_
    # By random matrix theory, expected largest singular value of B/sqrt(d_L*d_R)
    # scales as ~sqrt(out) * p / sqrt(p*(1-p)) / sqrt(d_L*d_R) -- rough bound
    # Simpler: for random bipartite graph, sigma_2 ~ sqrt(out/in_) / sqrt(d_L)
    # This is only a rough comparison, not a tight theoretical bound
    density = d_L / in_
    random_bound = math.sqrt(max(out, in_) * density * (1 - density)) / math.sqrt(d_L * d_R) * 2

    return SpectralReport(
        d_L=d_L,
        d_R=d_R,
        out_features=out,
        in_features=in_,
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        ramanujan_bound=ramanujan_bound,
        is_ramanujan=sigma_2 <= ramanujan_bound + tol,
        spectral_gap=sigma_1 - sigma_2,
        random_bound=random_bound,
    )


def verify_batch(
    masks: list,
    configs: list,
    verbose: bool = True,
) -> list:
    """
    Verify a list of (mask, config) pairs. Returns list of SpectralReports.
    Prints summary if verbose=True.
    """
    reports = []
    for mask, config in zip(masks, configs):
        report = verify_ramanujan(mask, config)
        if verbose:
            print(report.summary())
        reports.append(report)
    return reports