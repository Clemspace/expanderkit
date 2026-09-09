"""
spectral_verify.py
------------------
Spectral verification for bipartite biregular sparse masks.

For a (d_L, d_R)-biregular bipartite graph with adjacency matrix A (shape m×n),
the Ramanujan bound (Feng & Li, 1996) is:

    σ₂(A) ≤ √(d_L − 1) + √(d_R − 1)

where σ₂ is the second-largest singular value of A.

For the symmetric case d_L = d_R = d (square weight matrix), this reduces to
the classical Ramanujan bound 2√(d − 1).

σ₁ is analytically known for biregular graphs: σ₁ = √(d_L × d_R), with
left singular vector u₁ = 1/√m · 1_m and right singular vector v₁ = 1/√n · 1_n.
We deflate this component and estimate σ₂ via power iteration on the deflated
matrix B = A − σ₁ u₁ v₁ᵀ.

Power iteration cost: O(K × m × n), K=50 iterations. Feasible for
layer shapes up to ~(2048, 2048). For larger shapes, reduce K or use
a random projection approximation (not implemented here).

Public API
----------
ramanujan_bound(d_L, d_R)         → float
second_singular_value(mask, d_L, d_R) → float
ramanujan_margin(mask, d_L, d_R)  → float  (positive = satisfies bound)
is_ramanujan(mask, d_L, d_R)      → bool
spectral_report(mask, d_L, d_R)   → dict
"""

import math
import torch
from typing import Tuple


def ramanujan_bound(d_L: int, d_R: int) -> float:
    """
    Feng-Li Ramanujan bound for (d_L, d_R)-biregular bipartite graphs.
    Reduces to 2√(d-1) for the symmetric case d_L = d_R = d.
    """
    if d_L < 2 or d_R < 2:
        # Degree-1 graphs are trees; bound is 0 (only trivial eigenvalue)
        return 0.0
    return math.sqrt(d_L - 1) + math.sqrt(d_R - 1)


def second_singular_value(
    mask: torch.Tensor,
    d_L: int,
    d_R: int,
    n_iter: int = 50,
    seed: int = 0,
) -> float:
    """
    Estimate σ₂ of the biregular adjacency matrix via power iteration
    on the deflated matrix B = A − σ₁ u₁ v₁ᵀ.

    For a (d_L, d_R)-biregular graph, the top singular triplet is known:
        σ₁ = √(d_L × d_R)
        u₁ = 1/√m · 1_m   (left)
        v₁ = 1/√n · 1_n   (right)

    We deflate B = A − σ₁ u₁ v₁ᵀ implicitly (never forming B explicitly)
    and power-iterate on BᵀB to find σ₂² = ‖B‖ₛ².

    Matrix-vector products:
        B @ v  = A @ v − σ₁ · u₁ · (v₁ᵀ v)
        Bᵀ @ u = Aᵀ @ u − σ₁ · v₁ · (u₁ᵀ u)
    """
    A = mask.float()
    m, n = A.shape

    sigma1 = math.sqrt(d_L * d_R)
    u1 = torch.ones(m, dtype=torch.float32) / math.sqrt(m)  # (m,)
    v1 = torch.ones(n, dtype=torch.float32) / math.sqrt(n)  # (n,)

    # Initialise with reproducible random vector orthogonal to v₁
    gen = torch.Generator()
    gen.manual_seed(seed)
    v = torch.randn(n, generator=gen)
    v = v - v1 * (v1 @ v)   # project out v₁ component
    v = v / (v.norm() + 1e-12)

    sigma2_sq = 0.0
    for _ in range(n_iter):
        # Bv = A@v - σ₁ · u₁ · (v₁ᵀ v)
        Bv = A @ v - sigma1 * u1 * (v1 @ v)
        # BᵀBv = Aᵀ@Bv - σ₁ · v₁ · (u₁ᵀ Bv)
        BTBv = A.T @ Bv - sigma1 * v1 * (u1 @ Bv)

        sigma2_sq = (v @ BTBv).item()
        norm = BTBv.norm().item()
        if norm < 1e-12:
            break
        v = BTBv / norm

    return math.sqrt(abs(sigma2_sq))


def ramanujan_margin(mask: torch.Tensor, d_L: int, d_R: int) -> float:
    """
    Ramanujan margin = bound − σ₂.
    Positive: graph satisfies the Feng-Li Ramanujan bound.
    Negative: graph violates the bound.
    Larger positive margin = better spectral expander.
    """
    bound = ramanujan_bound(d_L, d_R)
    sigma2 = second_singular_value(mask, d_L, d_R)
    return bound - sigma2


def is_ramanujan(mask: torch.Tensor, d_L: int, d_R: int, tol: float = 1e-6) -> bool:
    """
    True iff σ₂(mask) ≤ √(d_L−1) + √(d_R−1) + tol.
    tol accounts for power-iteration approximation error.
    """
    return ramanujan_margin(mask, d_L, d_R) >= -tol


def spectral_report(mask: torch.Tensor, d_L: int, d_R: int) -> dict:
    """
    Full spectral summary for a single mask.
    Suitable for logging to wandb or printing.
    """
    m, n = mask.shape
    sigma1_theoretical = math.sqrt(d_L * d_R)
    bound = ramanujan_bound(d_L, d_R)
    sigma2 = second_singular_value(mask, d_L, d_R)
    margin = bound - sigma2

    return {
        "shape":               (m, n),
        "d_L":                 d_L,
        "d_R":                 d_R,
        "sigma1_theoretical":  round(sigma1_theoretical, 4),
        "sigma2":              round(sigma2, 4),
        "ramanujan_bound":     round(bound, 4),
        "ramanujan_margin":    round(margin, 4),
        "is_ramanujan":        margin >= -1e-6,
    }