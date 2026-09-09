"""Degree arithmetic for biregular masks.

Pure arithmetic: which (d_L, d_R) pairs are realisable for a given layer shape,
and which one sits closest to a requested sparsity. No graph construction here.

MIN DEGREE
----------
The default ``min_degree=3`` is not a heuristic. Kim, Sudakov & Vu (Random
Structures & Algorithms 22(1), 2002), Corollary 1.2: for all 3 <= d <= n - 4 the
random d-regular graph G_{n,d} is almost surely asymmetric. Degree 2 gives a
disjoint union of cycles, whose automorphism group is large. Since the whole
point of a fixed mask is to collapse the permutation gauge from S_n to Aut(G),
dropping below 3 reintroduces exactly what the mask was meant to remove.

Caveat carried deliberately: KSV prove this for d-regular graphs. The bipartite
biregular case is not covered by that paper and, as far as we have found, not by
any other. Treat the floor as a well-motivated conjecture and certify asymmetry
per mask with ``gauge.certify_trivial_automorphisms``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MIN_DEGREE_DEFAULT = 3
"""Kim-Sudakov-Vu asymmetry floor. See module docstring."""


@dataclass(frozen=True)
class DegreeConfig:
    """A realisable biregular degree assignment for an (out, in) layer.

    Every row of the mask carries exactly ``d_L`` ones, every column exactly
    ``d_R``. Edge conservation ``d_L * out == d_R * in`` holds by construction.
    """

    out_features: int
    in_features: int
    d_L: int
    d_R: int

    def __post_init__(self) -> None:
        if self.d_L * self.out_features != self.d_R * self.in_features:
            raise ValueError(
                f"edge conservation violated: {self.d_L}*{self.out_features} "
                f"!= {self.d_R}*{self.in_features}"
            )
        if self.d_L < 1 or self.d_R < 1:
            raise ValueError(f"degrees must be positive, got {self.d_L}, {self.d_R}")
        if self.d_L > self.in_features or self.d_R > self.out_features:
            raise ValueError(
                f"degree exceeds dense: d_L={self.d_L} > in={self.in_features} "
                f"or d_R={self.d_R} > out={self.out_features}"
            )

    @property
    def sparsity(self) -> float:
        return 1.0 - self.d_L / self.in_features

    @property
    def density(self) -> float:
        return self.d_L / self.in_features

    @property
    def n_edges(self) -> int:
        return self.d_L * self.out_features

    @property
    def min_degree(self) -> int:
        return min(self.d_L, self.d_R)


def enumerate_valid_degrees(
    out_features: int,
    in_features: int,
    min_degree: int = MIN_DEGREE_DEFAULT,
) -> list[DegreeConfig]:
    """All realisable configs for a shape, sparsest first.

    The grid is exactly ``k * (in/g, out/g)`` for k in 1..g, g = gcd(out, in),
    so there are at most gcd(out, in) levels.
    """
    if out_features < 1 or in_features < 1:
        raise ValueError("layer dimensions must be positive")

    g = math.gcd(out_features, in_features)
    step_L = in_features // g
    step_R = out_features // g

    configs: list[DegreeConfig] = []
    for k in range(1, g + 1):
        d_L, d_R = k * step_L, k * step_R
        if d_L > in_features or d_R > out_features:
            break
        if min(d_L, d_R) < min_degree:
            continue
        configs.append(DegreeConfig(out_features, in_features, d_L, d_R))

    return sorted(configs, key=lambda c: c.sparsity, reverse=True)


def select_degree(
    out_features: int,
    in_features: int,
    target_sparsity: float,
    min_degree: int = MIN_DEGREE_DEFAULT,
) -> DegreeConfig | None:
    """Config whose achievable sparsity is closest to the target, or None."""
    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError(f"target_sparsity must be in [0, 1), got {target_sparsity}")
    configs = enumerate_valid_degrees(out_features, in_features, min_degree)
    if not configs:
        return None
    return min(configs, key=lambda c: abs(c.sparsity - target_sparsity))


def sparsity_range(
    out_features: int,
    in_features: int,
    min_degree: int = MIN_DEGREE_DEFAULT,
) -> tuple[float, float] | None:
    """(min, max) achievable sparsity for a shape, or None if nothing is valid."""
    configs = enumerate_valid_degrees(out_features, in_features, min_degree)
    if not configs:
        return None
    s = [c.sparsity for c in configs]
    return (min(s), max(s))
