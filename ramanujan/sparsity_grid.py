"""
sparsity_grid.py
----------------
Pure arithmetic layer: enumerating valid (d_L, d_R) pairs for a given
(out, in) layer shape, and selecting the degree closest to a target sparsity.

No graph construction here -- this module only answers:
  "What sparsity levels are actually achievable for this shape?"
"""

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class DegreeConfig:
    """
    A valid biregular degree assignment for a (out_features, in_features) layer.

    Every row of the mask has exactly d_L ones (left-degree).
    Every column of the mask has exactly d_R ones (right-degree).

    Edge conservation: d_L * out == d_R * in  (always satisfied by construction).
    """
    out_features: int
    in_features: int
    d_L: int   # connections per output neuron
    d_R: int   # connections per input neuron

    @property
    def sparsity(self) -> float:
        return 1.0 - self.d_L / self.in_features

    @property
    def density(self) -> float:
        return self.d_L / self.in_features

    @property
    def n_edges(self) -> int:
        return self.d_L * self.out_features

    def __post_init__(self):
        assert self.d_L * self.out_features == self.d_R * self.in_features, (
            f"Edge conservation violated: {self.d_L}*{self.out_features} "
            f"!= {self.d_R}*{self.in_features}"
        )
        assert self.d_L >= 1 and self.d_R >= 1


def enumerate_valid_degrees(
    out_features: int,
    in_features: int,
    min_degree: int = 3,
) -> List[DegreeConfig]:
    """
    Return all valid DegreeConfigs for a (out, in) layer, sorted by sparsity
    descending (sparsest first).

    A DegreeConfig is valid iff:
      - d_L * out == d_R * in  (edge conservation)
      - d_L >= min_degree and d_R >= min_degree
      - d_L <= in_features and d_R <= out_features  (can't exceed dense)

    The full grid is parameterized by k in [1, g] where g = gcd(out, in):
      d_L = k * (in / g)
      d_R = k * (out / g)

    So there are exactly gcd(out, in) valid levels.
    """
    g = math.gcd(out_features, in_features)
    step_L = in_features // g   # minimum non-zero d_L
    step_R = out_features // g  # minimum non-zero d_R

    configs = []
    for k in range(1, g + 1):
        d_L = k * step_L
        d_R = k * step_R

        if d_L < min_degree or d_R < min_degree:
            continue
        if d_L > in_features or d_R > out_features:
            break  # all further k will also exceed bounds

        configs.append(DegreeConfig(
            out_features=out_features,
            in_features=in_features,
            d_L=d_L,
            d_R=d_R,
        ))

    # Sorted sparsest first
    return sorted(configs, key=lambda c: c.sparsity, reverse=True)


def select_degree(
    out_features: int,
    in_features: int,
    target_sparsity: float,
    min_degree: int = 3,
) -> Optional[DegreeConfig]:
    """
    Select the DegreeConfig whose actual sparsity is closest to target_sparsity.
    Returns None if no valid config exists (e.g. layer too small for min_degree).
    """
    configs = enumerate_valid_degrees(out_features, in_features, min_degree)
    if not configs:
        return None

    return min(configs, key=lambda c: abs(c.sparsity - target_sparsity))


def sparsity_range(
    out_features: int,
    in_features: int,
    min_degree: int = 3,
) -> Optional[tuple]:
    """
    Return (min_achievable_sparsity, max_achievable_sparsity) for this shape.
    Useful for surfacing to the user before construction.
    """
    configs = enumerate_valid_degrees(out_features, in_features, min_degree)
    if not configs:
        return None
    sparsities = [c.sparsity for c in configs]
    return (min(sparsities), max(sparsities))