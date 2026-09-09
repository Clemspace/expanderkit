"""Biregular sparse masks with explicit gauge accounting.

The numpy core (sparsity_grid, masks, spectral, automorphisms, registry) has no
torch dependency. ``ramanujan.layer`` requires torch and is imported lazily.

Gauge MEASUREMENT on parameter vectors -- the symmetry tax, scale
canonicalisation, architecture permutation groups -- lives in the audit package,
not here. This package answers questions about masks.
"""

from .sparsity_grid import (
    MIN_DEGREE_DEFAULT,
    DegreeConfig,
    enumerate_valid_degrees,
    select_degree,
    sparsity_range,
)
from .masks import (
    CONSTRUCTION_VERSION,
    METHODS,
    MaskConstructionError,
    MaskSpec,
    assert_biregular,
    build_mask,
    gather_index,
    indices_to_mask,
    mask_to_indices,
)
from .spectral import (
    SpectralReport,
    analyse,
    empirical_sigma2_null,
    ramanujan_bound,
    singular_values,
)
from .automorphisms import (
    AsymmetryCertificate,
    apply_mask_permutation,
    automorphism_count,
    birthday_collision_estimate,
    certify_trivial_automorphisms,
    duplicate_neighbourhood_swaps,
    is_automorphism,
)
from .registry import MaskRegistry

__all__ = [
    "MIN_DEGREE_DEFAULT", "DegreeConfig", "enumerate_valid_degrees",
    "select_degree", "sparsity_range",
    "CONSTRUCTION_VERSION", "METHODS", "MaskConstructionError", "MaskSpec",
    "assert_biregular", "build_mask", "gather_index", "indices_to_mask",
    "mask_to_indices",
    "SpectralReport", "analyse", "empirical_sigma2_null", "ramanujan_bound",
    "singular_values",
    "AsymmetryCertificate", "apply_mask_permutation", "automorphism_count",
    "birthday_collision_estimate", "certify_trivial_automorphisms",
    "duplicate_neighbourhood_swaps", "is_automorphism",
    "MaskRegistry",
]
