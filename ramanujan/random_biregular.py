"""
random_biregular.py
-------------------

Random biregular bipartite masks for the non-Ramanujan control condition
in the sequential-curriculum escape-rate study.

Intent: sparsity-matched, structurally-matched (biregular), but without
the Ramanujan spectral guarantee. Any random biregular graph has some
sigma_2; Friedman's theorem says random biregular graphs are "close to
Ramanujan" with high probability as n -> infinity, but at small matrix
sizes (128 x 128, 128 x 512) the concentration is loose enough to
rejection-sample non-Ramanujan instances reliably.

Core contract:
    sample_non_ramanujan_biregular(out_dim, in_dim, sparsity, seed)
        -> (mask: torch.BoolTensor, spectral_info: dict)

The returned mask is guaranteed biregular (every row has the same
number of non-zeros, every column has the same number of non-zeros)
and guaranteed non-Ramanujan (sigma_2 > sqrt(d_L-1) + sqrt(d_R-1)).

Integration point: wire into your existing `make_linear` factory as a
new branch, parallel to the `ramanujan_layer` branch. The mask tensor
shape and dtype match what RamanujanGraphBuilder.build() returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SpectralInfo:
    """Spectral summary of a biregular bipartite mask."""

    sigma_1: float  # largest singular value (should be ~sqrt(d_L * d_R))
    sigma_2: float  # second-largest singular value (the one that matters)
    ramanujan_bound: float  # sqrt(d_L - 1) + sqrt(d_R - 1)
    is_ramanujan: bool  # sigma_2 <= ramanujan_bound
    margin: float  # sigma_2 - ramanujan_bound (positive => NOT Ramanujan)
    d_L: int  # non-zeros per row
    d_R: int  # non-zeros per column


def compute_spectral_info(mask: torch.Tensor) -> SpectralInfo:
    """Compute the Ramanujan-relevant spectral quantities for a mask."""
    M = mask.float().cpu().numpy()
    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)

    d_L = int(row_sums[0])
    d_R = int(col_sums[0])

    if not (np.all(row_sums == d_L) and np.all(col_sums == d_R)):
        raise ValueError(
            f"Mask is not biregular: row sums vary in "
            f"[{row_sums.min()}, {row_sums.max()}], col sums in "
            f"[{col_sums.min()}, {col_sums.max()}]"
        )

    # SVD: we only need the top 2 singular values but scipy-sparse
    # partial SVD is overkill at these sizes. Full SVD is fine.
    s = np.linalg.svd(M, compute_uv=False)
    sigma_1 = float(s[0])
    sigma_2 = float(s[1]) if len(s) > 1 else 0.0

    bound = float(np.sqrt(max(d_L - 1, 0)) + np.sqrt(max(d_R - 1, 0)))
    margin = sigma_2 - bound
    is_ram = margin <= 0

    return SpectralInfo(
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        ramanujan_bound=bound,
        is_ramanujan=is_ram,
        margin=margin,
        d_L=d_L,
        d_R=d_R,
    )


def _target_degrees(out_dim: int, in_dim: int, sparsity: float) -> tuple[int, int]:
    """
    Choose row-degree d_L and column-degree d_R so the resulting mask is
    (a) biregular and (b) as close as possible to the requested sparsity.

    Exact biregularity requires d_L * out_dim == d_R * in_dim. We pick
    d_L = round((1 - sparsity) * in_dim) and then compute the d_R that
    makes the count consistent. If the divisibility doesn't work, we
    adjust d_L by +/- 1.
    """
    density = 1.0 - sparsity
    d_L = max(1, int(round(density * in_dim)))
    total = d_L * out_dim
    if total % in_dim != 0:
        # Search nearby d_L values for exact divisibility
        for delta in range(1, max(in_dim, out_dim)):
            for d_L_try in (d_L - delta, d_L + delta):
                if d_L_try < 1 or d_L_try > in_dim:
                    continue
                if (d_L_try * out_dim) % in_dim == 0:
                    d_L = d_L_try
                    break
            else:
                continue
            break
    d_R = (d_L * out_dim) // in_dim
    return d_L, d_R


def _sample_biregular_config_model(
    out_dim: int, in_dim: int, d_L: int, d_R: int, rng: np.random.Generator
) -> torch.Tensor:
    """
    Random simple biregular bipartite graph via configuration model with
    double-edge-swap repair of multi-edges.

    Algorithm:
      1. Configuration model: attach d_L stubs to each row-vertex and d_R
         stubs to each column-vertex, then match stubs by random permutation.
         Produces a multigraph with the correct degree sequence.
      2. Repair multi-edges with double-edge swaps. For each duplicate
         edge (r, c), find a partner edge (r', c') where r != r' and
         c != c', such that neither (r, c') nor (r', c) are currently
         edges. Swap to produce (r, c') and (r', c). This preserves all
         vertex degrees and strictly reduces the total multi-edge count.

    Convergence: each successful swap strictly decreases multi-edge
    occurrences by one. At densities below ~50%, valid partners almost
    always exist. We retry from a fresh config-model draw if we get
    stuck, which in practice is extremely rare at our matrix shapes.
    """
    E = d_L * out_dim
    if E != d_R * in_dim:
        raise ValueError(
            f"Degree sequence inconsistent: d_L*out_dim ({d_L*out_dim}) "
            f"!= d_R*in_dim ({d_R*in_dim})"
        )

    max_fresh_draws = 10
    for fresh in range(max_fresh_draws):
        # Config-model draw: random matching of stubs
        rows = np.repeat(np.arange(out_dim, dtype=np.int64), d_L)
        cols = np.repeat(np.arange(in_dim, dtype=np.int64), d_R)
        rng.shuffle(cols)
        edges_row = rows
        edges_col = cols.copy()

        # Location index: (r, c) -> list of edge indices with that endpoint pair
        loc: dict[tuple[int, int], list[int]] = {}
        for i in range(E):
            key = (int(edges_row[i]), int(edges_col[i]))
            loc.setdefault(key, []).append(i)

        swaps_done = 0
        max_swaps = 50 * E
        consecutive_stuck = 0
        max_consecutive_stuck = 500

        while swaps_done < max_swaps:
            multi_keys = [k for k, v in loc.items() if len(v) > 1]
            if not multi_keys:
                break  # simple graph achieved

            r, c = multi_keys[rng.integers(len(multi_keys))]
            idx_to_move = loc[(r, c)][-1]

            partner_found = False
            for _ in range(200):
                partner_idx = int(rng.integers(E))
                if partner_idx == idx_to_move:
                    continue
                r_p = int(edges_row[partner_idx])
                c_p = int(edges_col[partner_idx])
                if r_p == r or c_p == c:
                    continue
                # Refuse swap if either proposed new edge already exists.
                # Using .get() with empty-list default to avoid creating
                # phantom keys.
                if loc.get((r, c_p)):
                    continue
                if loc.get((r_p, c)):
                    continue

                # Execute swap.
                loc[(r, c)].remove(idx_to_move)
                if not loc[(r, c)]:
                    del loc[(r, c)]
                loc[(r_p, c_p)].remove(partner_idx)
                if not loc[(r_p, c_p)]:
                    del loc[(r_p, c_p)]
                edges_col[idx_to_move] = c_p
                edges_col[partner_idx] = c
                loc.setdefault((r, c_p), []).append(idx_to_move)
                loc.setdefault((r_p, c), []).append(partner_idx)

                swaps_done += 1
                consecutive_stuck = 0
                partner_found = True
                break

            if not partner_found:
                consecutive_stuck += 1
                if consecutive_stuck >= max_consecutive_stuck:
                    break  # abandon this draw, try a fresh one

        # Final validation
        if all(len(v) == 1 for v in loc.values()):
            mask = torch.zeros(out_dim, in_dim, dtype=torch.bool)
            mask[edges_row, edges_col] = True
            if not (mask.sum(dim=1) == d_L).all():
                raise RuntimeError("row degrees not uniform after swap-repair")
            if not (mask.sum(dim=0) == d_R).all():
                raise RuntimeError("col degrees not uniform after swap-repair")
            logger.debug(
                "Sampled simple biregular (%d x %d, d_L=%d, d_R=%d) in "
                "%d swaps on draw %d.",
                out_dim, in_dim, d_L, d_R, swaps_done, fresh + 1,
            )
            return mask

    raise RuntimeError(
        f"Could not sample simple biregular after {max_fresh_draws} fresh "
        f"draws at ({out_dim}, {in_dim}, d_L={d_L}, d_R={d_R}). Check degree "
        f"sequence consistency or reduce density."
    )


def sample_non_ramanujan_biregular(
    out_dim: int,
    in_dim: int,
    sparsity: float,
    seed: int,
    max_attempts: int = 200,
    min_margin: float = 0.05,
) -> tuple[torch.Tensor, SpectralInfo]:
    """
    Sample random biregular masks until one fails the Ramanujan bound
    by at least `min_margin` in sigma_2.

    The margin guards against borderline-Ramanujan samples that would
    make the control condition weak. A non-zero margin ensures the
    control is unambiguously not Ramanujan even under numerical noise.
    """
    rng = np.random.default_rng(seed)
    d_L, d_R = _target_degrees(out_dim, in_dim, sparsity)
    effective_sparsity = 1.0 - (d_L * out_dim) / (out_dim * in_dim)

    if abs(effective_sparsity - sparsity) > 0.02:
        logger.warning(
            "Requested sparsity %.3f adjusted to %.3f for exact biregularity "
            "at dims (%d, %d). d_L=%d, d_R=%d.",
            sparsity,
            effective_sparsity,
            out_dim,
            in_dim,
            d_L,
            d_R,
        )

    for attempt in range(max_attempts):
        attempt_seed = seed * 10_000 + attempt
        mask = _sample_biregular_config_model(
            out_dim, in_dim, d_L, d_R, np.random.default_rng(attempt_seed)
        )
        info = compute_spectral_info(mask)
        if not info.is_ramanujan and info.margin >= min_margin:
            logger.info(
                "Non-Ramanujan biregular sampled after %d attempts "
                "(d_L=%d, d_R=%d, sigma_2=%.3f, bound=%.3f, margin=%.3f).",
                attempt + 1,
                info.d_L,
                info.d_R,
                info.sigma_2,
                info.ramanujan_bound,
                info.margin,
            )
            return mask, info

    raise RuntimeError(
        f"Could not sample non-Ramanujan biregular at ({out_dim}, {in_dim}) "
        f"sparsity {sparsity} with margin >= {min_margin} after "
        f"{max_attempts} attempts. At these dimensions most random biregular "
        f"masks are close to Ramanujan (Friedman); consider relaxing "
        f"min_margin or using unstructured random sparse as a secondary "
        f"control."
    )


def sample_unstructured_random_sparse(
    out_dim: int,
    in_dim: int,
    sparsity: float,
    seed: int,
) -> tuple[torch.Tensor, SpectralInfo]:
    """
    Secondary control: Bernoulli-sampled unstructured random sparse mask.

    Not biregular. Use only if the biregular non-Ramanujan sampler fails
    or if you want the crudest possible "is it sparsity or structure?"
    comparison.
    """
    rng = np.random.default_rng(seed)
    density = 1.0 - sparsity
    mask = torch.from_numpy(rng.random((out_dim, in_dim)) < density)

    # For non-biregular masks, compute_spectral_info will fail. Compute
    # sigma_2 directly and skip the biregularity assertion.
    M = mask.float().cpu().numpy()
    s = np.linalg.svd(M, compute_uv=False)
    sigma_1 = float(s[0])
    sigma_2 = float(s[1]) if len(s) > 1 else 0.0
    d_L = int(M.sum(axis=1).mean())  # average, since not biregular
    d_R = int(M.sum(axis=0).mean())
    bound = float(np.sqrt(max(d_L - 1, 0)) + np.sqrt(max(d_R - 1, 0)))
    info = SpectralInfo(
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        ramanujan_bound=bound,
        is_ramanujan=False,  # by convention for non-biregular
        margin=sigma_2 - bound,
        d_L=d_L,
        d_R=d_R,
    )
    return mask, info


# ---------------------------------------------------------------------------
# Minimal self-test. Run `python random_biregular.py` to sanity-check.
# ---------------------------------------------------------------------------


def _selftest() -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    for shape in [(128, 128), (512, 128), (128, 512)]:
        out_dim, in_dim = shape
        for sparsity in [0.75, 0.90]:
            mask, info = sample_non_ramanujan_biregular(
                out_dim, in_dim, sparsity, seed=42, min_margin=0.05
            )
            actual_sparsity = 1.0 - mask.float().mean().item()
            print(
                f"shape={shape} target_sparsity={sparsity:.2f} "
                f"actual={actual_sparsity:.3f} "
                f"sigma_2={info.sigma_2:.3f} bound={info.ramanujan_bound:.3f} "
                f"margin={info.margin:+.3f} "
                f"is_ramanujan={info.is_ramanujan}"
            )


if __name__ == "__main__":
    _selftest()