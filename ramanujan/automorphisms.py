"""What symmetry a fixed mask leaves behind.

Under a fixed mask M, a permutation of hidden units maps M to P M Q^T, which is
a different network unless (P, Q) is an automorphism of the mask's bipartite
graph. So the permutation gauge collapses from S_n to Aut(G), and the size of
Aut(G) becomes an architectural choice made before training.

This module answers questions about the MASK ALONE. Anything that needs a
parameter vector -- the symmetry tax, scale canonicalisation, permutation groups
of a real architecture -- lives in the audit package, so that auditing a dense
model does not drag in a sparse-mask library.

CERTIFYING TRIVIALITY
---------------------
Colour refinement is useless here: every left vertex of a biregular graph has
the same degree, so 1-WL never splits anything. The certificate is spectral.

Let B = U S V^T. A side-preserving automorphism satisfies P B Q^T = B, so it
permutes singular vectors within each singular value's eigenspace. If a singular
value is SIMPLE its vector maps to plus or minus itself, so |u_k(i)| is an
automorphism invariant of left vertex i, and likewise |v_k(j)| on the right.

Colour each vertex by those magnitudes over the simple singular values. The
colouring is automorphism-invariant by construction, so if it is DISCRETE every
automorphism fixes every vertex and Aut(G) is trivial. That is a proof, subject
only to the numerical separation, which is reported rather than assumed.

The converse does not hold: a non-discrete colouring means unresolved, not
symmetric. ``duplicate_neighbourhood_swaps`` finds the concrete generators in
the case that actually occurs in practice, and ``automorphism_count`` gives the
exact order when pynauty is installed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

MAX_EXACT_SEPARATION_N = 4096
"""Above this many vertices per side, refuse to certify rather than approximate."""

MAX_COLOUR_DIMS = 16
"""Singular vectors used for the colouring.

Memory, not principle: the exact pairwise separation builds a
(chunk x n x dims) array, so an uncapped colouring at n = 1024 with a
thousand simple singular values wants gigabytes. Discreteness needs a few
well-separated coordinates, not all of them, and using fewer only makes the
certificate more conservative -- it can fail to certify, never certify
falsely."""


@dataclass
class AsymmetryCertificate:
    certified_trivial: bool
    n_simple_singular_values: int
    min_separation: float
    tolerance: float
    unresolved_left: int
    unresolved_right: int
    reason: str

    def summary(self) -> str:
        if self.certified_trivial:
            return (
                f"Aut(G) TRIVIAL (certified): {self.n_simple_singular_values} simple "
                f"singular values, min colour separation {self.min_separation:.2e} "
                f"vs tol {self.tolerance:.1e}"
            )
        return (
            f"Aut(G) UNRESOLVED: {self.reason} "
            f"(unresolved: {self.unresolved_left} left, {self.unresolved_right} right)"
        )


def certify_trivial_automorphisms(
    mask: np.ndarray,
    tol: float = 1e-6,
    simple_gap: float = 1e-6,
) -> AsymmetryCertificate:
    """Try to prove the mask graph has no non-trivial side-preserving automorphism."""
    if max(mask.shape) > MAX_EXACT_SEPARATION_N:
        # Checked before the SVD: refusing after the decomposition costs the
        # same answer and a great deal more time.
        return AsymmetryCertificate(
            False, 0, 0.0, tol, mask.shape[0], mask.shape[1],
            f"graph too large for exact separation (> {MAX_EXACT_SEPARATION_N} per "
            "side); use automorphism_count with pynauty instead",
        )

    B = mask.astype(np.float64)
    U, s, Vt = np.linalg.svd(B, full_matrices=False)

    keep: list[int] = []
    for k in range(s.size):
        lo = s[k - 1] - s[k] if k > 0 else np.inf
        hi = s[k] - s[k + 1] if k + 1 < s.size else np.inf
        if min(lo, hi) > simple_gap:
            keep.append(k)

    if len(keep) > MAX_COLOUR_DIMS:
        # Keep the most separated singular values: the best-conditioned
        # coordinates, so the colouring is as discriminative as it can be
        # within the memory budget.
        gaps = []
        for k in keep:
            lo = s[k - 1] - s[k] if k > 0 else np.inf
            hi = s[k] - s[k + 1] if k + 1 < s.size else np.inf
            gaps.append(min(lo, hi))
        keep = [keep[i] for i in np.argsort(gaps)[::-1][:MAX_COLOUR_DIMS]]

    if not keep:
        return AsymmetryCertificate(
            False, 0, 0.0, tol, mask.shape[0], mask.shape[1],
            "no simple singular values; the spectrum is fully degenerate",
        )

    left = np.abs(U[:, keep])
    right = np.abs(Vt[keep, :].T)

    def separation(colours: np.ndarray) -> tuple[float, int]:
        """EXACT minimum Chebyshev distance between rows, and how many pairs
        sit within ``tol``.

        Pairwise in chunks, not from sorted neighbours: a lexsort only
        guarantees that identical rows are adjacent, so a neighbour scan can
        overestimate the separation and certify falsely. Certification must
        never rest on a bound that is loose in the unsafe direction.
        """
        n = colours.shape[0]
        if n < 2:
            return np.inf, 0
        best, close = np.inf, 0
        for start in range(0, n, 512):
            block = colours[start:start + 512]
            d = np.abs(block[:, None, :] - colours[None, :, :]).max(axis=2)
            d[np.arange(block.shape[0]), np.arange(block.shape[0]) + start] = np.inf
            best = min(best, float(d.min()))
            close += int((d <= tol).sum())
        return best, close // 2

    sep_l, close_l = separation(left)
    sep_r, close_r = separation(right)
    min_sep = float(min(sep_l, sep_r))

    if min_sep > tol:
        return AsymmetryCertificate(
            True, len(keep), min_sep, tol, 0, 0,
            "discrete automorphism-invariant colouring",
        )
    return AsymmetryCertificate(
        False, len(keep), min_sep, tol, close_l, close_r,
        f"colouring not discrete at tol={tol:.1e}",
    )


def duplicate_neighbourhood_swaps(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Concrete automorphism generators from vertices with identical neighbourhoods.

    Two columns wired to exactly the same rows can be swapped; the mask is
    unchanged, so the swap is a gauge move. Same for two identical rows.

    This is not a curiosity. It is the mechanism by which wide, very sparse
    layers lose asymmetry despite satisfying the degree floor: with ``in``
    right nodes of degree ``d_R`` drawn from C(out, d_R) possible
    neighbourhoods, collisions follow a birthday law. Measured at
    (128, 512, d_L=12, d_R=3), certification is 1.00 when this function returns
    nothing and 0.00 when it returns anything.

    Returns (P, Q) index-array pairs, each a transposition. Verify with
    ``is_automorphism``.
    """
    out, in_ = mask.shape
    gens: list[tuple[np.ndarray, np.ndarray]] = []

    by_cols: dict[bytes, list[int]] = defaultdict(list)
    for j in range(in_):
        by_cols[mask[:, j].tobytes()].append(j)
    for group in by_cols.values():
        for a, b in zip(group, group[1:]):
            Q = np.arange(in_)
            Q[a], Q[b] = b, a
            gens.append((np.arange(out), Q))

    by_rows: dict[bytes, list[int]] = defaultdict(list)
    for i in range(out):
        by_rows[mask[i].tobytes()].append(i)
    for group in by_rows.values():
        for a, b in zip(group, group[1:]):
            P = np.arange(out)
            P[a], P[b] = b, a
            gens.append((P, np.arange(in_)))

    return gens


def birthday_collision_estimate(out_features: int, in_features: int,
                                d_L: int, d_R: int) -> dict:
    """Expected duplicate-neighbourhood pairs, and P(at least one).

    lambda_right = C(in, 2) / C(out, d_R), lambda_left = C(out, 2) / C(in, d_L).
    Design condition: both far below 1, i.e. C(out, d_R) >> C(in, 2). Sharper
    and more binding than the degree floor, and it bites hardest exactly on
    wide, very sparse layers.
    """
    from math import comb, exp

    lam_r = comb(in_features, 2) / comb(out_features, d_R) if d_R <= out_features else 0.0
    lam_l = comb(out_features, 2) / comb(in_features, d_L) if d_L <= in_features else 0.0
    lam = lam_r + lam_l
    return {
        "lambda_right": lam_r,
        "lambda_left": lam_l,
        "lambda_total": lam,
        "p_any_collision": 1.0 - exp(-lam),
    }


def automorphism_count(mask: np.ndarray) -> int | None:
    """Exact |Aut(G)| of the vertex-coloured bipartite graph, if pynauty is present.

    The colouring separating left from right vertices is REQUIRED: without it a
    square mask can pick up part-swapping automorphisms, which are not
    realisable as weight-space gauge moves. Returns None when pynauty is absent.
    """
    try:
        import pynauty  # type: ignore
    except ImportError:
        return None

    out, in_ = mask.shape
    adj = {i: [out + int(j) for j in np.flatnonzero(mask[i])] for i in range(out)}
    g = pynauty.Graph(
        number_of_vertices=out + in_,
        directed=False,
        adjacency_dict=adj,
        vertex_coloring=[set(range(out)), set(range(out, out + in_))],
    )
    _, grpsize1, grpsize2, _, _ = pynauty.autgrp(g)
    return int(round(grpsize1 * 10 ** grpsize2))


def apply_mask_permutation(mask: np.ndarray, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """P M Q^T for permutations given as gather-index arrays."""
    return mask[np.ix_(P, Q)]


def is_automorphism(mask: np.ndarray, P: np.ndarray, Q: np.ndarray) -> bool:
    return bool(np.array_equal(apply_mask_permutation(mask, P, Q), mask))
