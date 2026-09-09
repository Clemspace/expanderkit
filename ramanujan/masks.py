"""Mask construction, with the construction method as first-class identity.

Four samplers, one interface -- two biregular, two controls:

  "row_regular"   exactly d_L entries per row, columns unconstrained. NOT
                  biregular. This is the control that separates "biregularity"
                  from "no dead output neurons": a Bernoulli mask at the same
                  density has P(row degree = 0) = (1-p)^in_features, which at
                  d_L = 3 over 128 inputs is ~4.8% of rows dead before training
                  starts. Row-regular removes that defect without adding column
                  regularity, so B vs row_regular isolates what biregularity
                  itself contributes. (Column degrees vary here, so some INPUT
                  units may feed nothing -- a different defect, and one the
                  forward pass tolerates.)

  "matching"      d_L successive random perfect matchings, capacity-ordered.
                  Fast, always biregular, but NOT uniform over biregular graphs
                  -- the capacity ordering biases the distribution. Kept because
                  it is what the published Ramanujan results used.

  "config_model"  configuration model plus double-edge-swap repair. Closer to
                  the uniform distribution the asymmetry and spectral theorems
                  are stated for. Prefer this for anything where the theory is
                  load-bearing.

The two produce different graphs from the same (shape, degree, seed). That is
precisely why ``MaskSpec`` carries ``method`` and why the registry keys on the
whole spec: an arm-blind cache silently serves one arm's mask to another.

Matching uses an ITERATIVE augmenting-path search. The recursive version
overflows Python's stack somewhere above a thousand left nodes, which is inside
the range of shapes this module advertises.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict

import numpy as np

from .sparsity_grid import DegreeConfig

CONSTRUCTION_VERSION = 1
"""Bump when a sampler's output distribution changes. Invalidates caches."""

METHODS = ("matching", "config_model", "row_regular", "bernoulli")


@dataclass(frozen=True)
class MaskSpec:
    """Everything that determines a mask. This is the cache identity.

    ``constraint`` records post-hoc spectral rejection sampling:
      "any"            first draw accepted
      "ramanujan"      resampled until sigma_2 <= bound - margin
      "non_ramanujan"  resampled until sigma_2 >= bound + margin
    """

    out_features: int
    in_features: int
    d_L: int
    d_R: int
    method: str
    seed: int
    constraint: str = "any"
    margin: float = 0.0
    version: int = CONSTRUCTION_VERSION

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; expected one of {METHODS}")
        if self.constraint not in ("any", "ramanujan", "non_ramanujan"):
            raise ValueError(f"unknown constraint {self.constraint!r}")
        if self.method in ("bernoulli", "row_regular") and self.constraint != "any":
            raise ValueError(
                f"{self.method} masks are not biregular, so the Feng-Li bound "
                "does not apply; constraint must be 'any'"
            )

    @classmethod
    def from_config(cls, config: DegreeConfig, method: str, seed: int, **kw) -> "MaskSpec":
        return cls(
            out_features=config.out_features,
            in_features=config.in_features,
            d_L=config.d_L,
            d_R=config.d_R,
            method=method,
            seed=seed,
            **kw,
        )

    @property
    def config(self) -> DegreeConfig:
        return DegreeConfig(self.out_features, self.in_features, self.d_L, self.d_R)

    @property
    def key(self) -> str:
        """Stable content hash over EVERY field, method and version included."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:32]


class MaskConstructionError(RuntimeError):
    """Raised only after all retries are exhausted."""


# ----------------------------------------------------------------- matching

def _augmenting_path_matching(adj: list[list[int]], n_left: int, n_right: int) -> list[int]:
    """Kuhn's algorithm, explicit stack. Returns match_left[i] = j or -1.

    Each stack frame is [u, next_edge_index, entering_v], where entering_v is
    the right node u currently holds (-1 for the search root). On reaching a
    free right node we walk the stack backwards reassigning, which is the
    augmentation.
    """
    match_left = [-1] * n_left
    match_right = [-1] * n_right

    for root in range(n_left):
        visited: set[int] = set()
        stack: list[list[int]] = [[root, 0, -1]]
        while stack:
            u, i, _ = stack[-1]
            if i >= len(adj[u]):
                stack.pop()
                continue
            stack[-1][1] += 1
            v = adj[u][i]
            if v in visited:
                continue
            visited.add(v)
            w = match_right[v]
            if w == -1:
                cur = v
                for frame in reversed(stack):
                    match_left[frame[0]] = cur
                    match_right[cur] = frame[0]
                    cur = frame[2]
                    if cur == -1:
                        break
                break
            stack.append([w, 0, v])
    return match_left


def _build_matching(config: DegreeConfig, rng: np.random.Generator) -> np.ndarray:
    out, in_ = config.out_features, config.in_features
    mask = np.zeros((out, in_), dtype=bool)

    for pass_k in range(config.d_L):
        col_remaining = config.d_R - mask.sum(axis=0)
        available = np.flatnonzero(col_remaining > 0)

        adj: list[list[int]] = []
        for i in range(out):
            valid = available[~mask[i, available]]
            if valid.size:
                # Capacity-descending, shuffled within ties: keeps popular
                # columns from being drained early. This is the step that
                # makes the sampler non-uniform -- see module docstring.
                caps = col_remaining[valid]
                order = np.lexsort((rng.random(valid.size), -caps))
                valid = valid[order]
            adj.append(valid.tolist())

        assignment = _augmenting_path_matching(adj, out, in_)
        if -1 in assignment:
            n_bad = sum(1 for j in assignment if j == -1)
            raise MaskConstructionError(
                f"matching failed at pass {pass_k}: {n_bad} left nodes unmatched"
            )
        mask[np.arange(out), assignment] = True

    return mask


# ------------------------------------------------------------- config model

def _build_config_model(config: DegreeConfig, rng: np.random.Generator) -> np.ndarray:
    """Configuration model + double-edge swaps until the multigraph is simple."""
    out, in_ = config.out_features, config.in_features
    d_L, d_R = config.d_L, config.d_R
    n_edges = config.n_edges

    rows = np.repeat(np.arange(out, dtype=np.int64), d_L)
    cols = np.repeat(np.arange(in_, dtype=np.int64), d_R)
    rng.shuffle(cols)

    loc: dict[tuple[int, int], list[int]] = {}
    for e in range(n_edges):
        loc.setdefault((int(rows[e]), int(cols[e])), []).append(e)

    max_swaps = 50 * n_edges
    stuck = 0
    swaps = 0
    while swaps < max_swaps:
        multi = [k for k, v in loc.items() if len(v) > 1]
        if not multi:
            break
        r, c = multi[rng.integers(len(multi))]
        moving = loc[(r, c)][-1]

        for _ in range(200):
            partner = int(rng.integers(n_edges))
            if partner == moving:
                continue
            rp, cp = int(rows[partner]), int(cols[partner])
            if rp == r or cp == c:
                continue
            if loc.get((r, cp)) or loc.get((rp, c)):
                continue
            loc[(r, c)].remove(moving)
            if not loc[(r, c)]:
                del loc[(r, c)]
            loc[(rp, cp)].remove(partner)
            if not loc[(rp, cp)]:
                del loc[(rp, cp)]
            cols[moving], cols[partner] = cp, c
            loc.setdefault((r, cp), []).append(moving)
            loc.setdefault((rp, c), []).append(partner)
            swaps += 1
            stuck = 0
            break
        else:
            stuck += 1
            if stuck >= 500:
                raise MaskConstructionError(
                    f"edge-swap repair stalled after {swaps} swaps"
                )

    if any(len(v) > 1 for v in loc.values()):
        raise MaskConstructionError("multi-edges remain after swap budget")

    mask = np.zeros((out, in_), dtype=bool)
    mask[rows, cols] = True
    return mask


# --------------------------------------------------------------- row regular

def _build_row_regular(config: DegreeConfig, rng: np.random.Generator) -> np.ndarray:
    """Exactly d_L live entries per row, columns chosen uniformly without
    replacement. Row-regular but not column-regular."""
    out, in_ = config.out_features, config.in_features
    mask = np.zeros((out, in_), dtype=bool)
    for i in range(out):
        mask[i, rng.choice(in_, size=config.d_L, replace=False)] = True
    return mask


# ----------------------------------------------------------------- bernoulli

def _build_bernoulli(config: DegreeConfig, rng: np.random.Generator) -> np.ndarray:
    """Unstructured control. NOT biregular; degrees vary row to row."""
    p = config.density
    return rng.random((config.out_features, config.in_features)) < p


# -------------------------------------------------------------------- public

_BUILDERS = {
    "matching": _build_matching,
    "config_model": _build_config_model,
    "row_regular": _build_row_regular,
    "bernoulli": _build_bernoulli,
}

BIREGULAR_METHODS = ("matching", "config_model")
"""Methods whose output is (d_L, d_R)-biregular; the others are controls."""


def assert_biregular(mask: np.ndarray, d_L: int, d_R: int) -> None:
    rows = mask.sum(axis=1)
    cols = mask.sum(axis=0)
    if not (rows == d_L).all():
        bad = np.flatnonzero(rows != d_L)[:5]
        raise AssertionError(f"row sums != {d_L}: {rows[bad]} at rows {bad.tolist()}")
    if not (cols == d_R).all():
        bad = np.flatnonzero(cols != d_R)[:5]
        raise AssertionError(f"col sums != {d_R}: {cols[bad]} at cols {bad.tolist()}")


def build_mask(spec: MaskSpec, max_retries: int = 8) -> np.ndarray:
    """Construct the mask a spec names.

    A failed draw is retried with a derived seed rather than raised, so one bad
    seed does not kill a sweep. Retries are deterministic given the spec.
    """
    builder = _BUILDERS[spec.method]
    config = spec.config
    last: Exception | None = None

    for attempt in range(max_retries):
        rng = np.random.default_rng([spec.seed, attempt, hash(spec.method) % (2**31)])
        try:
            mask = builder(config, rng)
        except MaskConstructionError as exc:
            last = exc
            continue
        if spec.method in BIREGULAR_METHODS:
            assert_biregular(mask, config.d_L, config.d_R)
        elif spec.method == "row_regular":
            rows = mask.sum(axis=1)
            if not (rows == config.d_L).all():
                raise AssertionError(f"row_regular mask has non-uniform row degree")
        return mask

    raise MaskConstructionError(
        f"could not build {spec.method} mask for {config} after {max_retries} "
        f"attempts; last error: {last}"
    )


def mask_to_indices(mask: np.ndarray) -> np.ndarray:
    """(2, nnz) int32 row/col indices in row-major order -- the storage form."""
    r, c = np.nonzero(mask)
    return np.ascontiguousarray(np.stack([r, c]).astype(np.int32))


def indices_to_mask(indices: np.ndarray, out_features: int, in_features: int) -> np.ndarray:
    mask = np.zeros((out_features, in_features), dtype=bool)
    mask[indices[0].astype(np.int64), indices[1].astype(np.int64)] = True
    return mask


def gather_index(mask: np.ndarray, d_L: int) -> np.ndarray:
    """(out, d_L) int32 column indices per row -- the dense fan-in layout.

    This is the layout that makes a biregular layer both a gather-matmul and a
    gauge-fixed coordinate system: weights store as a dense (out, d_L) array
    with no mask alongside.
    """
    assert_biregular(mask, d_L, int(mask.sum(axis=0)[0]))
    r, c = np.nonzero(mask)
    return c.reshape(mask.shape[0], d_L).astype(np.int32)
