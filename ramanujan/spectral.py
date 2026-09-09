"""Spectral properties of biregular masks. One convention, stated once.

CONVENTION
----------
All singular values here are of the RAW biadjacency matrix B, not of any
normalised version. The biregular Ramanujan bound is then

    sigma_2(B) <= sqrt(d_L - 1) + sqrt(d_R - 1)

with sigma_1(B) = sqrt(d_L * d_R) exactly. The normalised quantities (divide
both by sqrt(d_L * d_R)) are reported as derived fields so a figure can use
either without a second code path. The previous module pair carried both
conventions in different files, which is a footgun rather than a bug.

The Alon-Boppana-type bound for biregular bipartite graphs is due to Feng & Li
(1996). Friedman (2008) proves random regular graphs are almost-Ramanujan; the
bipartite biregular analogue is Brito, Dumitriu & Harris (2022).

NO HAND-WAVY BASELINES
----------------------
The old ``random_bound`` was an admitted approximation with no derivation. It
is gone. If you want to know how a mask compares to chance, sample the null:
``empirical_sigma2_null`` returns the actual distribution of sigma_2 over draws
at the same shape and degree, which is both defensible and more informative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .sparsity_grid import DegreeConfig

EXACT_SVD_MAX_DIM = 2048
"""Above this min(out, in), exact SVD is expensive; approximation must be opted in."""


@dataclass
class SpectralReport:
    out_features: int
    in_features: int
    d_L: int
    d_R: int

    sigma_1: float
    sigma_2: float
    ramanujan_bound: float
    is_ramanujan: bool
    margin: float
    exact: bool
    biregular: bool
    singular_values: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def sigma_1_normalised(self) -> float:
        return self.sigma_1 / math.sqrt(self.d_L * self.d_R)

    @property
    def sigma_2_normalised(self) -> float:
        return self.sigma_2 / math.sqrt(self.d_L * self.d_R)

    @property
    def spectral_gap(self) -> float:
        return self.sigma_1 - self.sigma_2

    def summary(self) -> str:
        tag = "RAMANUJAN    " if self.is_ramanujan else "NOT RAMANUJAN"
        approx = "" if self.exact else "  [APPROX SVD]"
        return (
            f"[{tag}] {self.out_features}x{self.in_features} "
            f"d_L={self.d_L} d_R={self.d_R} | "
            f"s1={self.sigma_1:.4f} s2={self.sigma_2:.4f} "
            f"bound={self.ramanujan_bound:.4f} margin={self.margin:+.4f}{approx}"
        )


def ramanujan_bound(d_L: int, d_R: int) -> float:
    """Feng-Li bound for (d_L, d_R)-biregular bipartite graphs, raw scale."""
    return math.sqrt(max(d_L - 1, 0)) + math.sqrt(max(d_R - 1, 0))


def singular_values(mask: np.ndarray, allow_approximate: bool = False,
                    k: int = 8) -> tuple[np.ndarray, bool]:
    """Singular values of the raw biadjacency, exact unless opted out.

    Returns (values, exact). The approximate path is randomised subspace
    iteration; it is only entered when explicitly allowed, because an
    approximate sigma_2 can flip a Ramanujan verdict near the bound.
    """
    B = mask.astype(np.float64)
    if min(B.shape) <= EXACT_SVD_MAX_DIM:
        return np.linalg.svd(B, compute_uv=False), True
    if not allow_approximate:
        raise ValueError(
            f"exact SVD refused for shape {B.shape} (min dim > {EXACT_SVD_MAX_DIM}). "
            "Pass allow_approximate=True and treat sigma_2 near the bound as unresolved."
        )
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((B.shape[1], k + 8))
    for _ in range(8):
        Q, _ = np.linalg.qr(B.T @ (B @ Q))
    s = np.linalg.svd(B @ Q, compute_uv=False)
    return s, False


def analyse(mask: np.ndarray, config: DegreeConfig | None = None,
            allow_approximate: bool = False, tol: float = 1e-9) -> SpectralReport:
    """Spectral report for a mask. Degrees are read off the mask, not assumed."""
    rows = mask.sum(axis=1)
    cols = mask.sum(axis=0)
    biregular = bool((rows == rows[0]).all() and (cols == cols[0]).all())
    d_L = int(rows[0]) if biregular else int(round(rows.mean()))
    d_R = int(cols[0]) if biregular else int(round(cols.mean()))

    if config is not None and biregular and (d_L, d_R) != (config.d_L, config.d_R):
        raise ValueError(
            f"mask degrees ({d_L}, {d_R}) disagree with config ({config.d_L}, {config.d_R})"
        )

    s, exact = singular_values(mask, allow_approximate=allow_approximate)
    sigma_1 = float(s[0])
    sigma_2 = float(s[1]) if s.size > 1 else 0.0
    bound = ramanujan_bound(d_L, d_R)

    return SpectralReport(
        out_features=int(mask.shape[0]),
        in_features=int(mask.shape[1]),
        d_L=d_L,
        d_R=d_R,
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        ramanujan_bound=bound,
        is_ramanujan=biregular and sigma_2 <= bound + tol,
        margin=sigma_2 - bound,
        exact=exact,
        biregular=biregular,
        singular_values=s[:16],
    )


def empirical_sigma2_null(config: DegreeConfig, method: str = "config_model",
                          n_draws: int = 100, seed: int = 0) -> dict:
    """Distribution of sigma_2 over independent draws at this shape and degree.

    This replaces the old analytic ``random_bound``. It also answers a question
    nobody seems to have published: what fraction of random biregular masks at
    realistic layer sizes actually satisfy the Ramanujan bound?
    """
    from .masks import MaskSpec, build_mask  # local import avoids a cycle

    vals = np.empty(n_draws)
    ram = np.zeros(n_draws, dtype=bool)
    for i in range(n_draws):
        spec = MaskSpec.from_config(config, method=method, seed=seed * 100_000 + i)
        rep = analyse(build_mask(spec), config)
        vals[i] = rep.sigma_2
        ram[i] = rep.is_ramanujan

    return {
        "n_draws": n_draws,
        "method": method,
        "bound": ramanujan_bound(config.d_L, config.d_R),
        "sigma2_mean": float(vals.mean()),
        "sigma2_sd": float(vals.std(ddof=1)) if n_draws > 1 else 0.0,
        "sigma2_min": float(vals.min()),
        "sigma2_max": float(vals.max()),
        "ramanujan_fraction": float(ram.mean()),
        "samples": vals,
    }
