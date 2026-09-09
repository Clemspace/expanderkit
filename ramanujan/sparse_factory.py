"""
sparse_factory.py
-----------------
Universal factory for sparse linear layers, covering all four conditions
in experiment 5.3.

Conditions
----------
  D  — Dense             nn.Linear, no sparsity
  U  — Unstructured      Bernoulli mask, matched density, no structural guarantee
  B  — RandomBiregular   Augmenting-path matching, exact (d_L, d_R)-biregularity,
                         NO spectral filtering. σ₂ is measured and reported.
  R  — Ramanujan         Same as B, but resample until σ₂ ≤ √(d_L−1) + √(d_R−1).
                         Guarantees the Feng-Li Ramanujan bound.

All sparse conditions:
  - Use the same weight initialisation (Kaiming uniform, fan-in = d_L)
  - Apply the mask at forward time (weight * mask), never zero the weight tensor
  - Register mask as buffer so it serialises with the checkpoint
  - Raise rather than silently fall back to dense if no valid config exists

Design constraint: ALL layers of the transformer are sparse (including FFN
linear1, which was incorrectly left dense in all prior runs). This is now
fixed — gcd arithmetic confirms (512, 128) is constructible at d_L=3, d_R=12,
sparsity≈0.977.

Factory entry point
-------------------
    SparseLinearFactory.create(
        out_features, in_features, target_sparsity,
        mode, seed, min_degree, bias
    ) -> nn.Module

Transformer
-----------
    SparseGrokTransformer(mode, ...) uses the factory for all projections.
    Architecture is identical to exp 5.1 GrokTransformer except:
      - Explicit Q/K/V/O projections (enables per-projection sparsity)
      - All MLP layers sparse (linear1 and linear2)
      - Per-layer spectral reports available via .spectral_reports()
"""

from __future__ import annotations

import math
import warnings
from typing import Literal, Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ramanujan.sparsity_grid import DegreeConfig, select_degree, sparsity_range
from ramanujan.mask_registry import MaskRegistry
from ramanujan.mask_builder import build_biregular_mask
from ramanujan.spectral_verify import spectral_report, is_ramanujan, second_singular_value, ramanujan_bound

SparseMode = Literal["dense", "unstructured", "random_biregular", "ramanujan"]


# ── Condition D ───────────────────────────────────────────────────────────────

class DenseLinear(nn.Linear):
    """
    nn.Linear with a sparsity_info() stub for uniform downstream introspection.
    """
    def sparsity_info(self) -> dict:
        return {"mode": "dense", "sparsity": 0.0}

    def spectral_report(self) -> dict:
        return {"mode": "dense"}


# ── Condition U ───────────────────────────────────────────────────────────────

class UnstructuredSparseLinear(nn.Module):
    """
    Bernoulli mask at matched density. Not biregular.
    Row and column degrees are Binomial random variables.
    """
    def __init__(
        self,
        out_features: int,
        in_features: int,
        target_sparsity: float,
        bias: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.out_features    = out_features
        self.in_features     = in_features
        self.target_sparsity = target_sparsity

        gen = torch.Generator().manual_seed(seed)
        density = 1.0 - target_sparsity
        mask = torch.rand(out_features, in_features, generator=gen) < density
        self.register_buffer("mask", mask)

        row_sums = mask.sum(dim=1).float()
        col_sums = mask.sum(dim=0).float()
        self._actual_sparsity   = 1.0 - mask.float().mean().item()
        self._row_degree_mean   = row_sums.mean().item()
        self._row_degree_std    = row_sums.std().item()

        zero_rows = int((row_sums == 0).sum().item())
        if zero_rows > 0:
            warnings.warn(
                f"UnstructuredSparseLinear({out_features}, {in_features}) "
                f"produced {zero_rows} dead output neurons at sparsity={target_sparsity}.",
                RuntimeWarning,
            )

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in_eff = max(1, int(round((1 - self.target_sparsity) * self.in_features)))
            nn.init.uniform_(self.bias, -1.0 / math.sqrt(fan_in_eff),
                                         1.0 / math.sqrt(fan_in_eff))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    def sparsity_info(self) -> dict:
        return {
            "mode": "unstructured",
            "target_sparsity": self.target_sparsity,
            "actual_sparsity": self._actual_sparsity,
            "row_degree_mean":  self._row_degree_mean,
            "row_degree_std":   self._row_degree_std,
        }

    def spectral_report(self) -> dict:
        return {"mode": "unstructured"}  # no spectral structure to report


# ── Conditions B and R ────────────────────────────────────────────────────────

class BiregularSparseLinear(nn.Module):
    """
    Base class for biregular sparse linear layers (conditions B and R).
    Constructs the mask via augmenting-path matching (build_biregular_mask).
    Subclasses differ in whether they enforce the Ramanujan spectral bound.

    Parameters
    ----------
    verify_ramanujan : bool
        If True (condition R): resample until σ₂ ≤ Feng-Li bound.
        If False (condition B): accept the first sample, measure σ₂, report.
    max_attempts : int
        Maximum resampling attempts for condition R. Raises if exhausted.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        target_sparsity: float,
        bias: bool = True,
        seed: int = 0,
        min_degree: int = 3,
        verify_ramanujan: bool = False,
        max_attempts: int = 20,
        registry: Optional[MaskRegistry] = None,
    ):
        super().__init__()
        self.out_features    = out_features
        self.in_features     = in_features
        self.target_sparsity = target_sparsity
        self.verify_ramanujan = verify_ramanujan

        config = select_degree(out_features, in_features, target_sparsity, min_degree)
        if config is None:
            s_range = sparsity_range(out_features, in_features, min_degree)
            raise ValueError(
                f"No valid biregular config for ({out_features}, {in_features}) "
                f"target_sparsity={target_sparsity}, min_degree={min_degree}. "
                + (f"Achievable range: [{s_range[0]:.4f}, {s_range[1]:.4f}]"
                   if s_range else "Layer too small.")
            )
        self.config: DegreeConfig = config

        # Build mask — with or without Ramanujan verification
        mask = self._build_mask(config, seed, max_attempts)
        self.register_buffer("mask", mask)

        # Spectral properties (always measured, regardless of condition)
        self._sigma2 = second_singular_value(mask, config.d_L, config.d_R)
        self._bound  = ramanujan_bound(config.d_L, config.d_R)
        self._margin = self._bound - self._sigma2

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def _build_mask(
        self, config: DegreeConfig, seed: int, max_attempts: int
    ) -> torch.Tensor:
        if not self.verify_ramanujan:
            # Condition B: single build, no spectral filtering
            return build_biregular_mask(config, seed=seed)

        # Condition R: resample until Ramanujan bound is satisfied
        for attempt in range(max_attempts):
            mask = build_biregular_mask(config, seed=seed + attempt * 997)
            if is_ramanujan(mask, config.d_L, config.d_R):
                return mask
        raise RuntimeError(
            f"Could not construct Ramanujan-verified mask for "
            f"({config.out_features}, {config.in_features}) "
            f"d_L={config.d_L}, d_R={config.d_R} "
            f"after {max_attempts} attempts. "
            f"Last σ₂={second_singular_value(mask, config.d_L, config.d_R):.4f}, "
            f"bound={ramanujan_bound(config.d_L, config.d_R):.4f}."
        )

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.config.d_L  # sparse fan-in, not in_features
            nn.init.uniform_(self.bias, -1.0 / math.sqrt(fan_in),
                                         1.0 / math.sqrt(fan_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    def sparsity_info(self) -> dict:
        mode = "ramanujan" if self.verify_ramanujan else "random_biregular"
        return {
            "mode":            mode,
            "target_sparsity": self.target_sparsity,
            "actual_sparsity": self.config.sparsity,
            "d_L":             self.config.d_L,
            "d_R":             self.config.d_R,
            "sigma2":          round(self._sigma2, 4),
            "bound":           round(self._bound, 4),
            "margin":          round(self._margin, 4),
            "is_ramanujan":    self._margin >= -1e-6,
        }

    def spectral_report(self) -> dict:
        return self.sparsity_info()

    def extra_repr(self) -> str:
        mode = "R" if self.verify_ramanujan else "B"
        return (
            f"[{mode}] ({self.out_features}, {self.in_features}) "
            f"d_L={self.config.d_L} d_R={self.config.d_R} "
            f"sparsity={self.config.sparsity:.4f} "
            f"σ₂={self._sigma2:.3f} bound={self._bound:.3f} "
            f"margin={self._margin:+.3f}"
        )


# ── Factory ───────────────────────────────────────────────────────────────────

class SparseLinearFactory:
    """
    Universal factory for all four sparsity conditions.

    Usage
    -----
        layer = SparseLinearFactory.create(
            out_features=512, in_features=128,
            target_sparsity=0.977, mode="ramanujan", seed=42
        )

    If a biregular layer cannot be constructed (layer too small or shape
    incompatible with min_degree), raises ValueError — never silently falls
    back to dense. This is intentional: silent fallback would mean different
    conditions have different layer coverage, confounding the ablation.
    """

    @staticmethod
    def create(
        out_features: int,
        in_features: int,
        target_sparsity: float,
        mode: SparseMode,
        seed: int = 0,
        min_degree: int = 3,
        bias: bool = True,
    ) -> nn.Module:
        if mode == "dense":
            layer = DenseLinear(in_features, out_features, bias=bias)
            return layer

        if mode == "unstructured":
            return UnstructuredSparseLinear(
                out_features, in_features, target_sparsity,
                bias=bias, seed=seed,
            )

        if mode == "random_biregular":
            return BiregularSparseLinear(
                out_features, in_features, target_sparsity,
                bias=bias, seed=seed, min_degree=min_degree,
                verify_ramanujan=False,
            )

        if mode == "ramanujan":
            return BiregularSparseLinear(
                out_features, in_features, target_sparsity,
                bias=bias, seed=seed, min_degree=min_degree,
                verify_ramanujan=True,
            )

        raise ValueError(f"Unknown mode: {mode!r}. Choose from {SparseMode}.")

    @staticmethod
    def achievable_sparsity(
        out_features: int,
        in_features: int,
        target_sparsity: float,
        min_degree: int = 3,
    ) -> Optional[float]:
        """
        Returns the actual sparsity that will be used for a biregular layer,
        or None if the shape is incompatible. Useful for reporting before
        model construction.
        """
        config = select_degree(out_features, in_features, target_sparsity, min_degree)
        return config.sparsity if config is not None else None


# ── Transformer block ─────────────────────────────────────────────────────────

class SparseTransformerBlock(nn.Module):
    """
    Single transformer block with explicit Q/K/V/O projections and sparse MLP.
    All projection and MLP layers use the same mode and target sparsity.

    Layer shapes for d_model=128, d_mlp=512:
        Attention:  q/k/v/o_proj  (128, 128) — square
        FFN up:     linear1       (512, 128) — rectangular, 4:1 ratio
        FFN down:   linear2       (128, 512) — rectangular, 1:4 ratio

    All three shapes are constructible at sparsity≈0.977:
        (128, 128): d_L=d_R=3,  sparsity=1−3/128≈0.977
        (512, 128): d_L=3,d_R=12, sparsity=1−3/128≈0.977
        (128, 512): d_L=12,d_R=3, sparsity=1−12/512≈0.977
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        mode: SparseMode,
        target_sparsity: float,
        layer_idx: int = 0,
        seed_offset: int = 0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        def make(out_f, in_f, local_id):
            return SparseLinearFactory.create(
                out_features=out_f, in_features=in_f,
                target_sparsity=target_sparsity, mode=mode,
                seed=seed_offset + layer_idx * 100 + local_id,
            )

        # Attention projections (square)
        self.q_proj = make(d_model, d_model, 0)
        self.k_proj = make(d_model, d_model, 1)
        self.v_proj = make(d_model, d_model, 2)
        self.o_proj = make(d_model, d_model, 3)

        # MLP — both layers sparse, including FFN up (previously left dense)
        self.linear1 = make(d_mlp,    d_model, 4)  # up-projection   (512, 128)
        self.linear2 = make(d_model,  d_mlp,   5)  # down-projection (128, 512)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        # Attention (pre-norm, full non-causal)
        h = self.ln1(x)
        Q = self.q_proj(h).reshape(B, T, H, Dh).transpose(1, 2)
        K = self.k_proj(h).reshape(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(h).reshape(B, T, H, Dh).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Dh)
        attn   = F.softmax(scores, dim=-1)
        out    = (attn @ V).transpose(1, 2).reshape(B, T, D)
        x      = x + self.o_proj(out)

        # MLP (pre-norm)
        h = self.ln2(x)
        x = x + self.linear2(F.gelu(self.linear1(h)))

        return x

    def spectral_reports(self) -> List[dict]:
        """Per-layer spectral info for all sparse projections."""
        reports = []
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "linear1", "linear2"):
            layer = getattr(self, name)
            if hasattr(layer, "spectral_report"):
                rep = layer.spectral_report()
                rep["layer"] = name
                reports.append(rep)
        return reports


# ── Transformer model ─────────────────────────────────────────────────────────

class SparseGrokTransformer(nn.Module):
    """
    Transformer for modular arithmetic with all four sparsity conditions.
    Architecture identical to GrokTransformer (exp 5.1) except:
      - Explicit Q/K/V/O projections (enables per-projection sparsity)
      - ALL MLP layers sparse (linear1 now included, previously was dense)
      - .spectral_reports() exposes per-layer Ramanujan margins

    Input:  [a, op(=p), b, eq(=p+1)]  token indices, shape (B, 4)
    Output: logit vector over Z_p from final position
    """

    def __init__(
        self,
        p: int = 97,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        d_mlp: int = 512,
        mode: SparseMode = "dense",
        target_sparsity: float = 0.977,
        seed: int = 0,
    ):
        super().__init__()
        self.mode = mode

        vocab_size       = p + 2
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(4, d_model)
        self.blocks      = nn.ModuleList([
            SparseTransformerBlock(
                d_model=d_model, n_heads=n_heads, d_mlp=d_mlp,
                mode=mode, target_sparsity=target_sparsity,
                layer_idx=i, seed_offset=seed * 10_000,
            )
            for i in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, p, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h   = self.token_embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h)
        return self.unembed(self.ln_f(h[:, -1, :]))

    def spectral_reports(self) -> List[dict]:
        """
        All per-layer spectral reports across all blocks.
        For conditions B and R: includes σ₂, bound, margin, is_ramanujan.
        For D and U: mode field only.
        """
        reports = []
        for i, block in enumerate(self.blocks):
            for rep in block.spectral_reports():
                rep["block"] = i
                reports.append(rep)
        return reports

    def sparsity_summary(self) -> dict:
        """
        Aggregate Ramanujan margin stats across all biregular layers.
        Returns mean, min, and fraction satisfying the bound.
        """
        reports = self.spectral_reports()
        biregular = [
            r for r in reports
            if r.get("mode") in ("random_biregular", "ramanujan")
        ]
        if not biregular:
            return {"mode": self.mode, "n_sparse_layers": 0}

        margins   = [r["margin"]       for r in biregular]
        is_ram    = [r["is_ramanujan"] for r in biregular]
        return {
            "mode":                  self.mode,
            "n_sparse_layers":       len(biregular),
            "margin_mean":           round(sum(margins) / len(margins), 4),
            "margin_min":            round(min(margins), 4),
            "frac_ramanujan":        sum(is_ram) / len(is_ram),
        }