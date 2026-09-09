"""Fixed-mask sparse linear layer. Torch is imported here and nowhere else.

Three changes from the previous version, each of which mattered:

1. INITIALISATION USED THE DENSE FAN-IN.
   ``kaiming_uniform_(self.weight, a=sqrt(5))`` derives fan_in from
   ``weight.shape[1] = in_features``, but only ``d_L`` of those entries are
   live. At 0.977 sparsity that is a variance deficit of 12/512, roughly a
   6.5x shrink in activation scale per layer. The bias was already correct,
   which is what makes the mismatch easy to miss.

   It also gives every row the SAME scale regardless of how many live
   connections it has. A biregular mask has uniform row degree so every row is
   equally (mis)scaled; an unstructured mask does not, so its low-degree rows
   are starved. Any comparison between the two under this init confounds
   "structure" with "who got a usable initialisation". ``legacy_dense_fanin``
   reproduces the old behaviour exactly so that confound can be measured
   rather than argued about.

2. THE MASK WAS PERSISTENT.
   ``register_buffer`` defaults to persistent=True, so every checkpoint in a
   zoo carried a copy of its mask. Here it is non-persistent and reconstructed
   from the spec at load time.

3. DEAD WEIGHTS RODE ALONG IN THE PARAMETER VECTOR.
   Masked entries receive exactly zero gradient but AdamW's decoupled decay
   still shrinks them, so they end up as small run-dependent junk. Feeding a
   raw state_dict to a weight-space encoder at 0.977 sparsity means 97.7% of
   every input coordinate is noise. ``flat_parameters`` returns only the live
   weights, in mask order -- the dense (out, d_L) gather layout, which is
   simultaneously the efficient-kernel layout and the gauge-fixed coordinate
   system.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("ramanujan.layer requires torch; the rest of the package does not") from exc

from .masks import MaskSpec, gather_index
from .registry import MaskRegistry
from .sparsity_grid import DegreeConfig, MIN_DEGREE_DEFAULT, select_degree, sparsity_range


class SparseLinear(nn.Module):
    """Linear layer on a fixed biregular mask."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        target_sparsity: float,
        bias: bool = True,
        method: str = "config_model",
        seed: int = 0,
        min_degree: int = MIN_DEGREE_DEFAULT,
        registry: MaskRegistry | None = None,
        legacy_dense_fanin: bool = False,
    ):
        super().__init__()
        config = select_degree(out_features, in_features, target_sparsity, min_degree)
        if config is None:
            rng = sparsity_range(out_features, in_features, min_degree)
            detail = (f"achievable range [{rng[0]:.4f}, {rng[1]:.4f}]" if rng
                      else "no degree meets the asymmetry floor; layer too small")
            raise ValueError(
                f"no valid biregular config for ({out_features}, {in_features}) "
                f"at min_degree={min_degree}: {detail}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.target_sparsity = target_sparsity
        self.legacy_dense_fanin = legacy_dense_fanin
        self.config: DegreeConfig = config
        self.spec: MaskSpec = MaskSpec.from_config(config, method=method, seed=seed)

        reg = registry if registry is not None else MaskRegistry()
        mask_np = reg.get(self.spec)

        # Non-persistent: reconstructed from self.spec, never shipped in a checkpoint.
        self.register_buffer("mask", torch.from_numpy(mask_np), persistent=False)
        self.register_buffer(
            "gather_idx", torch.from_numpy(gather_index(mask_np, config.d_L).astype(np.int64)),
            persistent=False,
        )

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        if not bias:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """Initialise on the EFFECTIVE fan-in, and zero the dead entries.

        PyTorch's default for nn.Linear is kaiming_uniform_(a=sqrt(5)), which
        works out to bound = 1/sqrt(fan_in). We keep that formula and simply
        use the fan-in the layer actually has.
        """
        fan_in = self.in_features if self.legacy_dense_fanin else self.config.d_L
        bound = 1.0 / math.sqrt(fan_in)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            self.weight.mul_(self.mask)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    # ------------------------------------------------------- weight space

    def flat_parameters(self) -> torch.Tensor:
        """Live weights in mask row-major order, then bias. The zoo's coordinates.

        Length is ``out * d_L (+ out)``. Deterministic given the spec, so two
        checkpoints from the same spec are directly comparable coordinate by
        coordinate -- which is the whole point of fixing the mask.
        """
        live = self.weight[self.mask]
        return live if self.bias is None else torch.cat([live, self.bias])

    def load_flat_parameters(self, flat: torch.Tensor) -> None:
        n = self.config.n_edges
        if flat.numel() != n + (0 if self.bias is None else self.out_features):
            raise ValueError(f"flat vector has {flat.numel()} entries, expected {n} + bias")
        with torch.no_grad():
            self.weight.zero_()
            self.weight[self.mask] = flat[:n]
            if self.bias is not None:
                self.bias.copy_(flat[n:])

    def dense_gather_weight(self) -> torch.Tensor:
        """(out, d_L) live weights, aligned with ``gather_idx``.

        The layout a gather-matmul kernel wants. Same numbers as
        ``flat_parameters`` minus the bias, reshaped.
        """
        return self.weight[self.mask].view(self.out_features, self.config.d_L)

    def zero_dead_weights(self) -> None:
        """Clear masked entries. Call after optimiser steps if weight decay is on."""
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"sparsity={self.config.sparsity:.4f} (target={self.target_sparsity:.4f}), "
            f"d_L={self.config.d_L}, d_R={self.config.d_R}, "
            f"method={self.spec.method}, seed={self.spec.seed}"
            + (", legacy_dense_fanin=True" if self.legacy_dense_fanin else "")
        )
