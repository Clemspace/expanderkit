"""
ramanujan_layer.py
------------------
Drop-in replacement for nn.Linear with Ramanujan sparse mask.

Usage:
    # Direct construction (auto-selects closest sparsity)
    layer = RamanujanLinear(in_features=512, out_features=512, target_sparsity=0.9)

    # With shared registry (recommended for architectures with repeated shapes)
    registry = MaskRegistry()
    layer = RamanujanLinear(in_features=512, out_features=512,
                            target_sparsity=0.9, registry=registry)

    # Pre-build all masks before instantiating a model
    from ramanujan.mask_registry import MaskRegistry
    from ramanujan.sparsity_grid import select_degree
    specs = [(select_degree(512, 512, 0.9), 0), ...]
    MaskRegistry().prefetch(specs)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ramanujan.sparsity_grid import DegreeConfig, select_degree, sparsity_range
from ramanujan.mask_registry import MaskRegistry


class RamanujanLinear(nn.Module):
    """
    Linear layer with a fixed biregular Ramanujan sparse mask.
    Weights at masked positions are zeroed in forward(); they still
    receive gradients through the sparse positions only.

    The mask is a bool buffer -- not a parameter, not saved with the
    model's state_dict by default (it can be reconstructed from
    construction_info at load time).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        target_sparsity: float,
        bias: bool = True,
        seed: int = 0,
        min_degree: int = 3,
        registry: Optional[MaskRegistry] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.target_sparsity = target_sparsity
        self.seed = seed

        # Degree selection
        config = select_degree(out_features, in_features, target_sparsity, min_degree)
        if config is None:
            s_range = sparsity_range(out_features, in_features, min_degree)
            raise ValueError(
                f"No valid Ramanujan config for ({out_features}, {in_features}) "
                f"min_degree={min_degree}. "
                + (f"Achievable range: [{s_range[0]:.4f}, {s_range[1]:.4f}]"
                   if s_range else "Layer too small.")
            )
        self.config: DegreeConfig = config

        # Get mask (from registry if provided, else build directly)
        reg = registry or MaskRegistry()
        mask = reg.get_mask(config, seed=seed)  # bool tensor (out, in)
        self.register_buffer("mask", mask)

        # Parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform over the effective fan-in (sparse connections only)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.config.d_L  # actual connections, not in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"sparsity={self.config.sparsity:.4f} "
            f"(target={self.target_sparsity:.4f}), "
            f"d_L={self.config.d_L}, d_R={self.config.d_R}"
        )