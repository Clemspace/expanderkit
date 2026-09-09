"""
mask_registry.py
----------------
Two-level cache (in-memory + on-disk) for constructed masks.

Masks are stored as sparse COO indices (int32) on disk -- much smaller
than dense float32 tensors. Reconstructed to dense bool at load time.

Thread-safety: single-process use only. For multi-process training,
call prefetch() in the main process before spawning workers.
"""

import hashlib
import math
import torch
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from ramanujan.sparsity_grid import DegreeConfig, select_degree
from ramanujan.mask_builder import build_biregular_mask


class MaskRegistry:
    """
    Singleton registry. Construct once, use across all model instantiations
    in a session.

    Storage format: (2, nnz) int32 tensor of (row, col) indices.
    At 90% sparsity, a 4096x16384 layer has ~6.7M edges = ~54MB as indices,
    vs 256MB as dense float32.
    """
    _instance: Optional["MaskRegistry"] = None

    def __new__(cls, cache_dir: str = "~/.cache/candide/masks"):
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._memory: Dict[str, Tuple[torch.Tensor, DegreeConfig]] = {}
            inst._cache_dir = Path(cache_dir).expanduser()
            inst._cache_dir.mkdir(parents=True, exist_ok=True)
            inst._hits = 0
            inst._misses = 0
            cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_mask(self, config: DegreeConfig, seed: int = 0) -> torch.Tensor:
        """
        Return a boolean mask of shape (out, in) for the given config.
        Hits memory cache, then disk cache, then builds from scratch.
        """
        key = _cache_key(config, seed)

        if key in self._memory:
            self._hits += 1
            indices, _ = self._memory[key]
            return _indices_to_mask(indices, config)

        disk_path = self._cache_dir / f"{key}.pt"
        if disk_path.exists():
            self._hits += 1
            indices = torch.load(disk_path, weights_only=True)
            self._memory[key] = (indices, config)
            return _indices_to_mask(indices, config)

        self._misses += 1
        mask = build_biregular_mask(config, seed=seed)
        indices = mask.nonzero(as_tuple=False).T.to(torch.int32).contiguous()
        self._memory[key] = (indices, config)
        torch.save(indices, disk_path)
        return mask

    def get_or_build(
        self,
        out_features: int,
        in_features: int,
        target_sparsity: float,
        seed: int = 0,
        min_degree: int = 3,
    ) -> Tuple[torch.Tensor, DegreeConfig]:
        """
        Convenience: select best degree config and return (mask, config).
        Raises ValueError if no valid config exists for this shape.
        """
        config = select_degree(out_features, in_features, target_sparsity, min_degree)
        if config is None:
            raise ValueError(
                f"No valid Ramanujan config for shape ({out_features}, {in_features}) "
                f"with min_degree={min_degree}. "
                f"Layer may be too small."
            )
        mask = self.get_mask(config, seed=seed)
        return mask, config

    def prefetch(self, specs: List[Tuple[DegreeConfig, int]]) -> None:
        """
        Build and cache masks for a list of (config, seed) pairs.
        Call this before model construction to batch all expensive builds.

        Example:
            specs = [(select_degree(out, in_, sparsity), 0)
                     for (out, in_) in model_layer_shapes]
            MaskRegistry().prefetch(specs)
            # All subsequent get_mask() calls are cache hits
        """
        # Deduplicate: same (config, seed) used by multiple layers
        unique = {}
        for config, seed in specs:
            key = _cache_key(config, seed)
            if key not in unique and key not in self._memory:
                unique[key] = (config, seed)

        if not unique:
            return

        print(f"[MaskRegistry] Prefetching {len(unique)} unique masks...")
        for i, (key, (config, seed)) in enumerate(unique.items()):
            disk_path = self._cache_dir / f"{key}.pt"
            if disk_path.exists():
                indices = torch.load(disk_path, weights_only=True)
                self._memory[key] = (indices, config)
                print(f"  [{i+1}/{len(unique)}] Loaded from disk: {config.out_features}x{config.in_features} d_L={config.d_L}")
            else:
                mask = build_biregular_mask(config, seed=seed)
                indices = mask.nonzero(as_tuple=False).T.to(torch.int32).contiguous()
                self._memory[key] = (indices, config)
                torch.save(indices, disk_path)
                print(f"  [{i+1}/{len(unique)}] Built: {config.out_features}x{config.in_features} "
                      f"d_L={config.d_L} sparsity={config.sparsity:.4f}")

    def cache_stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "n_cached": len(self._memory),
            "cache_dir": str(self._cache_dir),
        }

    def clear_memory(self) -> None:
        """Release in-memory cache. Disk cache is unaffected."""
        self._memory.clear()
        self._hits = 0
        self._misses = 0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _cache_key(config: DegreeConfig, seed: int) -> str:
    payload = (
        f"{config.out_features}_{config.in_features}_"
        f"{config.d_L}_{config.d_R}_{seed}"
    )
    return hashlib.md5(payload.encode()).hexdigest()


def _indices_to_mask(indices: torch.Tensor, config: DegreeConfig) -> torch.Tensor:
    mask = torch.zeros(config.out_features, config.in_features, dtype=torch.bool)
    mask[indices[0].long(), indices[1].long()] = True
    return mask