"""Two-level mask cache keyed on the FULL construction spec.

The bug this replaces: the old key hashed only (out, in, d_L, d_R, seed), so the
Ramanujan arm and the non-Ramanujan control -- different samplers, same
arguments -- collided. Whichever built first silently served both. Every number
downstream would have been well-formatted and meaningless.

Here the key is a hash of the entire ``MaskSpec``, construction method and
version included, so a change to a sampler invalidates its caches rather than
serving stale masks under a new name.

Also not a singleton. The old ``__new__`` returned the first instance forever
and ignored later ``cache_dir`` arguments, so per-arm cache directories quietly
became one shared directory. Construct one registry and pass it around.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from .masks import MaskSpec, build_mask, indices_to_mask, mask_to_indices
from .sparsity_grid import DegreeConfig, MIN_DEGREE_DEFAULT, select_degree, sparsity_range


class MaskRegistry:
    """In-memory + on-disk cache. Explicit instance; not global state."""

    def __init__(self, cache_dir: str | os.PathLike | None = "~/.cache/candide/masks"):
        self._memory: dict[str, np.ndarray] = {}
        self._dir: Path | None = None
        if cache_dir is not None:
            self._dir = Path(cache_dir).expanduser()
            self._dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    # ---------------------------------------------------------------- api

    def get(self, spec: MaskSpec) -> np.ndarray:
        key = spec.key
        if key in self._memory:
            self.hits += 1
            return indices_to_mask(self._memory[key], spec.out_features, spec.in_features)

        path = None if self._dir is None else self._dir / f"{key}.npz"
        if path is not None and path.exists():
            with np.load(path) as z:
                indices = z["indices"]
            self.hits += 1
            self._memory[key] = indices
            return indices_to_mask(indices, spec.out_features, spec.in_features)

        self.misses += 1
        mask = build_mask(spec)
        indices = mask_to_indices(mask)
        self._memory[key] = indices
        if path is not None:
            self._write_atomic(path, indices, spec)
        return mask

    def get_or_build(
        self,
        out_features: int,
        in_features: int,
        target_sparsity: float,
        method: str = "config_model",
        seed: int = 0,
        min_degree: int = MIN_DEGREE_DEFAULT,
        **spec_kw,
    ) -> tuple[np.ndarray, DegreeConfig, MaskSpec]:
        config = select_degree(out_features, in_features, target_sparsity, min_degree)
        if config is None:
            rng = sparsity_range(out_features, in_features, min_degree)
            detail = (f"achievable range [{rng[0]:.4f}, {rng[1]:.4f}]" if rng
                      else "no degree meets the asymmetry floor; layer too small")
            raise ValueError(
                f"no valid biregular config for ({out_features}, {in_features}) "
                f"at min_degree={min_degree}: {detail}"
            )
        spec = MaskSpec.from_config(config, method=method, seed=seed, **spec_kw)
        return self.get(spec), config, spec

    def prefetch(self, specs: list[MaskSpec], verbose: bool = True) -> None:
        seen: set[str] = set()
        todo = [s for s in specs if not (s.key in seen or seen.add(s.key))]
        if verbose:
            print(f"[MaskRegistry] {len(todo)} unique specs")
        for i, spec in enumerate(todo, 1):
            before = self.misses
            self.get(spec)
            if verbose:
                how = "built" if self.misses > before else "cached"
                print(f"  [{i}/{len(todo)}] {how}: {spec.out_features}x{spec.in_features} "
                      f"d_L={spec.d_L} method={spec.method} key={spec.key[:8]}")

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "n_memory": len(self._memory),
            "cache_dir": str(self._dir) if self._dir else None,
        }

    def clear_memory(self) -> None:
        self._memory.clear()
        self.hits = self.misses = 0

    # ------------------------------------------------------------- internal

    @staticmethod
    def _write_atomic(path: Path, indices: np.ndarray, spec: MaskSpec) -> None:
        """Write via a temp file in the same directory, then rename.

        A partially written .npz read by a sibling process is a silent
        corruption, and multi-seed sweeps run in parallel.
        """
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        try:
            np.savez_compressed(tmp, indices=indices, spec_key=np.array(spec.key))
            os.replace(tmp + ".npz" if not tmp.endswith(".npz") else tmp, path)
        finally:
            for leftover in (tmp, tmp + ".npz"):
                if os.path.exists(leftover):
                    os.unlink(leftover)
