#!/usr/bin/env python3
"""Dependency-free test suite for the ramanujan package.

Runs on numpy alone. Torch-dependent tests are skipped with a printed notice
rather than failing, so the suite is meaningful in a bare container and complete
on the GPU box.

    python -m ramanujan.tests.test_core

Every test in the REGRESSION section corresponds to a specific defect found in
the pre-fix module. They exist to stop those defects coming back, so if one of
them starts failing, read its docstring before "fixing" the test.
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ramanujan import (  # noqa: E402
    MIN_DEGREE_DEFAULT, DegreeConfig, MaskRegistry, MaskSpec,
    analyse, assert_biregular, birthday_collision_estimate, build_mask,
    certify_trivial_automorphisms, duplicate_neighbourhood_swaps,
    empirical_sigma2_null, enumerate_valid_degrees, gather_index,
    indices_to_mask, is_automorphism, mask_to_indices, ramanujan_bound,
    select_degree, singular_values, sparsity_range,
)
from ramanujan.masks import MaskConstructionError  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []
SKIPPED: list[tuple[str, str]] = []


def test(fn):
    """Minimal registry so the file needs no pytest."""
    fn._is_test = True
    return fn


class Skip(Exception):
    pass


# ===================================================================== grid

@test
def test_edge_conservation_holds_everywhere():
    for shape in [(64, 64), (128, 512), (256, 1024), (768, 3072), (512, 2048)]:
        configs = enumerate_valid_degrees(*shape)
        assert configs, f"no configs for {shape}"
        for c in configs:
            assert c.d_L * c.out_features == c.d_R * c.in_features


@test
def test_grid_is_sorted_sparsest_first():
    s = [c.sparsity for c in enumerate_valid_degrees(256, 256)]
    assert s == sorted(s, reverse=True)


@test
def test_min_degree_floor_is_enforced():
    """The KSV asymmetry floor is min(d_L, d_R) >= 3, not just d_L >= 3.

    For a wide layer d_R is the small one, and the old check that looked at
    both was correct -- this pins it so a future refactor cannot quietly drop
    the d_R half and reintroduce degree-2 masks, whose Aut is large.
    """
    for shape in [(128, 512), (512, 128), (64, 4096)]:
        for c in enumerate_valid_degrees(*shape, min_degree=MIN_DEGREE_DEFAULT):
            assert min(c.d_L, c.d_R) >= MIN_DEGREE_DEFAULT, (shape, c)


@test
def test_sparsest_valid_level_matches_published_operating_point():
    """(128, 512) at the floor gives d_L=12, sparsity 0.9766 -- the CompLearn point."""
    configs = enumerate_valid_degrees(128, 512)
    sparsest = configs[0]
    assert sparsest.d_L == 12 and sparsest.d_R == 3, sparsest
    assert abs(sparsest.sparsity - 0.9765625) < 1e-9


@test
def test_select_degree_picks_the_closest():
    for target in (0.5, 0.75, 0.9, 0.95, 0.99):
        c = select_degree(256, 256, target)
        best = min(abs(x.sparsity - target) for x in enumerate_valid_degrees(256, 256))
        assert abs(c.sparsity - target) <= best + 1e-12


@test
def test_impossible_shapes_report_rather_than_crash():
    assert select_degree(2, 2, 0.9) is None
    assert sparsity_range(2, 2) is None


@test
def test_degree_config_rejects_inconsistent_degrees():
    for bad in [(64, 64, 4, 5), (64, 64, 0, 0), (64, 64, 65, 65)]:
        try:
            DegreeConfig(*bad)
        except ValueError:
            continue
        raise AssertionError(f"DegreeConfig accepted {bad}")


# ==================================================================== masks

@test
def test_both_methods_produce_biregular_masks():
    for method in ("matching", "config_model"):
        for shape in [(64, 64), (128, 512), (64, 256)]:
            cfg = select_degree(*shape[::-1][::-1], 0.9) if False else select_degree(shape[0], shape[1], 0.9)
            m = build_mask(MaskSpec.from_config(cfg, method=method, seed=0))
            assert m.shape == (cfg.out_features, cfg.in_features)
            assert_biregular(m, cfg.d_L, cfg.d_R)


@test
def test_masks_are_reproducible_from_the_spec():
    cfg = select_degree(128, 128, 0.9)
    for method in ("matching", "config_model"):
        s = MaskSpec.from_config(cfg, method=method, seed=7)
        assert np.array_equal(build_mask(s), build_mask(s))


@test
def test_different_seeds_give_different_masks():
    cfg = select_degree(128, 128, 0.9)
    a = build_mask(MaskSpec.from_config(cfg, method="config_model", seed=0))
    b = build_mask(MaskSpec.from_config(cfg, method="config_model", seed=99))
    assert not np.array_equal(a, b)


@test
def test_bernoulli_control_is_not_biregular():
    cfg = select_degree(128, 128, 0.9)
    m = build_mask(MaskSpec(128, 128, cfg.d_L, cfg.d_R, "bernoulli", seed=0))
    rows = m.sum(axis=1)
    assert rows.min() != rows.max(), "bernoulli mask came out regular; check the sampler"


@test
def test_row_regular_control_is_row_regular_but_not_biregular():
    """The control that separates biregularity from dead-neuron elimination.

    A Bernoulli mask at d_L=3 over 128 inputs kills ~4.8% of output rows before
    training starts, which is most of what "biregularity avoids dead neurons"
    reports. Row-regular has zero dead rows and no column regularity, so the
    B vs row_regular contrast isolates what biregularity itself contributes.
    """
    cfg = select_degree(128, 128, 0.977)
    m = build_mask(MaskSpec.from_config(cfg, method="row_regular", seed=0))
    rows, cols = m.sum(axis=1), m.sum(axis=0)
    assert (rows == cfg.d_L).all(), "row degrees are not uniform"
    assert cols.min() != cols.max(), "column degrees are uniform; this is biregular, not the control"
    assert (rows == 0).sum() == 0

    b = build_mask(MaskSpec(128, 128, cfg.d_L, cfg.d_R, "bernoulli", seed=0))
    assert (b.sum(axis=1) == 0).sum() > 0, "bernoulli arm has no dead rows; check the density"


@test
def test_non_biregular_methods_reject_a_spectral_constraint():
    """Feng-Li applies to biregular graphs. Asking for a Ramanujan-filtered
    Bernoulli mask is a category error and must fail loudly rather than
    silently filtering on a bound that does not hold."""
    for method in ("bernoulli", "row_regular"):
        try:
            MaskSpec(128, 128, 3, 3, method, seed=0, constraint="ramanujan", margin=0.05)
        except ValueError:
            continue
        raise AssertionError(f"{method} accepted a spectral constraint")


@test
def test_gather_index_round_trips():
    cfg = select_degree(64, 256, 0.9)
    m = build_mask(MaskSpec.from_config(cfg, method="config_model", seed=1))
    g = gather_index(m, cfg.d_L)
    assert g.shape == (cfg.out_features, cfg.d_L)
    rebuilt = np.zeros_like(m)
    rebuilt[np.arange(cfg.out_features)[:, None], g] = True
    assert np.array_equal(rebuilt, m)


@test
def test_indices_round_trip():
    cfg = select_degree(128, 512, 0.95)
    m = build_mask(MaskSpec.from_config(cfg, method="matching", seed=3))
    assert np.array_equal(indices_to_mask(mask_to_indices(m), *m.shape), m)


@test
def test_matching_survives_layers_deeper_than_the_recursion_limit():
    """REGRESSION: the recursive augmenting-path search overflowed the stack.

    The old implementation recursed once per augmenting-path hop, so a layer
    with more left nodes than sys.getrecursionlimit() could raise
    RecursionError -- and the registry docstring advertised 4096-wide layers.
    """
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(120)  # far below the left-node count
    try:
        cfg = select_degree(1024, 1024, 0.99)
        m = build_mask(MaskSpec.from_config(cfg, method="matching", seed=0))
        assert_biregular(m, cfg.d_L, cfg.d_R)
    finally:
        sys.setrecursionlimit(old)


# ============================================================ REGRESSION: cache

@test
def test_spec_key_separates_construction_methods():
    """REGRESSION: the cache key hashed only (shape, degrees, seed).

    The Ramanujan arm and the non-Ramanujan control share all of those and
    differ only in sampler, so they collided: whichever built first served
    both arms. This is the single defect most likely to produce a clean,
    well-formatted, meaningless result.
    """
    cfg = select_degree(64, 64, 0.9)
    a = MaskSpec.from_config(cfg, method="matching", seed=0)
    b = MaskSpec.from_config(cfg, method="config_model", seed=0)
    assert a.key != b.key
    assert not np.array_equal(build_mask(a), build_mask(b))


@test
def test_spec_key_separates_spectral_constraints():
    """REGRESSION: same shape, same sampler, opposite spectral rejection."""
    cfg = select_degree(64, 64, 0.9)
    base = MaskSpec.from_config(cfg, method="config_model", seed=0)
    assert base.key != replace(base, constraint="ramanujan", margin=0.05).key
    assert (replace(base, constraint="ramanujan", margin=0.05).key
            != replace(base, constraint="non_ramanujan", margin=0.05).key)


@test
def test_construction_version_invalidates_the_cache():
    """A sampler change must not be served from caches built by the old one."""
    cfg = select_degree(64, 64, 0.9)
    s = MaskSpec.from_config(cfg, method="config_model", seed=0)
    assert s.key != replace(s, version=s.version + 1).key


@test
def test_registry_serves_distinct_masks_per_arm():
    tmp = Path(tempfile.mkdtemp())
    try:
        reg = MaskRegistry(tmp)
        cfg = select_degree(64, 64, 0.9)
        a = reg.get(MaskSpec.from_config(cfg, method="matching", seed=0))
        b = reg.get(MaskSpec.from_config(cfg, method="config_model", seed=0))
        assert not np.array_equal(a, b), "registry served one arm's mask to the other"
        assert reg.stats()["misses"] == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_registry_honours_its_own_cache_dir():
    """REGRESSION: the old singleton ignored cache_dir after first construction.

    Two registries built with different directories silently shared one, so
    per-arm cache separation did nothing.
    """
    t1, t2 = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    try:
        cfg = select_degree(64, 64, 0.9)
        spec = MaskSpec.from_config(cfg, method="config_model", seed=0)
        MaskRegistry(t1).get(spec)
        assert list(t1.glob("*.npz")), "nothing written to the first cache dir"
        assert not list(t2.glob("*.npz")), "second registry wrote into the first's dir"
        r2 = MaskRegistry(t2)
        r2.get(spec)
        assert r2.stats()["misses"] == 1, "second registry saw the first's cache"
    finally:
        shutil.rmtree(t1, ignore_errors=True)
        shutil.rmtree(t2, ignore_errors=True)


@test
def test_registry_disk_cache_hits_on_a_fresh_instance():
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = select_degree(64, 64, 0.9)
        spec = MaskSpec.from_config(cfg, method="config_model", seed=0)
        first = MaskRegistry(tmp).get(spec)
        reg = MaskRegistry(tmp)
        second = reg.get(spec)
        assert np.array_equal(first, second)
        assert reg.stats()["hits"] == 1 and reg.stats()["misses"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================= spectral

@test
def test_sigma1_is_exactly_sqrt_dl_dr():
    cfg = select_degree(128, 128, 0.9)
    r = analyse(build_mask(MaskSpec.from_config(cfg, method="config_model", seed=0)), cfg)
    assert abs(r.sigma_1 - math.sqrt(cfg.d_L * cfg.d_R)) < 1e-8, r.sigma_1
    assert abs(r.sigma_1_normalised - 1.0) < 1e-9


@test
def test_one_spectral_convention():
    """REGRESSION: two modules used raw and normalised sigma_2 with matching
    bounds. Both were internally consistent, which is exactly how such a pair
    gets mixed up in a figure. There is now one convention plus derived fields.
    """
    cfg = select_degree(128, 128, 0.9)
    r = analyse(build_mask(MaskSpec.from_config(cfg, method="config_model", seed=0)), cfg)
    scale = math.sqrt(cfg.d_L * cfg.d_R)
    assert abs(r.sigma_2_normalised * scale - r.sigma_2) < 1e-9
    assert abs(r.ramanujan_bound - ramanujan_bound(cfg.d_L, cfg.d_R)) < 1e-12


@test
def test_approximate_svd_must_be_opted_into():
    """REGRESSION: the old verifier silently switched to a randomised SVD above
    a size threshold, so a Ramanujan verdict near the bound could be decided by
    approximation error without any mark on the output."""
    big = np.zeros((4096, 4096), dtype=bool)
    big[np.arange(4096), np.arange(4096)] = True
    try:
        singular_values(big, allow_approximate=False)
    except ValueError:
        pass
    else:
        raise AssertionError("exact SVD was not refused for an oversized mask")
    _, exact = singular_values(big, allow_approximate=True)
    assert exact is False, "approximate path must report itself as approximate"


@test
def test_analyse_rejects_a_config_that_does_not_match_the_mask():
    cfg = select_degree(64, 64, 0.9)
    m = build_mask(MaskSpec.from_config(cfg, method="config_model", seed=0))
    other = select_degree(64, 64, 0.5)
    try:
        analyse(m, other)
    except ValueError:
        return
    raise AssertionError("mismatched config was accepted")


@test
def test_empirical_null_replaces_the_handwavy_bound():
    cfg = select_degree(64, 64, 0.9)
    null = empirical_sigma2_null(cfg, n_draws=12, seed=1)
    assert null["n_draws"] == 12
    assert null["sigma2_min"] <= null["sigma2_mean"] <= null["sigma2_max"]
    assert 0.0 <= null["ramanujan_fraction"] <= 1.0


# ==================================================================== gauge

def _circulant(n: int, offsets: tuple[int, ...]) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for o in offsets:
            m[i, (i + o) % n] = True
    return m


@test
def test_random_biregular_masks_certify_as_asymmetric():
    """The design claim in one assertion: a random biregular mask leaves no
    permutation gauge. If this ever fails at d >= 3, the ladder in the paper
    loses its bottom rung."""
    cfg = select_degree(64, 64, 0.9)
    certified = 0
    for seed in range(5):
        m = build_mask(MaskSpec.from_config(cfg, method="config_model", seed=seed))
        if certify_trivial_automorphisms(m).certified_trivial:
            certified += 1
    assert certified >= 4, f"only {certified}/5 random biregular masks certified"


@test
def test_a_deliberately_symmetric_mask_does_not_certify():
    """A circulant mask is vertex-transitive, so |Aut| >= n. The certificate
    must refuse it rather than mistaking a degenerate spectrum for distinctness.
    """
    m = _circulant(64, (0, 1, 2))
    cert = certify_trivial_automorphisms(m)
    assert not cert.certified_trivial, cert.summary()


@test
def test_circulant_shift_really_is_an_automorphism():
    """Sanity check on the negative case: exhibit the symmetry explicitly."""
    n = 64
    m = _circulant(n, (0, 1, 2))
    shift = (np.arange(n) + 1) % n
    assert is_automorphism(m, shift, shift)


@test
def test_certification_refuses_rather_than_approximating_when_too_large():
    from ramanujan.automorphisms import MAX_EXACT_SEPARATION_N
    n = MAX_EXACT_SEPARATION_N + 8
    m = np.zeros((n, n), dtype=bool)
    m[np.arange(n), np.arange(n)] = True
    cert = certify_trivial_automorphisms(m)
    assert not cert.certified_trivial and "too large" in cert.reason








@test
def test_duplicate_neighbourhoods_are_real_automorphisms():
    """The mechanism behind the birthday floor, verified rather than asserted.

    Two columns wired to the same rows can be swapped with the mask unchanged.
    Every generator this returns must actually be an automorphism, and a mask
    that has one must fail certification.
    """
    m = np.zeros((8, 6), dtype=bool)
    for j in range(6):
        m[[0, 1, 2], j] = True          # every column has the same neighbourhood
    gens = duplicate_neighbourhood_swaps(m)
    assert gens, "identical columns produced no generators"
    for P, Q in gens:
        assert is_automorphism(m, P, Q)
    assert not certify_trivial_automorphisms(m).certified_trivial


@test
def test_no_duplicates_means_no_swap_generators():
    cfg = select_degree(128, 128, 0.9)
    m = build_mask(MaskSpec.from_config(cfg, method="config_model", seed=0))
    assert duplicate_neighbourhood_swaps(m) == []
    assert certify_trivial_automorphisms(m).certified_trivial


@test
def test_birthday_estimate_matches_the_measured_collision_rate():
    """REGRESSION on the design rule, not the code.

    The degree floor (min degree >= 3) is necessary, not sufficient. At
    (128, 512, d_L=12, d_R=3) the estimate predicts collisions about a third of
    the time, and the measured non-certification rate matched it to three
    decimals. If a future change to the sampler breaks that agreement, the
    design rule in the README no longer describes what the code builds.
    """
    est = birthday_collision_estimate(128, 512, 12, 3)
    assert 0.28 < est["p_any_collision"] < 0.36, est
    cfg = select_degree(128, 512, 0.977)
    hits = sum(bool(duplicate_neighbourhood_swaps(
        build_mask(MaskSpec.from_config(cfg, method="config_model", seed=s))))
        for s in range(20))
    assert 0.10 < hits / 20 < 0.60, f"observed collision rate {hits/20}"


# ==================================================================== layer

def _torch_or_skip():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise Skip("torch not installed")


@test
def test_layer_init_uses_the_effective_fan_in():
    """REGRESSION: kaiming_uniform_ derived fan_in from in_features.

    Only d_L entries per row are live, so the old init shrank activations by
    sqrt(d_L / in_features) per layer -- about 6.5x at the published sparsity.
    The bias was already correct, which is what made the mismatch invisible.
    """
    _torch_or_skip()
    import torch
    from ramanujan.layer import SparseLinear

    tmp = Path(tempfile.mkdtemp())
    try:
        reg = MaskRegistry(tmp)
        fixed = SparseLinear(512, 128, 0.9766, registry=reg, seed=0)
        legacy = SparseLinear(512, 128, 0.9766, registry=reg, seed=0,
                              legacy_dense_fanin=True)
        d_L = fixed.config.d_L
        expected = 1.0 / math.sqrt(d_L)
        live = fixed.weight[fixed.mask].abs().max().item()
        assert live <= expected + 1e-6 and live > 0.5 * expected

        x = torch.randn(64, 512)
        ratio = fixed(x).std().item() / max(legacy(x).std().item(), 1e-12)
        assert ratio > 3.0, f"fixed init should be much wider; ratio={ratio:.2f}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_layer_zeroes_dead_weights_at_init():
    _torch_or_skip()
    from ramanujan.layer import SparseLinear
    tmp = Path(tempfile.mkdtemp())
    try:
        lay = SparseLinear(256, 64, 0.9, registry=MaskRegistry(tmp), seed=0)
        assert lay.weight[~lay.mask].abs().max().item() == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_mask_is_not_in_the_state_dict():
    """REGRESSION: register_buffer defaults to persistent=True, so every
    checkpoint in a zoo shipped a copy of its own mask."""
    _torch_or_skip()
    from ramanujan.layer import SparseLinear
    tmp = Path(tempfile.mkdtemp())
    try:
        lay = SparseLinear(256, 64, 0.9, registry=MaskRegistry(tmp), seed=0)
        keys = set(lay.state_dict())
        assert "mask" not in keys and "gather_idx" not in keys, keys
        assert keys == {"weight", "bias"}, keys
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_flat_parameters_round_trip_and_length():
    """The zoo's coordinate system: live weights only, deterministic order."""
    _torch_or_skip()
    import torch
    from ramanujan.layer import SparseLinear
    tmp = Path(tempfile.mkdtemp())
    try:
        lay = SparseLinear(256, 64, 0.9, registry=MaskRegistry(tmp), seed=0)
        flat = lay.flat_parameters().clone()
        assert flat.numel() == lay.config.n_edges + lay.out_features
        lay.weight.data.normal_()
        lay.load_flat_parameters(flat)
        assert torch.allclose(lay.flat_parameters(), flat, atol=0, rtol=0)
        assert lay.weight[~lay.mask].abs().max().item() == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_dense_gather_weight_agrees_with_flat():
    _torch_or_skip()
    import torch
    from ramanujan.layer import SparseLinear
    tmp = Path(tempfile.mkdtemp())
    try:
        lay = SparseLinear(256, 64, 0.9, registry=MaskRegistry(tmp), seed=0)
        g = lay.dense_gather_weight()
        assert g.shape == (lay.out_features, lay.config.d_L)
        assert torch.allclose(g.reshape(-1), lay.flat_parameters()[:lay.config.n_edges])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_masked_weights_get_no_gradient():
    _torch_or_skip()
    import torch
    from ramanujan.layer import SparseLinear
    tmp = Path(tempfile.mkdtemp())
    try:
        lay = SparseLinear(256, 64, 0.9, registry=MaskRegistry(tmp), seed=0)
        lay(torch.randn(8, 256)).sum().backward()
        assert lay.weight.grad[~lay.mask].abs().max().item() == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ====================================================================== main

def main() -> int:
    tests = [v for v in list(globals().values())
             if callable(v) and getattr(v, "_is_test", False)]
    print("=" * 72)
    print(f"ramanujan test suite -- {len(tests)} tests, numpy {np.__version__}")
    print("=" * 72)
    t0 = time.perf_counter()
    for fn in tests:
        name = fn.__name__
        try:
            fn()
        except Skip as s:
            SKIPPED.append((name, str(s)))
            print(f"SKIP  {name}  ({s})")
        except Exception as e:  # noqa: BLE001
            FAILED.append((name, f"{type(e).__name__}: {e}"))
            print(f"FAIL  {name}\n        {type(e).__name__}: {e}")
        else:
            PASSED.append(name)
            print(f"ok    {name}")
    dt = time.perf_counter() - t0
    print("=" * 72)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed, {len(SKIPPED)} skipped "
          f"in {dt:.1f}s")
    if SKIPPED:
        print("skipped: " + ", ".join(n for n, _ in SKIPPED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
