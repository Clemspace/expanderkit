# Fixes applied, 2026-09-07

Commit the pre-fix module first, unmodified, then apply this on a branch. The
initialisation change touches a published result and the diff is the artifact
that makes the post-mortem checkable.

## Blocking

**1. The mask cache was arm-blind.** `_cache_key` hashed only
`(out, in, d_L, d_R, seed)`. The matching sampler and the configuration-model
sampler share all of those and differ only in construction, so they collided:
whichever built first served both arms. Replaced by `MaskSpec`, a frozen
dataclass carrying shape, degrees, `method`, `seed`, spectral `constraint`,
`margin` and a `version` counter, hashed whole. Bumping `CONSTRUCTION_VERSION`
invalidates every cache built by an older sampler.
Regression tests: `test_spec_key_separates_construction_methods`,
`test_spec_key_separates_spectral_constraints`,
`test_construction_version_invalidates_the_cache`,
`test_registry_serves_distinct_masks_per_arm`.

**2. Initialisation used the dense fan-in.** `kaiming_uniform_(weight,
a=sqrt(5))` derives `fan_in` from `in_features`, but only `d_L` entries per row
are live. At sparsity 0.9766 that is a variance deficit of 12/512 — roughly a
6.5x shrink in activation scale per layer. The bias was already correct, which
is what made the mismatch invisible.

It also gives every row the same scale regardless of live degree. A biregular
mask has uniform row degree so every row is equally misscaled; an unstructured
mask does not, so its low-degree rows are starved. **Any Ram-free vs Un-free
comparison run under this init confounds structure with initialisation
quality.** `legacy_dense_fanin=True` reproduces the old behaviour exactly so the
confound can be measured rather than argued about.
Regression test: `test_layer_init_uses_the_effective_fan_in`.

**3. Masks and dead weights rode along in every checkpoint.**
`register_buffer` defaults to `persistent=True`, so each zoo checkpoint shipped
a copy of its mask; and `weight` stayed a dense `(out, in)` parameter whose
masked entries take zero gradient but are still shrunk by decoupled weight
decay, leaving small run-dependent junk. Feeding raw state dicts to a
weight-space encoder at 0.9766 sparsity means 97.7% of every coordinate is
noise. Now `persistent=False`, plus `flat_parameters()` / `dense_gather_weight()`
returning live weights in mask order — simultaneously the gather-matmul layout
and the gauge-fixed coordinate system.
Regression tests: `test_mask_is_not_in_the_state_dict`,
`test_flat_parameters_round_trip_and_length`.

## Correctness and hygiene

- **Recursive matching overflowed the stack.** Augmenting-path search is now
  iterative with an explicit frame stack. Tested at recursion limit 120 on a
  1024-wide layer.
- **A failed matching killed the run.** `build_mask` retries with derived seeds
  before raising, deterministically.
- **The registry was a singleton that ignored `cache_dir`.** Per-arm cache
  directories silently became one shared directory. Now an ordinary instance.
- **Disk writes were not atomic.** A partially written cache file read by a
  sibling process is silent corruption, and seed sweeps run in parallel. Now
  temp-file plus rename.
- **Two spectral conventions in two modules.** One raw-scale convention
  (`sigma_2 <= sqrt(d_L-1) + sqrt(d_R-1)`, `sigma_1 = sqrt(d_L d_R)` exactly),
  normalised values as derived properties.
- **Approximate SVD was silent.** `singular_values` refuses above
  `EXACT_SVD_MAX_DIM` unless `allow_approximate=True`, and the report carries
  `exact`. An approximate `sigma_2` near the bound can flip a verdict.
- **`random_bound` was an undefensible formula.** Removed. `empirical_sigma2_null`
  samples the actual distribution instead.

## New: automorphism accounting (ramanujan.automorphisms)

- `certify_trivial_automorphisms` — proves `Aut(G)` trivial when an
  automorphism-invariant colouring from simple singular vectors is discrete.
  Colour refinement is useless on biregular graphs (uniform degree, nothing
  splits), hence the spectral route. Exact pairwise separation, never a
  sorted-neighbour bound, because a loose bound here would certify falsely.
  Refuses rather than approximating above 4096 per side.
- `automorphism_count` — exact `|Aut|` via pynauty when installed, on the
  vertex-coloured bipartite graph so a square mask cannot pick up
  part-swapping automorphisms that are not realisable as gauge moves.
- `duplicate_neighbourhood_swaps` / `birthday_collision_estimate` — the concrete
  generators and the design rule behind the birthday finding below.

Scale canonicalisation and the two G estimators moved to the audit package:
they act on parameter vectors, not masks, and auditing a dense model should not
pull in a sparse-mask library.

## Measured while testing, 2026-09-07

30–40 config-model draws per shape, exact SVD:

| shape | d_L | d_R | sparsity | Ramanujan fraction | asymmetry certified |
|---|---|---|---|---|---|
| 128x512 | 12 | 3 | 0.9766 | 0.93 | 0.63 |
| 128x128 | 13 | 13 | 0.8984 | 1.00 | 1.00 |
| 64x64 | 6 | 6 | 0.9062 | 0.97 | 1.00 |
| 256x256 | 13 | 13 | 0.9492 | 1.00 | 1.00 |

Two things fall out.

**Ramanujan is nearly generic at these sizes.** 93–100% of random biregular
masks already satisfy the bound, so a "non-Ramanujan biregular" control has to
reject most draws — and more importantly, the deterministic construction is
buying a property a random draw almost always has anyway.

**The asymmetry floor is a birthday condition, not a degree condition.** At
128x512 the mask loses certified asymmetry 37% of the time. The mechanism is
duplicate neighbourhoods: with 512 right nodes of degree 3 drawn from
C(128,3) = 341,376 possible neighbourhoods, two columns share a neighbourhood
often, and swapping them is a non-trivial automorphism. Measured over 40 draws,
certification is 1.00 when there are zero duplicate neighbourhoods (n=27) and
0.00 when there is at least one (n=13). The birthday estimate
`C(512,2)/C(128,3) = 0.383` gives `P(collision) = 0.318` against an observed
non-certification rate of 0.325.

So the design condition is `C(out, d_R) >> C(in, 2)` on the right side and
symmetrically on the left — sharper and more binding than `d >= 3`, and it
rules out exactly the wide-and-very-sparse layers this module was built for.

## Known, not fixed

- CORRECTION (2026-09-09): an earlier version of this file claimed the
  configuration-model sampler gets slow above 512x512. That was wrong. The
  slowdown was `certify_trivial_automorphisms` allocating a (chunk x n x dims)
  array -- about 4 GB at n=1024 with an uncapped colouring -- not the sampler.
  Fixed by MAX_COLOUR_DIMS. Measured at n=2048, sparsity 0.98 (d=41):
  construct 2.9s, analyse 2.4s, certify 9.0s. Verification is the wall,
  construction is not.
- The matching sampler's capacity-ordered greedy is not uniform over biregular
  graphs. The asymmetry and spectral theorems are stated for uniform models, so
  prefer `config_model` wherever the theory is load-bearing.
- Bipartite biregular asymmetry has no theorem behind it (Kim-Sudakov-Vu covers
  d-regular only). Certify per mask; do not cite.
