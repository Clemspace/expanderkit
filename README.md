# expanderkit

Fixed sparse masks for neural network construction, with the symmetry they leave
behind measured rather than assumed.

The repository and distribution are `expanderkit`; the importable package is
`ramanujan`, so existing imports do not churn. `import expanderkit` also works —
it re-exports `ramanujan` — so `pip install expanderkit; import expanderkit`
does not surprise anyone. Expander is the family — Ramanujan graphs are its
extremal case, random biregular graphs are the workhorse, and the package is
built to hold constructions that are neither.

A network's function is fixed by its parameters, but the map from parameters to
functions is many-to-one: permuting hidden units, or rescaling a unit against
the next layer, changes nothing the network computes. In a dense layer of width
`n` every permutation is available, so `n!` parameter vectors describe one
function. Under a **fixed mask** `M`, a permutation sends `M ↦ P M Qᵀ`, which is
a different network unless `(P, Q)` is an automorphism of the mask graph. The
redundancy collapses from `S_n` to `Aut(G)` — before training, at no runtime
cost, as a property of the architecture.

```
Dense layer, width n         |Γ| = n!             log|Γ| ~ n log n
Fixed LPS Ramanujan mask     |Γ| = |Aut(G)| ≥ n   log|Γ| ~ log n
Fixed quasi-cyclic mask      |Γ| ≥ block size     log|Γ| ~ log b
Fixed random biregular mask  |Γ| = 1  (usually)   log|Γ| = 0
```

"Usually" is doing real work in that last row — see
[the floor is a birthday condition](#the-floor-is-a-birthday-condition).

The same property that collapses the gauge also gives the layer a dense
`(out, d_L)` gather layout: constant fan-in means the live weights store as an
array with no mask beside them, which is simultaneously the efficient-kernel
form and a coordinate system in which two checkpoints are comparable entry by
entry. Biregularity does three jobs at once — spectral, computational,
representational — and that is the reason to build on it.

## Install

```bash
git clone https://github.com/Clemspace/expanderkit && cd expanderkit
pip install -e .            # numpy only
pip install -e '.[torch]'   # adds the layer
pip install -e '.[exact]'   # adds pynauty for exact |Aut(G)|
```

The core — construction, spectra, automorphisms — needs numpy alone and is
testable anywhere. Torch is confined to `ramanujan.layer`.

## Quickstart

```python
from ramanujan import MaskRegistry, analyse, certify_trivial_automorphisms

reg = MaskRegistry("~/.cache/candide/masks")
mask, config, spec = reg.get_or_build(
    out_features=128, in_features=512,
    target_sparsity=0.977, method="config_model", seed=0,
)

print(config)                                    # d_L=12, d_R=3, sparsity=0.9766
print(analyse(mask, config).summary())           # sigma_2 against the Feng-Li bound
print(certify_trivial_automorphisms(mask).summary())
```

As a layer:

```python
from ramanujan.layer import SparseLinear

layer = SparseLinear(512, 128, target_sparsity=0.977, registry=reg, seed=0)
theta  = layer.flat_parameters()      # live weights in mask order, then bias
gather = layer.dense_gather_weight()  # (out, d_L) — the kernel layout
```

`flat_parameters` is the point. Because the mask is fixed and shared, two
checkpoints from the same `MaskSpec` are comparable coordinate by coordinate,
with no permutation search between them and no dead entries carried along. That
vector — not a `state_dict`, 97.7% of whose coordinates at this sparsity are
decayed junk from masked positions — is what belongs in a model zoo.

## Layout

```
ramanujan/
  sparsity_grid.py   which (d_L, d_R) a shape admits; the min-degree floor
  masks.py           MaskSpec + samplers: matching, config_model, bernoulli
  spectral.py        one convention: raw sigma, Feng-Li bound, empirical nulls
  automorphisms.py   what symmetry a mask leaves; certification and generators
  registry.py        spec-keyed two-level cache, atomic writes
  layer.py           SparseLinear (torch)
  tests/test_core.py 38 tests, no pytest needed
```

This package answers questions about **masks**. Gauge measurement on parameter
vectors — the symmetry tax, scale canonicalisation, permutation groups of a real
architecture — lives in
[`wsaudit`](https://github.com/Clemspace/wsaudit), so that
auditing a dense model pulls in no sparse-mask code. Nothing here imports that
package and nothing there imports this one: automorphism generators cross the
boundary as plain `(P, Q)` index arrays, and a masked model's parameter vector
crosses as the output of `flat_parameters()`.

For the paper, pin both repositories by commit SHA rather than by URL. The
artifact is a *combination*, and a reader who clones `main` of each six months
apart gets one you never ran.

## Construction method is part of a mask's identity

`MaskSpec` carries shape, degrees, sampler, seed, spectral constraint, margin
and a `CONSTRUCTION_VERSION`, and the cache keys on all of it. This is not
bookkeeping. Two samplers produce different graphs from the same shape, degree
and seed, so a key that ignores the sampler serves one arm's mask to another —
the failure mode that produces a clean, well-formatted, meaningless result.

Two samplers, one interface:

- **`config_model`** — configuration model with double-edge-swap repair. Closest
  to the uniform distribution the asymmetry and spectral theorems are stated
  for. Use this wherever theory is load-bearing.
- **`matching`** — `d_L` successive capacity-ordered perfect matchings. Fast,
  always biregular, **not** uniform. Kept because it is what the published
  results used.

`bernoulli` is the unstructured control and is deliberately not biregular.

## Measured

30–40 configuration-model draws per shape, exact SVD, 2026-09-07:

| shape | d_L | d_R | sparsity | Ramanujan fraction | asymmetry certified |
|---|---|---|---|---|---|
| 128×512 | 12 | 3 | 0.9766 | 0.93 | 0.63 |
| 128×128 | 13 | 13 | 0.8984 | 1.00 | 1.00 |
| 64×64 | 6 | 6 | 0.9062 | 0.97 | 1.00 |
| 256×256 | 13 | 13 | 0.9492 | 1.00 | 1.00 |

**Ramanujan is close to generic at these sizes.** 93–100% of random biregular
draws already satisfy the Feng-Li bound, so a deterministic Ramanujan
construction buys a property a random draw almost always has — and a
"non-Ramanujan biregular" control has to reject most of what it samples. What
does the work at these scales is biregularity, not extremal spectral gap. Treat
"Ramanujan" as a property to verify per mask, not a label the construction
earns by name.

### The floor is a birthday condition

Kim, Sudakov & Vu give `3 ≤ d ≤ n − 4` for asymmetry of random *d*-regular
graphs, and degree 2 is a union of cycles. That is necessary, not sufficient. At
128×512 the mask satisfies `min(d_L, d_R) = 3` and still loses certified
asymmetry 37% of the time.

The mechanism is duplicate neighbourhoods: 512 right nodes of degree 3 drawn
from `C(128,3) = 341,376` possibilities collide often, and swapping two columns
wired to the same rows is a non-trivial automorphism. Over 40 draws,
certification is 1.00 with zero duplicates (n=27) and 0.00 with at least one
(n=13). The birthday estimate `C(512,2)/C(128,3) = 0.383` predicts a 0.318
collision probability against an observed 0.325 non-certification rate.

```python
from ramanujan import birthday_collision_estimate, duplicate_neighbourhood_swaps

birthday_collision_estimate(128, 512, d_L=12, d_R=3)  # p_any_collision ≈ 0.318
duplicate_neighbourhood_swaps(mask)                    # the concrete generators
```

Design condition: `C(out, d_R) ≫ C(in, 2)` on the right and symmetrically on the
left. It binds hardest on wide, very sparse layers — the regime this package was
built for.

## Certifying asymmetry

Colour refinement is useless here: every left vertex of a biregular graph has the
same degree, so 1-WL never splits anything. The certificate is spectral instead.
A side-preserving automorphism satisfies `P B Qᵀ = B`, so it permutes singular
vectors within each singular value's eigenspace; if a singular value is
**simple**, its vector maps to ±itself and `|u_k(i)|` is an automorphism
invariant. Colour each vertex by those magnitudes over the simple singular
values. If the colouring is discrete, every automorphism fixes every vertex, so
`Aut(G)` is trivial — a proof, subject only to the reported numerical
separation, which is computed by exact pairwise comparison rather than a sorted-
neighbour bound that could be loose in the unsafe direction.

The converse does not hold: a non-discrete colouring means *unresolved*, not
*symmetric*. For an exact order install pynauty and call `automorphism_count`,
which colours the two sides so a square mask cannot pick up part-swapping
automorphisms that are not realisable as gauge moves.

## Reproducing the published results

The pre-fix module initialised weights with `kaiming_uniform_` deriving fan-in
from `in_features` rather than `d_L`. At sparsity 0.9766 that is a variance
deficit of 12/512 — roughly a 6.5× shrink in activation scale per layer. The
bias was already correct, which is what made the mismatch invisible.

It also gave every row the same scale regardless of live degree. A biregular
mask has uniform row degree so all rows are equally misscaled; an unstructured
mask does not, so its low-degree rows are starved. **Any
biregular-vs-unstructured comparison run under that init confounds structure
with initialisation quality.** The flag exists so this can be measured rather
than argued:

```python
SparseLinear(512, 128, 0.977, legacy_dense_fanin=True)  # pre-fix behaviour
```

`FIXES.md` lists every change with the regression test that pins it. The pre-fix
state is tagged `v0-prefix` so the diff is inspectable.

## Tests

```bash
python -m ramanujan.tests.test_core
```

No pytest required. Torch-dependent tests skip with a notice rather than
failing. Tests marked `REGRESSION` each guard a specific defect and carry its
description in the docstring — read that before "fixing" one that starts
failing.

## Known limitations

- Verification, not construction, is the scaling wall. At n=2048, sparsity 0.98
  (d=41): construct 2.9s, exact SVD 2.4s, certify 9.0s. For larger graphs use a
  deflated power iteration for σ₂ (0.2s vs 2.3s at n=2048, ~0.5% relative error
  — fine for reporting, not for a near-bound Ramanujan verdict) and pynauty
  instead of the pairwise separation.
- `matching` is not uniform over biregular graphs. The asymmetry and spectral
  theorems are stated for uniform models; prefer `config_model` where that
  matters.
- Bipartite biregular asymmetry has no theorem behind it. KSV covers d-regular
  graphs only. Certify per mask; do not cite.
- One mask per zoo means one architecture per zoo. Stated, not solved.

## References

- Kim, Sudakov & Vu, *On the asymmetry of random regular graphs and random
  graphs*, Random Structures & Algorithms 22(1), 2002.
- Feng & Li, bound on the second eigenvalue of biregular bipartite graphs, 1996.
- Friedman, *A proof of Alon's second eigenvalue conjecture*, 2008; Brito,
  Dumitriu & Harris, *Spectral gap in random bipartite biregular graphs*, 2022.
- Biswas et al., *Sparse Network Initialization using Deterministic Ramanujan
  Graphs*, TF2M @ ICML 2024. [OpenReview](https://openreview.net/pdf?id=MehPKPYbLA)

## Licence

MIT.
