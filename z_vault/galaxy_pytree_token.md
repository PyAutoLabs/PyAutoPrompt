Make `Galaxy` instances survive a `jax.jit` round-trip as identity-stable dict keys, so
`AdaptImages.galaxy_image_plane_mesh_grid_dict` (and its sibling `galaxy_image_dict`) can be
looked up by the *same* galaxy reference inside and outside the JIT boundary.

__Why this prompt exists__

`fit-imaging-pytree-delaunay` (issue #453, PR shipping the Delaunay-pixelization PoC) uncovered
that `GalaxiesToInversion.image_plane_mesh_grid_list` reads
`self.adapt_images.galaxy_image_plane_mesh_grid_dict[galaxy]` — an **identity-keyed** dict
whose keys are `Galaxy` instances. After `jax.jit(self.fit_from)(instance)` returns, the
`Galaxy` objects reachable through the new `Tracer` are fresh Python instances produced by the
pytree unflatten cycle (`cls.__new__(cls)` + `setattr(...)`). Their `id()` does not match any
key in the dict (whose keys are still the *original* galaxies attached to `adapt_images` —
which rides as aux/static through the pytree), so the lookup fails.

This is the same class of bug that
@PyAutoPrompt/autolens/linear_light_profile_intensity_dict_pytree.md fixed for
`LightProfileLinear` via the `pytree_token` pattern: an itertools.count-backed monotonic id
attached to each instance at `__init__`, exposed as a regular `__dict__` field so it survives
unflatten, with `__hash__` / `__eq__` overridden to use the token and
`__exclude_identifier_fields__ = ("pytree_token",)` so `Identifier` ignores the token when
hashing the model.

__Where it breaks__

@PyAutoGalaxy/autogalaxy/galaxy/to_inversion.py:420-446 currently has a *narrow* fallback added
in PR #38 (fit-imaging-pytree-delaunay):

```python
for galaxy in self.galaxies.galaxies_with_cls_list_from(cls=aa.Pixelization):
    try:
        image_plane_mesh_grid = (
            self.adapt_images.galaxy_image_plane_mesh_grid_dict[galaxy]
        )
    except (AttributeError, KeyError, TypeError):
        image_plane_mesh_grid = None

    if image_plane_mesh_grid is None:
        # Fallback for JAX JIT: ... when the dict contains exactly one mesh-grid
        # entry, take that single value by insertion order — this is always correct
        # in the one-pixelised-source case (Delaunay/Hilbert image-mesh fits).
        try:
            dict_ = self.adapt_images.galaxy_image_plane_mesh_grid_dict
            vals = list(dict_.values()) if dict_ else []
            if len(vals) == 1:
                image_plane_mesh_grid = vals[0]
        except (AttributeError, TypeError, KeyError):
            pass

    image_plane_mesh_grid_list.append(image_plane_mesh_grid)
```

This works for the PoC (one pixelised source) but **silently picks the wrong grid** in any
multi-source-pixelization scenario — see for example
@PyAutoLens/test_autolens/lens/test_to_inversion.py:273-277,329-331 which already build
`galaxy_image_plane_mesh_grid_dict={galaxy_pix_0: ..., galaxy_pix_1: ...}` with two distinct
pixelised sources. As soon as a real model has two pixelised sources, the fallback's
"one entry → take it by insertion order" branch can no longer fire, the dict lookup still
fails (identity), and `image_plane_mesh_grid` stays `None` — same crash as before.

A similar identity-keyed dict pattern lives at
@PyAutoGalaxy/autogalaxy/galaxy/to_inversion.py:555 (`adapt_images.galaxy_image_dict[galaxy]`)
which has its own try/except-None fallback. Both readers want the same fix.

__What needs to change__

Apply the **`pytree_token` pattern** to `Galaxy` so `Galaxy.__hash__` / `Galaxy.__eq__` are
keyed on a monotonic id that survives pytree flatten→unflatten unchanged. The template is
@PyAutoGalaxy/autogalaxy/profiles/light/linear/abstract.py:49-63.

Concretely:

1. On `Galaxy` (@PyAutoGalaxy/autogalaxy/galaxy/galaxy.py), add:

   ```python
   _pytree_token_counter = itertools.count()
   __exclude_identifier_fields__ = ("pytree_token",)

   def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self.pytree_token = next(Galaxy._pytree_token_counter)

   def __hash__(self):
       return self.pytree_token

   def __eq__(self, other):
       return isinstance(other, Galaxy) and self.pytree_token == other.pytree_token
   ```

   (Adjust to match `Galaxy.__init__`'s actual signature and existing dunder
   implementations — `Galaxy` may already define `__eq__`/`__hash__` and those need
   reconciling. Check `OperateImageGalaxies` / `OperateDeflections` MROs for any conflict.)

2. Verify `pytree_token` rides through the existing `register_instance_pytree(Galaxy, ...)`
   call as a dynamic leaf (it's a plain int, set in `__dict__`, not in `no_flatten`). After
   the unflatten cycle, the new `Galaxy` instance must carry the same `pytree_token` value as
   the one stored in `adapt_images`'s dict keys. Cross-reference with how
   `LightProfileLinear.pytree_token` survives — same machinery applies.

3. **Remove** the narrow fallback added in
   @PyAutoGalaxy/autogalaxy/galaxy/to_inversion.py:428-442 — the principled fix supersedes
   it. The `try/except (AttributeError, KeyError, TypeError)` around the dict lookup at
   lines 421-426 may stay (it pre-existed the fallback), but the `if image_plane_mesh_grid
   is None` block and its single-value-fallback should go.

4. Same for the sibling fallback at
   @PyAutoGalaxy/autogalaxy/galaxy/to_inversion.py near line 555 (`galaxy_image_dict`) —
   re-audit; once `Galaxy.__hash__` is token-based, that lookup should succeed natively too.

5. Confirm `Identifier` (@PyAutoFit/autofit/mapper/identifier.py) ignores `pytree_token` when
   building the model identifier hash. The `__exclude_identifier_fields__` hook on
   `LightProfileLinear` is the proven mechanism — same hook on `Galaxy` should work, but the
   lookup happens via class MRO so verify by running the PyAutoFit identifier tests.

__Validation__

1. Re-run @autolens_workspace_test/scripts/jax_likelihood_functions/imaging/delaunay_pytree.py
   with the narrow fallback **removed** and the `pytree_token` fix in place. Must PASS:
   NumPy and JIT log_likelihoods agree to rtol=1e-4.

2. Build a multi-source-pixelization Delaunay variant (or extend
   @PyAutoLens/test_autolens/lens/test_to_inversion.py with a JIT-path test) that has two
   pixelised sources at different redshifts. Without the principled fix, the narrow fallback
   would either crash or silently return wrong grids. With `Galaxy.pytree_token`, the
   identity-keyed lookup must succeed for both galaxies.

3. Run the full PyAutoGalaxy test suite — `pytest test_autogalaxy/ -x` — under the worktree.
   Identifier-hash tests are the highest risk: any test that pickles a `Galaxy`, or hashes
   one for use as a dict key, must continue to behave the same way as before the change.

4. Run @autolens_workspace_test/scripts/jax_likelihood_functions/imaging/mge_pytree.py and
   the other already-shipped pytree variants (lp, rectangular, mge_group, delaunay) — none
   should regress.

__Scope boundary__

- **Only** about `Galaxy` identity through pytree round-trips. Do not touch the per-variant
  `fit_*_pytree_*.md` PoC scripts — they're already passing via the narrow fallback.
- Do **not** change the public shape of `AdaptImages.galaxy_image_plane_mesh_grid_dict` /
  `galaxy_image_dict` — downstream code treats them as `Dict[Galaxy, Grid2DIrregular]` /
  `Dict[Galaxy, Array2D]`. The keys remain `Galaxy` instances; only their `__hash__`/`__eq__`
  semantics change.
- Do **not** regress the NumPy path. With `use_jax=False`, `Galaxy` instances were already
  unique objects per construction; switching to a token-based `__hash__`/`__eq__` should be
  a no-op for the NumPy code path because each `Galaxy.__init__` produces a fresh token.

__Starting points__

- @PyAutoGalaxy/autogalaxy/profiles/light/linear/abstract.py:49-63 — the proven template
- @PyAutoGalaxy/autogalaxy/galaxy/galaxy.py — class to modify
- @PyAutoGalaxy/autogalaxy/galaxy/to_inversion.py:420-446,555 — fallback to remove
- @PyAutoFit/autofit/mapper/identifier.py:127-128 — `__exclude_identifier_fields__` hook
- @PyAutoFit/autofit/jax/pytrees.py — `register_instance_pytree` machinery
- @PyAutoLens/test_autolens/lens/test_to_inversion.py:273-277,329-331 — multi-source
  pixelization fixture (use as scaffold for the multi-source JIT test)

__Deliverables__

1. PyAutoGalaxy library PR adding `pytree_token` to `Galaxy`, with `__exclude_identifier_fields__`
   wired in, plus removal of both narrow fallbacks in `to_inversion.py`.
2. PyAutoLens test PR (or PyAutoGalaxy if the test belongs there) adding a multi-source
   pixelization JIT round-trip that fails with the narrow fallback and passes with the
   principled fix.
3. PR body `## API Changes` section noting that `Galaxy.__hash__`/`__eq__` are now
   token-based (an internal change with no public API surface), and that the JAX JIT path
   for any `Pixelization` source now works without dict-lookup fallback heuristics.
