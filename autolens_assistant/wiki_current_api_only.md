Follow-up to the autolens_assistant API-drift task (stale `al.Kernel2D` / version-pin
baseline + drift-check). That task establishes the policy that the wiki documents **only
the current API**, with a version pin + drift-check replacing migration tables — but it
deliberately leaves the existing migration content in place so its own PR stays focused.
This task does the cleanup the policy enables.

Goal: remove the transitional migration scaffolding so `wiki/core/` describes only the
current API surface.

Work:

- **Delete** `@autolens_assistant/wiki/core/api_deltas_2026_05.md` (a dated 2026.05
  old->new migration reference covering Plotting, FITS, PSF/Convolver, mesh renames,
  Tracer, Cosmology, "Removed entirely").
- **Fix the 10 inbound links** that point at it — several say "see api_deltas for the full
  mapping". Rewrite each sentence to describe the current API directly instead of pointing
  at a mapping. Linkers (verify with `git grep api_deltas` before starting; may have
  changed):
  - skills: `_style.md`, `al_inspect_source_reconstruction.md`, `al_plot_fit_residuals.md`,
    `al_plot_tracer.md`
  - wiki: `api/plotting.md`, `concepts/inversions_and_pixelizations.md`,
    `concepts/lensing_basics.md`, `concepts/tracer.md`, `stack/autolens.md`
- **Strip the inline old->new notes** ("replaces the previous `Kernel2D`", "renamed from",
  "previously", "is removed") across `wiki/core/` (datasets.md, stack/autoarray.md,
  stack/autolens.md, concepts/*). Keep only current-API descriptions.
- Re-run the API audit (`work/audit_skill_apis.py --scope all`) afterwards to confirm no
  cited symbol now dangles, and the drift-check still passes.

Rationale: migration tables grow without bound every release and are themselves a
drift surface (they name removed symbols). With the version pin + drift-check from the
parent task, a user on the wrong autolens is told to upgrade rather than served an
old->new lookup — so the deltas doc is redundant going forward.

Do not start until the parent baseline/drift-check task has shipped (it defines the
current-API-only policy this references).
