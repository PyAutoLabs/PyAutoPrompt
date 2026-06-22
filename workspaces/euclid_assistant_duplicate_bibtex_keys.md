# Detect duplicate BibTeX entry keys

Add a deterministic, provenance-backed warning to `euclid_assistant` for literal
BibTeX keys that are defined more than once in referenced `.bib` files or inline
`thebibliography` entries. Include focused tests and update generated and
human-readable rule documentation.

## Original request

You are working in the `euclid_assistant` repository (a deterministic style
  auditor for Euclid/A&A LaTeX papers). Read `AGENTS.md` first and follow it:
  rules must carry Style-Guide/PDD provenance, tests must pass, never modify the
  original source files under `knowledge/sources/`, and keep dependencies light
  (PyYAML only).

  # Task
  Add a deterministic check for **duplicate BibTeX entry keys** — the same entry
  key defined two or more times across the bibliography. This is Style Guide
  Sect. 2.6 item 8 (ii) ("Multiple occurrences of the same paper ... the .bib file
  contains the same paper multiple times"), p. 20. BibTeX silently keeps only one
  such entry, so duplicates are real bugs.

  # How the linter is structured (read these before coding)
  - `src/euclid_assistant/lint/rule_engine.py` — defines `Finding` (dataclass:
    rule_id, severity, title, scope, file, line, column, message, snippet, source)
    and `RuleEngine`. Function-type rules dispatch to `checks.REGISTRY[name]` with
    signature `fn(doc, rule, engine) -> List[Finding]`.
  - `src/euclid_assistant/lint/checks.py` — the check functions + `REGISTRY` dict.
    Look at `bibliography_not_euclid` for the pattern that locates `\bibliography`/
    `\addbibresource` arguments via `_command_arg_spans(code, cmd)`.
  - `src/euclid_assistant/lint/latex_scanner.py` — `FlatDoc` (flattened .tex);
    `doc.code` is the comment-stripped flattened source, `doc.main_path` is the
    main .tex path; `_command_arg_spans` extracts command arguments.
  - `src/euclid_assistant/ingest/extract_latex_assets.py` — has `_BIB_ENTRY =
    re.compile(r"@(\w+)\s*\{\s*([^,\s]+)", re.MULTILINE)` which extracts BibTeX
    (type, key). Reuse this regex shape for parsing .bib files.
  - `rules/euclid_rules.yaml` — rule definitions. Existing bib rules:
    `EUCLID-BIB-FILE`, `EUCLID-REF-IN-PREP`, `EUCLID-REF-ARXIV-EPRINTS`.
  - `rules/bibliography_rules.yaml` — already documents the approved journal
    abbreviations and the manual checks (context only; no code reads it yet).

  # Implement

  1. **New rule** in `rules/euclid_rules.yaml` (mirror the existing schema exactly):
     ```yaml
     - id: EUCLID-BIB-DUPLICATE-KEY
       title: 'Duplicate BibTeX entry key'
       severity: warning
       scope: bibliography
       source: {file: style_guide, section: "2.6 References (item 8)", page: 20}
       rationale: >
         'The same BibTeX entry key is defined more than once; BibTeX keeps only one
         and silently drops the rest, so citations may resolve to the wrong paper.'
       detection:
         type: function
         name: bibliography_duplicate_keys
       autofix: {safe: false}
       examples:
         bad: "@ARTICLE{Smith2020,...}\n@ARTICLE{Smith2020,...}"
         good: "@ARTICLE{Smith2020,...}\n@ARTICLE{Smith2020b,...}"
     ```

  2. **New check** `bibliography_duplicate_keys(doc, rule, engine)` in `checks.py`,
     registered in `REGISTRY`:
     - Collect candidate `.bib` files: for `cmd in ("bibliography",
       "addbibresource")`, read each arg via `_command_arg_spans(doc.code, cmd)`,
       split on commas, append `.bib` if there is no extension, and resolve
       relative to `doc.main_path.parent`. Skip files that do not exist.
     - Also collect inline keys from `\bibitem{key}` / `\bibitem[..]{key}` in
       `doc.code` (these live in the flattened .tex, so you can map them with
       `doc.offset_to_location`).
     - Parse each existing `.bib` with the `@type{key,` regex, recording every
       key together with its **source file (relative path) and 1-based line
       number** (compute the line by counting `\n` up to the match start).
     - Build `key -> list of (file, line)`. For any key occurring more than once,
       emit findings. Because .bib findings are NOT in the flattened .tex line map,
       construct `Finding` objects directly (import `Finding` from
       `.rule_engine`) rather than using `engine.make_finding`; for `\bibitem`
       duplicates you may use `engine.make_finding`. Use:
         severity=rule["severity"], title=rule["title"],
         scope=rule.get("scope","bibliography"), source=rule.get("source",{}),
         file=<relative bib path or doc file>, line=<lineno>, column=1,
         message=f"Duplicate BibTeX key '{key}' (also defined at <other locations>)."
       Emit one finding per duplicate occurrence after the first (so a key defined
       3x yields 2 findings), and make the message list the other locations.
     - Be robust: a missing/unreadable .bib must not crash — skip it. Comments in
       .bib start with `%`; ignoring them is optional (the `@type{` regex already
       avoids most false hits).

  3. **Tests** in a new `tests/test_bibliography.py` (mirror `tests/test_lint_checks.py`):
     - Use `tmp_path`. Write a `main.tex` with `\bibliography{refs}` and a `refs.bib`
       containing two `@ARTICLE{Dup2020,...}` entries plus a unique one. Flatten,
       run `RuleEngine(load_rules()).run(doc)`, assert a `EUCLID-BIB-DUPLICATE-KEY`
       finding exists, that its message names `Dup2020`, and that its `file` ends in
       `refs.bib`.
     - A clean `refs.bib` (all unique keys) yields no such finding.
     - A `\bibitem{X}` appearing twice in a `thebibliography` block is flagged.
     - Optional: assert no duplicate-key findings on the real corpus .bib files
       under `knowledge/sources/paper_sources/*/` (good false-positive check).

  4. **Docs**: add the rule to `wiki/rules/references.md` (one bullet citing
     Sect. 2.6 item 8, p. 20, with the `EUCLID-BIB-DUPLICATE-KEY` id), regenerate the
     coverage matrix with `python -m euclid_assistant.cli wiki`, and tick the
     bibliography item in `wiki/project/open-issues.md` (note duplicate-key detection
     is done; journal-name normalisation and the JCAP volume fix remain).

  # Run and validate (cache env vars matter on this machine)
  ```
  NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python -m pytest -q
  NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python -m euclid_assistant.cli rules | tail -1
  ```
  All existing tests must still pass, and `audit` on the four papers under
  `knowledge/sources/paper_sources/` must still report 0 errors (the new rule is a
  warning). Do NOT add Claude/AI co-author trailers to any commit. Leave the commit
  to me unless I ask you to commit; if you do commit, use a plain descriptive
  message and do not push.

  A couple of notes for you:

  - The trickiest part is that duplicate keys live in .bib files, which aren't part of the flattened .tex the linter normally walks — so the check has to locate, read, and line-number the .bib itself and
  build Finding objects directly. The prompt spells that out, which is where Codex would otherwise stumble.
  - I scoped it to literal duplicate keys (a real, deterministic BibTeX error). The Style Guide's "same paper under different labels" is a separate, fuzzier problem — I'd leave that out of this task.
  - Once that's in, the natural follow-ups (same prompt style) are journal-name normalisation against the approved-abbreviation list and the JCAP volume/issue fix, both already documented in
  rules/bibliography_rules.yaml.
