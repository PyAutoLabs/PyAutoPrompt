# Remaining PyAutoPaper citation migration

Migrate every remaining legacy paper entry across `lensing_wiki`, `smbh_wiki`,
`cti_wiki`, `methods_wiki`, and `galaxies_wiki` in one consolidated PR based on
PyAutoPaper PR #2.

Verify canonical matches and claim context from papers or authoritative public
records. Add the canonical key, verified reference, relevant concepts, concise
support bullets, use guidance, and exclusion guidance. Remove local paths and
filename-inferred summaries. Use explicit TODOs for ambiguity. Keep concept and
entity links consistent, audit aliases, and run `make validate-literature-citations`
plus the repository tests.

The user explicitly replaced the earlier per-topic PR staging requirement with
one PR for all remaining entries. Topic-level commits and reports should still
be retained for reviewability.
