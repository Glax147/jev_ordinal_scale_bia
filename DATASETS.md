# Dataset provenance and selection

This public release does not redistribute the four ZIP archives that contain
upstream dataset text. Their archive hashes, decompressed-content hashes, row
counts, ordered ID hashes, and order-independent selection hashes remain in
`data/frozen_inputs/MANIFEST.json` as the audit identity of the private
`paper-v1` inputs.

The public package instead retains:

- `data/indexes/*.jsonl.gz`;
- `configs/sources.json`;
- source download configuration and integrity code;
- construction metadata and SHA-256 manifests.

Rebuild the prompts from their original providers and validate every row against
the committed canonical hash with:

```bash
python -m pip install -r requirements-data.txt
python scripts/rebuild_inputs.py --download --allow-unpinned
python scripts/verify_artifacts.py
```

Rebuilt files are written to the git-ignored `artifacts/frozen_inputs/`
directory. A mismatched source version, label, ordering decision, transformation,
or candidate permutation causes reconstruction to fail.

## Selection identities

- **paper-v1** is the exact evaluated selection. Each index row stores its
  `sample_id`, ordinal position, canonical row SHA-256, construction metadata,
  and the row without its usually large context field.
- **fresh hash samples** are produced by `scripts/sample_candidates.py` with the
  deterministic sampler described in the README. They are a new benchmark
  version and must not be paired with paper-v1 predictions.

## Panels

- `main40`: 40 sources × 5,000 = 200,000 prompts.
- `position9`: 9 sources × 5,000 × two non-identity permutations = 90,000.
- `kcurve4`: controlled `K=2..14` prompts from Civil Comments,
  RealToxicityPrompts, STSBenchmark, and DBpedia14 = 127,400.

The position sources are Civil Comments, DBpedia14, HelpSteer2, IBM Argument
Quality, RealToxicityPrompts, SHP, SNIPS, TripAdvisor, and Yahoo Answers.

## Upstream reconstruction boundary

`scripts/download_datasets.py` records resolved revisions and local tree hashes.
It is not needed to recompute metrics from the included prediction files, but it
is required when reconstructing prompt text or producing a new sample.

Some original source revisions were not captured immutably (notably EmoBank and
the ConvoKit-managed Stanford Politeness snapshot). The downloader refuses a
known unresolved source in strict mode. `--allow-unpinned` is an explicit opt-in
to a newly resolved and fully hashed snapshot, not a claim of historical identity.

## Historical-text quality note

The committed row hashes preserve the evaluated `paper-v1` prompt identity,
including upstream or historical decoding artifacts. In particular, 60 `main40` rows contain the
Unicode replacement character U+FFFD (381 occurrences), and 18 `position9` rows
contain it (148 occurrences); `kcurve4` contains none. These rows are retained so
the prompts remain aligned with the released predictions. Any cleaned-text study
must be versioned as a new benchmark rather than silently changing `paper-v1`.
