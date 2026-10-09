# Additional StartLux and candidate-distance predictions

This directory supplements the existing JEV/KEV release. It contains 24 compressed prediction files and one text-free index for the new seven-level candidate-distance experiment. Each prediction file has one successful row per sample ID. Positions are zero-based; probability arrays follow the presented candidate order.

| Experiment | Models | Rows per model |
|---|---:|---:|
| Main panel, 40 datasets | StartLux 0.8B, 2B, 4B, 9B, 27B | 200,000 |
| Randomized order, 9 datasets | Same five | 90,000 |
| Candidate-count curve, 4 datasets | Same five | 127,400 |
| Candidate-distance curve, 4 datasets and 7 levels | StartLux 0.8B, 2B, 4B, 9B, 27B; KEV 0.8B, 4B, 9B | 138,894 |
| Candidate-distance curve, 3 datasets and 7 levels | JEV 1.13 | 104,118 |

The JEV seven-level result covers WMT20 Translation Quality, Measuring Hate Speech, and Stanford Politeness. It **does not** contain the Word Concreteness seven-level curve. The other eight models cover all four sources. Earlier two-endpoint runs are superseded by these seven-level files where coverage overlaps; they are not counted again.

The first three experiments use the frozen indexes in the original repository under `data/indexes/`. The new experiment's compact index is `data/indexes/candidate_distance_4datasets_7levels.index.jsonl.gz`. It includes question IDs, source/condition, candidate labels, gold positions and ordinal mappings, and hashes of the source contexts. It does not include source text. The generator is `scripts/build_candidate_distance_curve.py`; run it against the original sampled batch-3 input with seed `20261007` to reconstruct the questions. The resulting input SHA-256 must equal `9614df38664115e420096d6427fc158f390a1ecb59bb205428ca61adc355f6d7`.

`MANIFEST.json` records the original and released file hashes, exact row counts, retry deduplication, and per-source coverage. All 24 releases have complete success status and no repeated sample IDs. The historical JEV/KEV release files are unchanged.

For GitHub, copy this whole directory into `supplement/startlux_distance_release_20261009/` of the existing repository. Keeping it under `supplement/` leaves the original 12-file release and its verification manifest intact.
