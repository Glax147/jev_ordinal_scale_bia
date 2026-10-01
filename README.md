<h1 align="center">More Choices, Fewer Decisions</h1>

<p align="center">
  <strong>Ordinal-Scale Bias in JEV-like Direct-Decision Models</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.38827">
    <img src="https://img.shields.io/badge/arXiv-2609.38827-B31B1B?logo=arxiv&logoColor=white" alt="arXiv paper">
  </a>
  <a href="https://huggingface.co/Glax147/kev-0.8b-ba-lora">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-KEV--0.8B--BA--LoRA-FFD21E" alt="KEV-0.8B BA-LoRA on Hugging Face">
  </a>
  <a href="https://huggingface.co/Glax147/kev-4b-ba-lora">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-KEV--4B--BA--LoRA-FFD21E" alt="KEV-4B BA-LoRA on Hugging Face">
  </a>
</p>

This GitHub-ready folder consolidates the code, frozen selection identities,
released predictions, environment requirements, and model/data download tools
for the current JEV/KEV bias study. Upstream dataset text is intentionally not
redistributed: the four private frozen-input ZIP files are replaced by pinned
source downloaders, compact IDs, and per-row SHA-256 checks.

## Released Models

| Model | Versioned Hugging Face snapshot | Initialization | Qwen base |
|---|---|---|---|
| KEV-0.8B + BA-LoRA | [`Glax147/kev-0.8b-ba-lora@6f3864f`](https://huggingface.co/Glax147/kev-0.8b-ba-lora/tree/6f3864febcef263cd5d520d5520c0c5aa92472f0) | `jaredpalmer/kev-0.8b@9a45d25` | `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2` |
| KEV-4B + BA-LoRA | [`Glax147/kev-4b-ba-lora@3d5932a`](https://huggingface.co/Glax147/kev-4b-ba-lora/tree/3d5932a57af805c36e3ce51256316997b8494e31) | `jaredpalmer/kev-4b@139fdd94` | `Qwen/Qwen3.5-4B-Base@1001bb4d` |

The reproducibility configuration pins the complete published snapshots to
Hugging Face revisions `6f3864febcef263cd5d520d5520c0c5aa92472f0`
(0.8B) and `3d5932a57af805c36e3ce51256316997b8494e31` (4B). The
adapter, pointer head, and base-model pin were finalized at revisions
`048a489a541cc872f526d2853e592f01dfa99782` and
`7945a3045d890bfe8b92a6c283f103fee0b56e76`, respectively; the later snapshot
commits add the complete model cards and machine-readable environment records.

Each model repository contains the complete post-trained LoRA adapter, the
jointly trained KEV pointer head (`head.pt`), tokenizer files, training
configuration, metrics, and reproduction environment. It does not duplicate
the Qwen3.5 foundation weights. The final adapter is relative to the pinned
Qwen base and already contains the warm-started KEV adapter weights, so users
must not stack it on top of the original `jaredpalmer/kev-*` adapter. The
download script fetches both the released snapshot and its exact Qwen base;
`head.pt` is required for inference.

It covers three experiments:

- **main40:** 40 datasets × 5,000 items = 200,000 prompts per model;
- **position9:** 9 datasets × 5,000 items × 2 non-identity orders = 90,000 prompts per model;
- **kcurve4:** four datasets under controlled candidate counts `K=2..14` = 127,400 prompts per model.

Released predictions are included for JEV 1.13 and KEV-0.8B/4B/9B. Exact
post-training outputs plus a lightweight KEV BA-LoRA-inspired reproduction pipeline are
included under `posttraining/`. The default analysis reproduction needs neither
an API key nor a GPU.

## 1. One-command setup

Python 3.13 is the supported runtime family; `.python-version` records the
known-working patch release, 3.13.7.

Windows PowerShell:

```powershell
Set-Location "C:\path\to\jev_bias_evaluation_public"
.\scripts\setup.ps1 -Profile analysis
```

Linux/macOS:

```bash
PYTHON_BIN=python3.13 bash scripts/setup.sh
```

This creates `.venv`, installs the pinned analysis profile from `requirements/`, and
verifies the public indexes, released results, post-training assets, and code
tree. It does not download dataset text or model weights.

Recompute all metrics and figures:

```powershell
.\.venv\Scripts\python.exe scripts\one_click.py
```

Outputs go to `outputs/metrics/` and `outputs/figures/`.

## 2. Environment profiles

All dependency profiles live in one directory:

```text
requirements/base.txt          analysis, figures, API runner, and tests
requirements/data.txt          base profile plus dataset reconstruction
requirements/kev.txt           base profile plus local KEV inference
requirements/posttraining.txt  exact GPU post-training environment
```

Install a profile directly if preferred:

```bash
python -m pip install -r requirements/base.txt
python -m pip install --no-deps -e .
```

For local KEV inference, use Python 3.13 and install `requirements/kev.txt`.
Choose a PyTorch build appropriate for the installed CUDA driver if the pinned
wheel is not compatible with the machine.

## 3. Reconstructing the exact evaluation prompts

The public release omits the four frozen-text ZIP files. To download upstream
sources, reconstruct all three panels, and verify every reconstructed prompt
against its committed `paper-v1` canonical row hash, run:

```bash
python -m pip install -r requirements/data.txt
python scripts/rebuild_inputs.py --download --allow-unpinned
python scripts/verify_artifacts.py
```

Equivalent Windows bootstrap shortcut:

```powershell
.\scripts\setup.ps1 -RebuildInputs -AllowUnpinnedData
```

Equivalent Linux/macOS invocation:

```bash
REBUILD_INPUTS=1 ALLOW_UNPINNED_DATA=1 bash scripts/setup.sh
```

Rebuilt files are written to the git-ignored `artifacts/frozen_inputs/` folder.
The reconstruction validates sample IDs, order, labels, transformations,
candidate permutations, and the canonical SHA-256 of every prompt. A source
change fails loudly instead of silently producing a different benchmark.

The combined exact input hashes are:

| Experiment | Rows | Decompressed JSONL SHA-256 |
|---|---:|---|
| main40 | 200,000 | `3d17d8dd2ac58f8a773c5fc93bebd11f4169b83c84695da05ee19144383ca8e2` |
| position9 | 90,000 | `e80a3b23e1047bfc3ba5db1f767cf9d1de74abfbcc75c665a05ae77bed0278b2` |
| kcurve4 | 127,400 | `f3c6bab5ebc558e4e85ebd94543d6c3fd138efb82c965065e5b9fb1eab69f114` |

`data/frozen_inputs/MANIFEST.json` preserves the hashes of the private inputs as
an audit record; the ZIP paths it lists are intentionally absent from this
public distribution. `PUBLIC_RELEASE.json` is the machine-readable description
of what was omitted and how to reconstruct it.

### Upstream download only

Metric and figure reproduction from the released predictions does not require
dataset downloads. To fetch snapshots without rebuilding prompts:

```bash
python -m pip install -r requirements/data.txt
python scripts/download_datasets.py --sources pinned
```

Direct files are checked against configured SHA-256 values. Hugging Face and Git
sources record their resolved commits and a full local tree digest. A small
number of original upstream revisions were never recorded; they are excluded by
`pinned`. `--sources all --allow-unpinned` explicitly downloads the currently
resolved versions and records that they are not the original snapshots. Exact
reconstruction of all 40 sources therefore uses `--allow-unpinned`; the row
hashes still prevent a changed snapshot from being mistaken for `paper-v1`.

## 4. Model downloads and reruns

### KEV

Download the adapter **and its Qwen3.5 base** into the bundle-local Hugging Face
cache:

```powershell
.\scripts\setup.ps1 -Profile kev -Models "kev-0.8b,kev-4b"
```

or:

```bash
python scripts/download_models.py --models kev-0.8b,kev-4b
```

Download the released BA-LoRA checkpoints and their pinned bases with:

```bash
python scripts/download_models.py --models kev-0.8b-ba-lora,kev-4b-ba-lora
```

This command downloads the complete versioned Hugging Face snapshots listed
above and automatically resolves the corresponding Qwen3.5 base revisions.
No manual foundation-model download or adapter stacking is required. The first
download requires network access; subsequent runs can use `--offline` with the
bundle-local cache.

Every downloaded file is hashed and summarized by a tree SHA-256 in
`artifacts/models/download_manifest.json`. The runner rechecks both the adapter
and base-model trees before inference and records their hashes plus the verified
KEV Git revision in its sidecar. Re-run offline with:

```bash
python scripts/run_kev_local.py \
  --model kev-0.8b --offline \
  --input artifacts/frozen_inputs/main40.jsonl \
  --output artifacts/reruns/kev-0.8b-main.jsonl
```

The exact KEV-0.8B and KEV-4B adapter and base revisions are pinned in
`configs/models.json`, together with the two released BA-LoRA checkpoints. The
Qwen3.5-9B base is pinned, but the exact KEV-9B
adapter revision used by the released run was not preserved. Strict download
therefore refuses KEV-9B. `--allow-unpinned` records the resolved current commit
but must not be described as an exact rerun of the released 9B result.

### JEV 1.13

JEV is hosted and has no downloadable checkpoint in this project. Set credentials
only in the process environment (never in a committed file):

```powershell
$env:JEV_API_URL = "https://authorized-endpoint.example/v1/systemone"
$env:JEV_API_KEY = "<read from your secret manager>"
python scripts/run_jev_api.py --input artifacts/frozen_inputs/main40.jsonl `
  --output artifacts/reruns/jev-main.jsonl --workers 40
```

The original request name was `jev-1.13`; successful responses recorded
`typesafe/jev-1.13-20260917`. Remote reruns can vary in timestamps, latency and
service behavior. The runner checks this returned model identity by default and
stops on drift; `--allow-model-drift` is an explicit non-exact mode. Released
manifests also provide a digest over stable
prediction fields rather than claiming byte-identical API logs.

## 5. Reproducible sampling

`scripts/sample_candidates.py` creates a new, versioned sample. Allocation is
balanced across sources first and across strata within each source; exhausted
cells are redistributed deterministically. Within each source × stratum cell,
rows are ranked by:

```text
SHA256(seed \x1f source \x1f stratum \x1f sample_id)
```

Content duplicates are grouped using source, normalized question, context, and
candidate text; the representative is selected deterministically. Conflicting
labels and duplicate IDs are rejected. The sidecar manifest records the candidate
pool content digest, pre/post-deduplication cell counts, sampler-code hash,
ordered-ID hash, selection hash, and output hash.

The released benchmark remains named **paper-v1** because it was collected before
this sampler existed. It is reproduced from frozen IDs and per-row hashes; the
new hash sampler is not retroactively claimed to have produced it.

## 6. Integrity commands

```bash
python -m pytest -q
python scripts/verify_artifacts.py
```

In a clean public clone, the verifier confirms that private ZIP files are absent
and checks compact indexes, 12 released prediction files, status counts, duplicate IDs, exact
result-to-index ID sets, the optional post-training release manifest, and the
top-level code/config/requirements tree in `BUNDLE_MANIFEST.json`. A failure
is fatal. If prompts have been reconstructed, it additionally checks every
prompt's canonical row hash. Manifests are not silently rewritten.

## 7. Optional BA-LoRA-inspired post-training

`posttraining/` contains the frozen 32,000-row training index, pinned source and
model manifests, integration patches, and the released 0.8B/4B post-training
outputs. Its executable pipeline is centralized in `scripts/posttraining/`, and
its environment is pinned in `requirements/posttraining.txt`. It uses the `main40`
evaluation input reconstructed in `artifacts/frozen_inputs/` instead of
redistributing or duplicating 200,000 prompts. Rebuild that input first with
the command in Section 3; model weights and public raw training files are
downloaded only when requested:

```bash
bash scripts/posttraining/run.sh 0.8b 4b
```

The method adapts BA-LoRA ideas to KEV choice logits; the official BA-LoRA code
is reference-only and is not imported by training. See `posttraining/README.md`
for the GPU workflow and its reproducibility
boundary. The released answer files and Hugging Face checkpoints are independently
versioned records of the reported runs. The bundle has not yet performed a full
checkpoint replay over all 200,000 items, so the answer-file-to-checkpoint link
is not claimed as independently replay-verified. Retraining remains a seeded
procedural replication.

## 8. Layout

```text
configs/                 pinned experiments, sources, and model revisions
data/frozen_inputs/      private-input audit manifest; no upstream text ZIPs
data/indexes/            IDs, construction metadata, and canonical row hashes
requirements/            pinned environment profiles
results/released/        12 canonical compressed prediction files
results/MANIFEST.json    result/archive/content/stable-prediction hashes
posttraining/            optional KEV BA-LoRA configs, hashes, and released outputs
src/jev_bias/            deterministic I/O, sampling, metrics, integrity helpers
scripts/                 all setup, reconstruction, inference, analysis, and training entry points
tests/                   determinism and metric tests
artifacts/               extracted/downloaded runtime assets (ignored)
outputs/                 regenerated metrics and figures (ignored)
```

## License

Package code is MIT-licensed. Dataset text, labels, model code, and model weights
retain their upstream licenses and terms; this package does not relicense them.
See `THIRD_PARTY.md` before redistributing the bundle.
