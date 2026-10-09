<h1 align="center">More Choices, Fewer Decisions</h1>

<p align="center"><strong>Ordinal-Scale Bias in JEV-like Direct-Decision Models</strong></p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.38827"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.38827-b31b1b.svg"></a>
  <a href="https://huggingface.co/Glax147/kev-0.8b-ba-lora"><img alt="KEV 0.8B + BA-LoRA" src="https://img.shields.io/badge/Hugging%20Face-KEV--0.8B--BA--LoRA-ffd21e.svg"></a>
  <a href="https://huggingface.co/Glax147/kev-4b-ba-lora"><img alt="KEV 4B + BA-LoRA" src="https://img.shields.io/badge/Hugging%20Face-KEV--4B--BA--LoRA-ffd21e.svg"></a>
</p>

This repository contains the frozen sample indexes, released predictions, analysis code, source-data downloaders, and optional KEV post-training code used in the paper.

| Experiment | Examples |
|---|---:|
| Main evaluation: 40 datasets | 200,000 |
| Position intervention: 9 datasets | 90,000 |
| Controlled candidate-count experiment: 4 datasets | 127,400 |

The original release covers JEV 1.13 and KEV-0.8B/4B/9B. The `supplement/startlux_distance_release_20261009/` release adds five StartLux sizes to those three experiments and a seven-level candidate-distance experiment for all nine models. The latter has four datasets for StartLux/KEV and three for JEV (Word Concreteness was not run for JEV). Raw source text and third-party model weights are not redistributed.

## Reproduce metrics and figures

Python 3.13 is recommended; the tested version is recorded in `.python-version`.

Windows PowerShell:

```powershell
Set-Location "path\to\jev_ordinal_scale_bia"
.\scripts\setup.ps1 -Profile analysis
.\.venv\Scripts\python.exe scripts\one_click.py
```

Linux/macOS:

```bash
PYTHON_BIN=python3.13 bash scripts/setup.sh
.venv/bin/python scripts/one_click.py
```

Tables and figures are written to `outputs/`.

To verify and analyze the additional results:

```bash
python scripts/analyze.py --supplement supplement/startlux_distance_release_20261009
```

The supplement contains 24 compressed prediction files, a text-free item index, and SHA-256/row-count manifests. The analysis writes `distance_by_condition.csv` alongside the extended main, position, and candidate-count tables. Use `--verify-only` to check the supplement without recomputing metrics.

## Reconstruct evaluation prompts

```bash
python -m pip install -r requirements/data.txt
python scripts/rebuild_inputs.py --download --allow-unpinned
python scripts/verify_artifacts.py
```

The reconstructed prompts are written to `artifacts/frozen_inputs/` and must match the released hashes. Exact historical source revisions are unavailable for EmoBank Valence and Stanford Politeness; if their upstream data changed, verification will fail rather than silently accept different rows.

For just the four candidate-distance sources, download and reconstruct the 20,000 frozen base items, then generate the seven-level versions:

```bash
python -m pip install -r requirements/data.txt
python scripts/rebuild_distance_inputs.py --download --allow-unpinned
```

This runs `download_datasets.py` for Word Concreteness, WMT20 Translation Quality, Measuring Hate Speech, and Stanford Politeness only, reconstructs each base item against its committed hash, and checks the generated 138,894-item file against the release SHA-256. The output is `artifacts/frozen_inputs/distance7.jsonl`. Stanford Politeness lacks a recorded historical snapshot revision, so current upstream content may fail the hash check. The released text-free index still supports exact analysis in that case. WMT20's dataset card lists its license as unknown; the derived file containing source text is therefore not redistributed here.

If `main40.jsonl` was already reconstructed, use `python scripts/rebuild_distance_inputs.py --base-input artifacts/frozen_inputs/main40.jsonl` to avoid downloading those sources again.

## Rerun KEV inference

```bash
python -m pip install -r requirements/kev.txt
python -m pip install --no-deps -e .
python scripts/download_models.py --models kev-0.8b,kev-4b
python scripts/run_kev_local.py --model kev-0.8b --input artifacts/frozen_inputs/main40.jsonl --output outputs/kev-0.8b-main40.jsonl
```

The released configuration pins KEV-0.8B and KEV-4B revisions. The historical KEV-9B revision was not recorded, so strict replay excludes it.

## Rerun StartLux inference

```bash
python -m pip install -r requirements/startlux.txt
python -m pip install --no-deps -e .
python scripts/download_models.py --models startlux-0.8b,startlux-2b,startlux-4b,startlux-9b,startlux-27b --allow-unpinned
python scripts/run_startlux_local.py --model startlux-0.8b --input artifacts/frozen_inputs/distance7.jsonl --output outputs/startlux-0.8b-distance7.jsonl
```

The downloader gets the official checkpoints and their bundled inference package, then records resolved Hugging Face commits and file hashes. The inference runner preserves candidate order, omits gold labels from requests, and can resume an interrupted output. Historical StartLux checkpoint revisions and some runtime settings were not recorded, so this is a **reference rerun**, not a claim of byte-identical predictions. Run the five sizes separately; 27B requires substantially more GPU memory.

## Rerun JEV inference

```powershell
$env:JEV_API_URL = "https://your-provider.example/v1"
$env:JEV_API_KEY = "YOUR_KEY"
python scripts/run_jev_api.py --input artifacts/frozen_inputs/main40.jsonl --output outputs/jev-1.13-main40.jsonl --workers 40
```

Hosted API outputs may not be byte-identical across providers or service revisions; the released prediction files are the evaluation record.

## Released post-trained KEV models

| Model | Adapter snapshot | Qwen base snapshot |
|---|---|---|
| KEV-0.8B + BA-LoRA | `Glax147/kev-0.8b-ba-lora@6f3864f` | `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2` |
| KEV-4B + BA-LoRA | `Glax147/kev-4b-ba-lora@3d5932a` | `Qwen/Qwen3.5-4B-Base@1001bb4d` |

Download the released adapters:

```bash
python scripts/download_models.py --models kev-0.8b-ba-lora,kev-4b-ba-lora
```

Recompute the released post-training comparison:

```bash
python scripts/posttraining/analyze_changes.py
```

To rerun training after reconstructing the inputs:

```bash
python -m pip install -r requirements/posttraining.txt
bash scripts/posttraining/run.sh 0.8b 4b
```

Training uses the fixed 32,000-example index in `posttraining/indexes/`, seed 42, and the settings in `posttraining/configs/`. Released checkpoint files and answer files are independently hash-verified; a complete checkpoint-to-answer replay has not been performed in this package.

## Integrity checks

```bash
python -m pytest -q
python scripts/verify_artifacts.py
python scripts/posttraining/verify_package.py --code-only
```

## License

Original code is released under the MIT License. Downloaded datasets, models, and hosted services remain subject to their upstream licenses and terms; their identifiers are recorded in `configs/`.
