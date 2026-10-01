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

Released predictions cover JEV 1.13 and KEV-0.8B/4B/9B. Raw source text and third-party base-model weights are not redistributed.

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

## Reconstruct evaluation prompts

```bash
python -m pip install -r requirements/data.txt
python scripts/rebuild_inputs.py --download --allow-unpinned
python scripts/verify_artifacts.py
```

The reconstructed prompts are written to `artifacts/frozen_inputs/` and must match the released hashes. Exact historical source revisions are unavailable for EmoBank Valence and Stanford Politeness; if their upstream data changed, verification will fail rather than silently accept different rows.

## Rerun KEV inference

```bash
python -m pip install -r requirements/kev.txt
python -m pip install --no-deps -e .
python scripts/download_models.py --models kev-0.8b,kev-4b
python scripts/run_kev_local.py --model kev-0.8b --input artifacts/frozen_inputs/main40.jsonl --output outputs/kev-0.8b-main40.jsonl
```

The released configuration pins KEV-0.8B and KEV-4B revisions. The historical KEV-9B revision was not recorded, so strict replay excludes it.

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
