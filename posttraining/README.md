# Optional KEV BA-LoRA-inspired choice-space post-training

This directory preserves the procedural recipe used for the paper's KEV-0.8B and KEV-4B post-training intervention. The KEV patch implements a **BA-LoRA-inspired adaptation in choice-logit space**; it does not import or execute the official BA-LoRA implementation. The upstream BA-LoRA repository is pinned and cloned as a reference-only source. This directory deliberately does **not** duplicate model weights, public raw data, or the 200,000-row evaluation file. The scripts download pinned assets, reconstruct the exact 32,000-record training set from a frozen index, train new adapters, and evaluate them on the parent package's hash-validated `main40` reconstruction.

The public package omits frozen dataset-text ZIPs. Before running the one-click
training workflow, reconstruct `main40` from the repository root:

```bash
python -m pip install -r requirements/data.txt
python scripts/rebuild_inputs.py --download --allow-unpinned
```

## Released Models

| Model | Versioned Hugging Face snapshot | Initialization |
|---|---|---|
| KEV-0.8B + BA-LoRA | [`Glax147/kev-0.8b-ba-lora@6f3864f`](https://huggingface.co/Glax147/kev-0.8b-ba-lora/tree/6f3864febcef263cd5d520d5520c0c5aa92472f0) | `jaredpalmer/kev-0.8b@9a45d25` |
| KEV-4B + BA-LoRA | [`Glax147/kev-4b-ba-lora@3d5932a`](https://huggingface.co/Glax147/kev-4b-ba-lora/tree/3d5932a57af805c36e3ce51256316997b8494e31) | `jaredpalmer/kev-4b@139fdd94` |

The bundle pins the complete published snapshots to revisions
`6f3864febcef263cd5d520d5520c0c5aa92472f0` and
`3d5932a57af805c36e3ce51256316997b8494e31`, respectively. The
adapter, pointer head, and Qwen base pin were finalized at revisions
`048a489a541cc872f526d2853e592f01dfa99782` and
`7945a3045d890bfe8b92a6c283f103fee0b56e76`; the later commits add
model-card and environment documentation. Each repository contains the complete
updated adapter and jointly trained `head.pt`, while the exact Qwen3.5 base is
downloaded separately and automatically. The original KEV checkpoint is the
training initialization and is not an additional inference-time adapter.

## Included

- exact 32,000-row training index (8 datasets × 4,000 records) with source IDs, labels, and normalized-text SHA-256 values;
- pinned source/model manifests with per-file SHA-256 values;
- the KEV code revision, the reference-only BA-LoRA revision, and the two KEV choice-space adaptation patches;
- exact top-level Python requirements used for the released runs;
- complete released 200,000-row outputs and score summaries for the 0.8B and 4B BA-LoRA runs, compressed deterministically;
- scripts for download, reconstruction, hash verification, training, and evaluation.

The training datasets are IBM Argument Quality, HelpSteer2 Correctness, Civil Comments Toxicity, Measuring Hate Speech, Wine Quality, Word Concreteness, WMT20 Translation Quality, and RealToxicity Continuation.

Recompute the paired before/after table (40 datasets × 2 model sizes) with:

```bash
python scripts/posttraining/analyze_changes.py
```

The table reports Accuracy, `Uarg`, raw `R`, bias distance `D_R=|R-1|`, TVD,
normalized MAE, QWK, `Usoft`, and margin before/after plus their changes.

## Run

Linux, Python 3.13, Git, a CUDA-capable PyTorch setup, and sufficient GPU memory are required. From the bundle root:

```bash
bash scripts/posttraining/run.sh 0.8b 4b
```

Run only one size by passing either `0.8b` or `4b`. The pipeline:

1. clones KEV at the commit in `configs/posttraining.json` (set `CLONE_BA_LORA_REFERENCE=1` only to fetch the unused reference source);
2. applies the checked-in KEV patches;
3. creates the environment from `requirements/posttraining.txt`;
4. downloads the exact model revisions and verifies every expected model file;
5. downloads the eight pinned public training sources;
6. reconstructs the exact training rows and verifies every ID, label, and normalized-text hash;
7. trains the requested model(s) with seed 42;
8. evaluates on `artifacts/frozen_inputs/main40.jsonl` and writes results beneath `posttraining/results/generated/`.

For preparation without training:

```bash
bash scripts/posttraining/setup.sh
cd posttraining/code/kev
uv run --no-sync python ../../../scripts/posttraining/download_models.py 0.8b 4b
uv run --no-sync python ../../../scripts/posttraining/download_raw_data.py
uv run --no-sync python ../../../scripts/posttraining/prepare_kev_training_data.py
uv run --no-sync python ../../../scripts/posttraining/verify_package.py
uv run --no-sync python ../../../scripts/posttraining/verify_models.py --full-hash
```

## Reproducibility boundary

The released answer files, current recipe, and published Hugging Face checkpoints are independently versioned and hash-verifiable. The bundle has not yet rerun each published checkpoint over the complete 200,000-item panel, so it does **not** claim an independently replay-verified cryptographic link from checkpoint to released answer file. Retraining uses frozen samples, pinned code/models, fixed hyperparameters, and seed 42, but CUDA kernels can still prevent bit-for-bit identical weights. Treat retraining as procedural replication; use the pinned Hugging Face checkpoints for the released model artifacts and the included answer files as the immutable record of the reported evaluation runs.

## Known limitation: legacy loss scaling

The released recipe retains the historical scaling used in the reported runs: the choice-space diversity and SVD losses are computed as microbatch means and then divided by the accumulation group's record count. Their effective coefficients therefore depend on microbatch size (approximately an additional `1/32` factor for the 0.8B setting and `1/8` for 4B). The nominal coefficients are **not directly comparable across model sizes**, and the released results must not support causal cross-size claims about regularization strength. Correcting this scaling requires a versioned recipe and fresh training; this package intentionally does not silently change the historical procedure.

All paths recorded in released summaries are portable labels or bundle-relative paths; local machine paths and relay/provider metadata are not included.
