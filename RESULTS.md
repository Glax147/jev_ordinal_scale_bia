# Released predictions

`results/released/` contains one gzip JSONL for every model/experiment pair:
four models × three experiments = 12 files. Each file has exactly one row per
frozen `sample_id` after last-success canonicalization.

`results/MANIFEST.json` records:

- compressed archive bytes and SHA-256;
- exact decompressed JSONL bytes and SHA-256;
- row count, unique-ID count, status counts, and duplicate checks;
- ordered ID and order-independent selection digests;
- an order-independent stable prediction digest that excludes latency,
  timestamps, request counts, and token usage and compares probabilities at the
  shared four-decimal recorded precision;
- an order-independent decision digest that ignores probability precision,
  plus an ordered digest for the exact released row sequence.

The stable digest is useful for comparing reruns whose operational metadata must
differ. It does not imply hosted-model outputs are deterministic.

## Post-training checkpoints

The KEV BA-LoRA-inspired checkpoints corresponding to the post-training study
are released at:

- [Glax147/kev-0.8b-ba-lora](https://huggingface.co/Glax147/kev-0.8b-ba-lora/tree/6f3864febcef263cd5d520d5520c0c5aa92472f0)
  (complete snapshot `6f3864febcef263cd5d520d5520c0c5aa92472f0`;
  weights/configuration finalized at `048a489a541cc872f526d2853e592f01dfa99782`);
- [Glax147/kev-4b-ba-lora](https://huggingface.co/Glax147/kev-4b-ba-lora/tree/3d5932a57af805c36e3ce51256316997b8494e31)
  (complete snapshot `3d5932a57af805c36e3ce51256316997b8494e31`;
  weights/configuration finalized at `7945a3045d890bfe8b92a6c283f103fee0b56e76`).

Each snapshot contains the complete updated adapter and jointly trained pointer
head. Its pinned Qwen3.5 base is downloaded automatically by
`scripts/download_models.py`; the original KEV adapter is an initialization
source and must not be stacked again at inference time.

Their released 200,000-item predictions remain under
`posttraining/results/released/` and are covered by `posttraining/MANIFEST.json`.

Three KEV main-panel files retain one `rejected` row each. Analysis excludes
non-`ok` rows but verification requires those records to remain present so the
result ID set still equals the frozen input ID set.

Run:

```bash
python scripts/verify_artifacts.py
```

before analysis. To recompute all tables and figures from released predictions:

```bash
python scripts/one_click.py
```

Primary distributional reporting uses support deviation
`D_R = |N_eff(pred) / N_eff(gold) - 1|`, for which lower is better. The signed
ratio `R` is retained in machine-readable outputs for directional diagnosis.
