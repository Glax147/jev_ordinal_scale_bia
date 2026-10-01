# Third-party assets and terms

The root `LICENSE` covers only original bundle code. It does not relicense any
dataset text, released prediction, model, or upstream source project.

- **KEV** is downloaded from `jaredpalmer/kev` at the commits recorded in the
  configuration files and is distributed upstream under Apache-2.0. The
  post-training patch is applied to that checkout; retain KEV's notices when
  redistributing a patched source tree.
- **Qwen3.5 base models and KEV adapters** are downloaded on demand. Their
  repository/model-card licenses and acceptable-use terms apply.
- **Released BA-LoRA checkpoints** are published as
  [`Glax147/kev-0.8b-ba-lora@6f3864f`](https://huggingface.co/Glax147/kev-0.8b-ba-lora/tree/6f3864febcef263cd5d520d5520c0c5aa92472f0)
  and
  [`Glax147/kev-4b-ba-lora@3d5932a`](https://huggingface.co/Glax147/kev-4b-ba-lora/tree/3d5932a57af805c36e3ce51256316997b8494e31)
  on Hugging Face.
  They are initialized from the corresponding `jaredpalmer/kev-*` adapters and
  still require the Qwen3.5 base named in each checkpoint configuration. Their
  pinned revisions are recorded in `configs/models.json` and
  `posttraining/configs/posttraining.json`.
- **JEV 1.13** is a hosted service. This folder contains no JEV weights; access
  and use remain subject to the service provider's terms.
- **BA-LoRA** is cited and commit-pinned as the research reference for the
  choice-space-inspired adaptation. Its upstream repository is not bundled or
  imported by the training runtime. Check its upstream licensing status before
  redistributing its source.
- **Datasets** retain the licenses and terms listed by their original providers.
  This public release excludes the private frozen-prompt archives and ships only
  indexes, hashes, source download instructions, and derived prediction records,
  as described in `DATASETS.md`.
