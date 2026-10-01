# Frozen inputs are intentionally not redistributed

The four private/local ZIP archives are excluded from this public GitHub
release because they contain upstream dataset text governed by the original
providers' licenses. `MANIFEST.json` remains as an audit record of the exact
private evaluation inputs, but the archive paths it lists are intentionally
absent here.

Install the data profile, download upstream sources, rebuild the exact
`paper-v1` prompts, and verify every canonical row hash with:

```bash
python -m pip install -r requirements/data.txt
python scripts/rebuild_inputs.py --download --allow-unpinned
python scripts/verify_artifacts.py
```

Outputs are written to `artifacts/frozen_inputs/`, which is git-ignored. Most
sources are pinned to immutable revisions. The original revisions for EmoBank
and the ConvoKit-managed Stanford Politeness snapshot were not recorded;
`--allow-unpinned` downloads their current upstream states, and reconstruction
still fails rather than silently substituting data if any selected row no
longer matches its committed SHA-256.
