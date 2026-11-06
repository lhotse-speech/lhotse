---
name: review-recipe
description: Review a lhotse data preparation recipe (lhotse/recipes/<corpus>.py) against the recipe conventions — return value, file names, cuts, output_dir, memory, caching, CLI — and run it on a tiny fake corpus. Use when adding or changing a recipe, or when asked to review one.
---

# Review a lhotse recipe

Argument: the recipe name or file (e.g. `mycorpus` or `lhotse/recipes/mycorpus.py`). If none is given, review the recipes changed on the current branch (`git diff --name-only origin/master -- lhotse/recipes`).

The user-facing description of these rules is in `docs/corpus.rst` ("What you get back", "Files on disk", "Adding new corpora"). Good references: `lhotse/recipes/librispeech.py` (in-memory) and `lhotse/recipes/gigaspeech2.py` (streaming).

## 1. Read the code and check each rule

**API**
- [ ] `prepare_<corpus>(corpus_dir, output_dir: Optional[Pathlike] = None, ...)`; `download_<corpus>(target_dir, ...)` if the corpus is freely downloadable. Both exported in `lhotse/recipes/__init__.py`, the corpus listed in `docs/corpus.rst`.
- [ ] Returns `{part: {"recordings": RecordingSet, "supervisions": SupervisionSet, "cuts": CutSet}}`. Parts are the corpus's own split names, or `"all"`. Only extra level allowed: `{language: {part: ...}}` when one call prepares several languages.
- [ ] Exactly those inner keys. Recordings-only corpora omit `"supervisions"`; mixture/trial corpora may return only `"cuts"`; extra manifests use a noun key (e.g. `"uem"`). Never tuples, `None`, a bare CutSet, or keys like `"audio"`/`"cutset"`.
- [ ] Variants chosen by an argument (mic, language, segmentation, version) are not dict levels; one call returns one variant.
- [ ] Return annotation and `:return:` docstring match the actual structure.

**Files**
- [ ] Each manifest is written as `output_dir / f"{corpus}_{manifest}_{part}.jsonl.gz"`, directly in `output_dir`, matching `result[part][manifest]`.
- [ ] `{corpus}` is lowercase, words joined by `-`, never `_`; variants appended with `-` (`ami-ihm`, `mls-polish`). Always `.jsonl.gz`.
- [ ] Nothing is written outside `output_dir` (not into `corpus_dir`, not the cwd). Intermediate files follow the same pattern (`<corpus>-<stage>_<manifest>_<part>.jsonl.gz`).

**Behaviour**
- [ ] Cuts come from `lhotse.recipes.utils.create_cuts_per_recording(...)`, or `recording.to_cut()` + that recording's supervisions inside a streaming loop: one cut per recording, cut ID == recording ID, never trimmed to supervisions. Built after `fix_manifests`/validation.
- [ ] `output_dir=None` writes nothing and returns eager manifests. Requiring `output_dir` is only OK when the recipe writes new audio, with a clear `ValueError`.
- [ ] Large corpora (≳100k recordings; judge from the docstring, download function or your knowledge of the corpus, and say so if unsure) stream recordings, supervisions and cuts to disk in one pass with `<Manifest>.open_writer(...)` and return lazy manifests; nothing lazy is materialized just to make cuts.
- [ ] If parts already on disk are skipped (`manifests_exist`), they are loaded and returned too (`read_manifests_if_cached`). A rerun never returns fewer parts.
- [ ] IDs are deterministic and path-independent: no `hash()`, no absolute paths, and no ID or channel index derived from unsorted `glob`/`rglob` order (plain iteration order is fine; manifests are sorted later). Supervisions reference existing recording IDs and the right `channel`.
- [ ] Optional dependencies are imported inside functions with an install hint; module import has no side effects.

**CLI**
- [ ] `lhotse/bin/modes/recipes/<corpus>.py` wraps `prepare` (and `download`), exposes the recipe's options, and is imported in `lhotse/bin/modes/recipes/__init__.py`. Run `lhotse prepare <corpus> --help` (and `lhotse download <corpus> --help`) to confirm.

## 2. Run it on a fake corpus

Write a throwaway script outside the repo that builds a tiny corpus mirroring the real layout, inferred from the recipe code, its docstrings and the download function (2–3 short but real audio files per part written with `soundfile`, the real transcript format, multi-channel/variants if supported), then runs the recipe three times — with `output_dir`, again with the same `output_dir` (cache path), and with `output_dir=None` — checking each result with `.claude/skills/review-recipe/check_recipe.py`:

```python
sys.path.insert(0, ".claude/skills/review-recipe")
from check_recipe import check_parts, check_output_dir_names
check_parts(result, output_dir, corpus="ami-ihm", parts=["train", "dev", "test"])  # per language if nested
check_output_dir_names(output_dir)
```

Run it from the repo root. `check_parts` asserts the structure, file names, cut/recording/supervision consistency, lazy cuts on disk, and that audio loads. Use `has_supervisions=False` for recordings-only corpora and `extra_keys=[...]` for extra manifests. Stub network fetches in the script, never in the recipe.

## 3. Report

List rule violations first, most severe first: `file:line`, the rule broken, the consequence, the fix. Then other bugs you noticed (e.g. crashes on missing files), in the same format. Then say what you ran and what you could not check (e.g. layout details you had to guess). If everything passes, say so in one line.
