"""
Checks that a recipe's output follows the lhotse recipe convention.

Use it from a small script that fakes a tiny corpus and runs the recipe:

    sys.path.insert(0, ".claude/skills/review-recipe")
    from check_recipe import check_parts, check_output_dir_names

    for output_dir in (out, out, None):  # write, rerun from cache, in memory
        result = prepare_mycorpus(corpus_dir, output_dir)
        check_parts(result, output_dir, corpus="mycorpus", parts=["train", "test"])
        if output_dir is not None:
            check_output_dir_names(output_dir)
"""
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse.cut import MonoCut, MultiCut

CORPUS_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
FILE_RE = re.compile(r"^(?P<corpus>[a-z0-9]+(?:-[a-z0-9]+)*)_(?P<manifest>[a-z]+)_(?P<part>.+)\.jsonl\.gz$")


def manifest_path(output_dir: Path, corpus: str, manifest: str, part: str) -> Path:
    assert CORPUS_RE.match(corpus), f"Corpus name must be lowercase words joined by '-': {corpus!r}"
    return Path(output_dir) / f"{corpus}_{manifest}_{part}.jsonl.gz"


def check_part(
    manifests: dict,
    output_dir: Optional[Path],
    corpus: str,
    part: str,
    has_supervisions: bool = True,
    extra_keys: Iterable[str] = (),
    load_audio: bool = True,
) -> None:
    """Check one ``{"recordings", "supervisions", "cuts", ...}`` dict against the convention."""
    expected = {"recordings", "cuts", *extra_keys} | ({"supervisions"} if has_supervisions else set())
    assert set(manifests) == expected, f"{corpus}/{part}: keys {set(manifests)} != {expected}"
    recordings, cuts = manifests["recordings"], manifests["cuts"]
    assert isinstance(recordings, RecordingSet) and isinstance(cuts, CutSet)
    recordings, cuts = list(recordings), list(cuts)
    assert recordings, f"{corpus}/{part}: no recordings"
    assert [c.id for c in cuts] == [r.id for r in recordings], f"{corpus}/{part}: cut IDs != recording IDs"
    for cut, rec in zip(cuts, recordings):
        assert cut.start == 0 and cut.duration == rec.duration, (corpus, part, cut.id)
        assert cut.recording.id == rec.id
        assert isinstance(cut, MonoCut if rec.num_channels == 1 else MultiCut), (corpus, part, cut.id)
    if has_supervisions:
        supervisions = manifests["supervisions"]
        assert isinstance(supervisions, SupervisionSet)
        by_recording = defaultdict(list)
        for s in supervisions:
            by_recording[s.recording_id].append(s.id)
        for cut in cuts:
            assert [s.id for s in cut.supervisions] == by_recording[cut.id], (corpus, part, cut.id)
        assert sum(len(c.supervisions) for c in cuts) == sum(len(v) for v in by_recording.values())
    else:
        assert all(not c.supervisions for c in cuts)
    if output_dir is not None:
        for manifest in expected:
            path = manifest_path(output_dir, corpus, manifest, part)
            assert path.is_file(), f"missing {path}"
        assert manifests["cuts"].is_lazy, f"{corpus}/{part}: cuts written to disk should be returned lazily"
    if load_audio:
        audio = cuts[0].load_audio()
        assert audio.shape == (cuts[0].num_channels, cuts[0].num_samples), (corpus, part, audio.shape)


def check_parts(result: dict, output_dir: Optional[Path], corpus: str, parts: Iterable[str], **kwargs) -> None:
    """Check a ``{part: {"recordings", "supervisions", "cuts"}}`` recipe result."""
    parts = list(parts)
    assert isinstance(result, dict) and set(result) == set(parts), f"{corpus}: parts {set(result)} != {set(parts)}"
    for part in parts:
        check_part(result[part], output_dir, corpus, part, **kwargs)


def check_output_dir_names(output_dir: Path, allowed_dirs: Iterable[str] = ()) -> None:
    """Every manifest file in ``output_dir`` must be named ``<corpus>_<manifest>_<part>.jsonl.gz``."""
    allowed_dirs = set(allowed_dirs)
    for path in Path(output_dir).iterdir():
        if path.is_dir():
            assert path.name in allowed_dirs, f"unexpected directory in output_dir: {path}"
            continue
        assert FILE_RE.match(path.name), f"file name does not follow the convention: {path.name}"
