"""
Generates manifests that use the legacy, flat ``Features`` layout.

The files next to this script were produced by running it from the repository root
with lhotse at commit 16cc369e, before ``Features`` became a ``TemporalArray``::

    python test/fixtures/legacy_features/generate.py

Do not regenerate them with a newer lhotse: their purpose is to pin what older
versions wrote, together with the arrays those versions loaded.
"""

import pickle
import shutil
from pathlib import Path

import numpy as np

from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, MonoCut, Recording
from lhotse.cut import MultiCut
from lhotse.dataset.webdataset import export_to_webdataset

ROOT = Path("test/fixtures/legacy_features")


def main():
    (ROOT / "feats.lca").unlink(missing_ok=True)
    shutil.rmtree(ROOT / "shar", ignore_errors=True)
    (ROOT / "shar").mkdir()
    extractor = Fbank(FbankConfig(num_mel_bins=8, sampling_rate=8000))
    mono = Recording.from_file("test/fixtures/mono_c0.wav", recording_id="mono")
    stereo = Recording.from_file("test/fixtures/stereo.wav", recording_id="stereo")

    with LilcomChunkyWriter(ROOT / "feats") as writer:
        # A long cut whose features are computed once and shared by its sub-cuts.
        full = MonoCut(
            "full", start=0, duration=mono.duration, channel=0, recording=mono
        ).compute_and_store_features(extractor, writer)
        sub = full.truncate(offset=0.1, duration=0.3, preserve_id=False).with_id("sub")
        feats_only = sub.drop_recording().with_id("feats-only")
        multi = MultiCut(
            "multi", start=0, duration=0.5, channel=[0, 1], recording=stereo
        ).compute_and_store_features(extractor, writer)
        other = MonoCut(
            "other", start=0.2, duration=0.3, channel=0, recording=mono
        ).compute_and_store_features(extractor, writer)
    mixed = full.truncate(duration=0.5).mix(other, offset_other_by=0.3).with_id("mixed")
    padded = sub.pad(duration=0.6, direction="both").with_id("padded")
    cuts = CutSet([full, sub, feats_only, multi, mixed, padded])
    cuts.to_file(ROOT / "cuts.jsonl")

    expected = {}
    for cut in cuts:
        expected[cut.id] = cut.load_features()
        expected[f"{cut.id}-trunc"] = cut.truncate(
            offset=0.05, duration=0.2
        ).load_features()
        expected[f"{cut.id}-memory"] = cut.move_to_memory().load_features()
    np.savez(ROOT / "expected.npz", **expected)

    memory_cuts = CutSet([c.move_to_memory() for c in [full, sub, feats_only]])
    with open(ROOT / "memory_cuts.pkl", "wb") as f:
        pickle.dump(list(memory_cuts), f)
    export_to_webdataset(memory_cuts, str(ROOT / "memory_cuts.tar"))
    CutSet([full, sub, multi]).to_shar(
        ROOT / "shar", fields={"features": "lilcom"}, shard_size=10
    )


if __name__ == "__main__":
    main()
