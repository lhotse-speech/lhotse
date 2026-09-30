from pathlib import Path

from lhotse import AudioSource, Recording
from lhotse.recipes.babel import prepare_single_babel_language


def test_prepare_single_babel_language_includes_sph_and_wav(monkeypatch, tmp_path):
    stems_and_suffixes = (
        ("BABEL_BP_401_10001_20111024_205740_inLine", ".sph"),
        ("BABEL_BP_401_10002_20111024_205740_inLine", ".wav"),
    )
    for split in ("dev", "training"):
        audio_dir = tmp_path / "conversational" / split / "audio"
        text_dir = tmp_path / "conversational" / split / "transcription"
        audio_dir.mkdir(parents=True)
        text_dir.mkdir(parents=True)

        for stem, suffix in stems_and_suffixes:
            audio_path = audio_dir / f"{stem}{suffix}"
            audio_path.touch()
            (text_dir / f"{stem}.txt").write_text("[0.0]\nhello\n[1.0]\n")

    def fake_recording_from_file(path):
        path = Path(path)
        return Recording(
            id=path.stem,
            sources=[AudioSource(type="file", channels=[0], source=str(path))],
            sampling_rate=8000,
            num_samples=8000,
            duration=1.0,
        )

    monkeypatch.setattr(Recording, "from_file", staticmethod(fake_recording_from_file))

    manifests = prepare_single_babel_language(tmp_path, no_eval_ok=True)

    for split in ("dev", "training"):
        recordings = manifests[split]["recordings"]
        assert not recordings.is_lazy
        assert set(recordings.ids) == {stem for stem, _ in stems_and_suffixes}
        assert {Path(r.sources[0].source).suffix for r in recordings} == {
            ".sph",
            ".wav",
        }
