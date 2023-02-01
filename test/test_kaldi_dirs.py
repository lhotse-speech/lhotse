import contextlib
import os
from pathlib import Path

import pytest

import lhotse

pytest.importorskip(
    "kaldi_native_io", reason="Kaldi tests require kaldi_native_io to be installed."
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures"

MINILIB_PATH = FIXTURE_PATH / "mini_librispeech"


@pytest.fixture
def multi_file_recording():
    recording = lhotse.RecordingSet.from_recordings(
        [
            lhotse.Recording(
                id="lbi-1272-135031-0000",
                sources=[
                    lhotse.AudioSource(
                        type="file", channels=[0], source="nonexistent-c1.wav"
                    ),
                    lhotse.AudioSource(
                        type="file", channels=[1], source="nonexistent-c2.wav"
                    ),
                ],
                sampling_rate=16000,
                num_samples=174160,
                duration=174160.0 / 16000.0,
                transforms=None,
            )
        ]
    )

    supervision = lhotse.SupervisionSet.from_segments(
        [
            lhotse.SupervisionSegment(
                id="lbi-1272-135031-0000-A",
                recording_id="lbi-1272-135031-0000",
                start=0.0,
                duration=10.885,
                channel=0,
                text="SOMETHING",
                speaker="lbi-1272-135031-A",
                gender="m",
                language="en-US",
            ),
            lhotse.SupervisionSegment(
                id="lbi-1272-135031-0000-B",
                recording_id="lbi-1272-135031-0000",
                start=0.0,
                duration=10.885,
                channel=1,
                text="SOMETHING ELSE",
                speaker="lbi-1272-135031-B",
                gender="f",
                language="en-US",
            ),
        ]
    )
    return recording, supervision


@pytest.fixture
def multi_channel_recording():
    recording = lhotse.RecordingSet.from_recordings(
        [
            lhotse.Recording(
                id="lbi-1272-135031-0000",
                sources=[
                    lhotse.AudioSource(
                        type="file", channels=[0, 1], source="nonexistent.wav"
                    ),
                ],
                sampling_rate=16000,
                num_samples=174160,
                duration=174160.0 / 16000.0,
                transforms=None,
            )
        ]
    )

    supervision = lhotse.SupervisionSet.from_segments(
        [
            lhotse.SupervisionSegment(
                id="lbi-1272-135031-0000-A",
                recording_id="lbi-1272-135031-0000",
                start=0.0,
                duration=10.885,
                channel=0,
                text="SOMETHING",
                speaker="lbi-1272-135031-A",
                gender="m",
            ),
            lhotse.SupervisionSegment(
                id="lbi-1272-135031-0000-B",
                recording_id="lbi-1272-135031-0000",
                start=0.0,
                duration=10.885,
                channel=1,
                text="SOMETHING ELSE",
                speaker="lbi-1272-135031-B",
                gender="f",
            ),
        ]
    )
    return recording, supervision


@pytest.fixture
def multi_channel_kaldi_dir():
    return {
        "wav.scp": {
            "lbi-1272-135031-0000_0": "ffmpeg -threads 1 -i nonexistent.wav -ar 16000 -map_channel 0.0.0  -f wav -threads 1 pipe:1 |",
            "lbi-1272-135031-0000_1": "ffmpeg -threads 1 -i nonexistent.wav -ar 16000 -map_channel 0.0.1  -f wav -threads 1 pipe:1 |",
        },
        "segments": {
            "lbi-1272-135031-0000-A-0": "lbi-1272-135031-0000_0 0.0 10.885",
            "lbi-1272-135031-0000-B-1": "lbi-1272-135031-0000_1 0.0 10.885",
        },
    }


def test_multi_file_recording(tmp_path, multi_file_recording):
    with working_directory(tmp_path):
        lhotse.kaldi.export_to_kaldi(
            multi_file_recording[0],
            multi_file_recording[1],
            output_dir=".",
            map_underscores_to=None,
            prefix_spk_id=False,
        )


def test_multi_channel_recording(
    tmp_path, multi_channel_recording, multi_channel_kaldi_dir
):
    with working_directory(tmp_path):
        lhotse.kaldi.export_to_kaldi(
            multi_channel_recording[0],
            multi_channel_recording[1],
            output_dir=".",
            map_underscores_to=None,
            prefix_spk_id=False,
        )

        wavs = open_and_load("wav.scp")
        segments = open_and_load("segments")
        assert wavs == multi_channel_kaldi_dir["wav.scp"]
        assert segments == multi_channel_kaldi_dir["segments"]


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@pytest.mark.parametrize("replace", [None, "b"])
def test_kaldi_import(replace):
    fixture_path = MINILIB_PATH
    with working_directory(fixture_path):
        out = lhotse.kaldi.load_kaldi_data_dir(
            fixture_path,
            sampling_rate=16000,
            frame_shift=0.01,
            map_string_to_underscores=replace,
            num_jobs=1,
            use_reco2dur=False,
        )

    lhotse_dir = "lhotse"
    if replace:
        lhotse_dir += "-" + replace

    with working_directory(fixture_path / lhotse_dir):
        recording_set = lhotse.RecordingSet.from_jsonl("recordings.jsonl.gz")
        supervision_set = lhotse.SupervisionSet.from_jsonl("supervisions.jsonl.gz")

    assert list(out[0]) == list(recording_set)
    assert out[1] == supervision_set


def open_and_load(path):
    return lhotse.kaldi.load_kaldi_text_mapping(Path(path))


def test_fail_on_unknown_source_type(tmp_path):
    source = lhotse.AudioSource(
        type="unknown", channels=[0], source="http://example.com/"
    )
    with pytest.raises(ValueError):
        lhotse.kaldi.make_wavscp_channel_string_map(source, 16000)


def test_fail_on_url_source_type(tmp_path):
    source = lhotse.AudioSource(type="url", channels=[0], source="http://example.com/")
    with pytest.raises(ValueError):
        lhotse.kaldi.make_wavscp_channel_string_map(source, 16000)


def test_fail_on_command_multichannel_source_type(tmp_path):
    source = lhotse.AudioSource(type="command", channels=[0, 1], source="false")
    with pytest.raises(ValueError):
        lhotse.kaldi.make_wavscp_channel_string_map(source, 16000)


def test_ok_on_command_singlechannel_source_type(tmp_path):
    source = lhotse.AudioSource(type="command", channels=[0], source="true")
    out = lhotse.kaldi.make_wavscp_channel_string_map(source, 16000)
    assert list(out.keys()) == [0]
    assert out[0] == "true |"


@pytest.mark.parametrize("channel", [0, 3])
def test_ok_on_file_singlechannel_wav_source_type(tmp_path, channel):
    source = lhotse.AudioSource(
        type="file", channels=[channel], source="nonexistent.wav"
    )
    out = lhotse.kaldi.make_wavscp_channel_string_map(source, 16000)
    assert list(out.keys()) == [channel]
    assert out[channel] == "nonexistent.wav"


@pytest.mark.parametrize("channel", [0, 3])
def test_ok_on_file_singlechannel_sph_source_type(tmp_path, channel):
    source = lhotse.AudioSource(
        type="file", channels=[channel], source="nonexistent.sph"
    )
    out = lhotse.kaldi.make_wavscp_channel_string_map(source, 16000)
    assert list(out.keys()) == [channel]
    assert out[channel].startswith("sph2pipe")
    assert "nonexistent.sph" in out[channel]
    assert "{" not in out[channel]


@pytest.mark.parametrize("channel", [0, 3])
def test_ok_on_file_singlechannel_mp3_source_type(tmp_path, channel):
    source = lhotse.AudioSource(
        type="file", channels=[channel], source="nonexistent.mp3"
    )
    out = lhotse.kaldi.make_wavscp_channel_string_map(source, 16000)
    assert list(out.keys()) == [channel]
    assert out[channel].startswith("ffmpeg")
    assert "nonexistent.mp3" in out[channel]
    assert "{" not in out[channel]


def test_ok_on_file_multichannel_wav_source_type(tmp_path):
    source = lhotse.AudioSource(
        type="file", channels=[0, 1, 2], source="nonexistent.wav"
    )
    out = lhotse.kaldi.make_wavscp_channel_string_map(source, 16000)
    assert list(out.keys()) == [0, 1, 2]
    for channel in out.keys():
        assert out[channel].startswith("ffmpeg")
        assert "nonexistent.wav" in out[channel]
        assert "{" not in out[channel]


def test_load_kaldi_text_mapping(tmp_path):
    with pytest.raises(ValueError):
        lhotse.kaldi.load_kaldi_text_mapping(tmp_path / "nonexistent", must_exist=True)


@pytest.mark.parametrize("load_durations", [False, True])
def test_load_durations(tmp_path, load_durations):
    fixture_path = MINILIB_PATH
    with working_directory(fixture_path):
        out = lhotse.kaldi.load_kaldi_data_dir(
            fixture_path,
            sampling_rate=16000,
            frame_shift=0.01,
            use_reco2dur=load_durations,
            num_jobs=1,
        )

    lhotse_dir = "lhotse"
    with working_directory(fixture_path / lhotse_dir):
        recording_set = lhotse.RecordingSet.from_jsonl("recordings.jsonl.gz")
        supervision_set = lhotse.SupervisionSet.from_jsonl("supervisions.jsonl.gz")

    if load_durations:
        for i, recording in enumerate(out[0]):
            assert recording_set[i].duration == pytest.approx(
                recording.duration,
                lhotse.audio.LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE,
            )
    else:
        assert list(out[0]) == list(recording_set)
    assert out[1] == supervision_set
    pass


@pytest.mark.parametrize("replace", [None, "b"])
@pytest.mark.parametrize("prefix", [False, True])
def test_kaldi_export(tmp_path, replace, prefix):
    fixture_path = MINILIB_PATH
    output_path = tmp_path

    lhotse_dir = "lhotse"
    if replace:
        lhotse_dir += "-" + replace

    with working_directory(fixture_path / lhotse_dir):
        recording_set = lhotse.RecordingSet.from_jsonl("recordings.jsonl.gz")
        supervision_set = lhotse.SupervisionSet.from_jsonl("supervisions.jsonl.gz")

    lhotse.kaldi.export_to_kaldi(
        recording_set, supervision_set, output_path, replace, prefix
    )

    with working_directory(output_path):
        assert Path("wav.scp").is_file()
        assert Path("segments").is_file()
        assert Path("text").is_file()
        assert Path("utt2spk").is_file()
        assert Path("utt2dur").is_file()
        assert Path("utt2gender").is_file()
        assert Path("reco2dur").is_file()

        wavs = open_and_load("wav.scp")
        text = open_and_load("text")
        segments = open_and_load("segments")
        utt2spk = open_and_load("utt2spk")

    with working_directory(fixture_path):
        wavs_orig = open_and_load("wav.scp")
        text_orig = open_and_load("text")
        segments_orig = open_and_load("segments")
        utt2spk_orig = open_and_load("utt2spk")

    assert len(text.keys()) == len(text_orig.keys())
    for nr, elem in enumerate(text_orig.keys()):
        if prefix:
            assert text_orig[elem] == text[utt2spk_orig[elem] + "-" + elem], text.keys()
        else:
            assert text_orig[elem] == text[elem], text.keys()

    assert len(wavs.keys()) == len(wavs_orig.keys())
    for elem in wavs_orig.keys():
        assert wavs_orig[elem].rstrip(" |") == wavs[elem].rstrip(" |")
        assert "{" not in wavs[elem]

    for elem in segments_orig.keys():
        elem_other = utt2spk_orig[elem] + "-" + elem if prefix else elem
        seg_orig = segments_orig[elem].split()
        seg = segments[elem_other].split()
        assert seg_orig[0] == seg[0]
        assert float(seg_orig[1]) == pytest.approx(float(seg[1]))
        assert float(seg_orig[2]) == pytest.approx(float(seg[2]))

    return
