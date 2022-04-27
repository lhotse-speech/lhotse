import pytest
import os.path
import contextlib
from pathlib import Path

import lhotse
import lhotse.utils
from lhotse.utils import is_module_available

FIXTURE_PATH = Path(os.path.realpath(__file__)).parent / "fixtures"

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


@pytest.mark.xfail(reason="multi fail recordings not supported yet")
def test_multi_file_recording(tmp_path, multi_file_recording):
    with working_directory(tmp_path):
        lhotse.kaldi.export_to_kaldi(
            multi_file_recording[0], multi_file_recording[1], ".", None, False
        )


def test_multi_channel_recording(tmp_path, multi_channel_recording):
    with working_directory(tmp_path):
        lhotse.kaldi.export_to_kaldi(
            multi_channel_recording[0], multi_channel_recording[1], ".", None, False
        )


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
        out = lhotse.kaldi.load_kaldi_data_dir(fixture_path, 16000, 0.01, replace, 1)

    lhotse_dir = "lhotse"
    if replace:
        lhotse_dir += "-" + replace

    with working_directory(fixture_path / lhotse_dir):
        recording_set = lhotse.RecordingSet.from_jsonl("recordings.jsonl.gz")
        supervision_set = lhotse.SupervisionSet.from_jsonl("supervisions.jsonl.gz")

    assert out[0] == recording_set
    assert out[1] == supervision_set


@pytest.mark.xfail(reason="Import ordering is somehow not working for now")
def test_get_duration_failurex(monkeypatch):
    def always_fail_is_module_available(path):
        return False

    monkeypatch.setattr(
        lhotse.utils.is_module_available, always_fail_is_module_available
    )

    with pytest.raises(ValueError):
        lhotse.kaldi.get_duration("true |")


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


@pytest.mark.parametrize("channel", [0, 3])
def test_ok_on_file_singlechannel_mp3_source_type(tmp_path, channel):
    source = lhotse.AudioSource(
        type="file", channels=[channel], source="nonexistent.mp3"
    )
    out = lhotse.kaldi.make_wavscp_channel_string_map(source, 16000)
    assert list(out.keys()) == [channel]
    assert out[channel].startswith("ffmpeg")
    assert "nonexistent.mp3" in out[channel]


def test_ok_on_file_multichannel_wav_source_type(tmp_path):
    source = lhotse.AudioSource(
        type="file", channels=[0, 1, 2], source="nonexistent.wav"
    )
    out = lhotse.kaldi.make_wavscp_channel_string_map(source, 16000)
    assert list(out.keys()) == [0, 1, 2]
    for channel in out.keys():
        assert out[channel].startswith("ffmpeg")
        assert "nonexistent.wav" in out[channel]


def test_load_kaldi_text_mapping(tmp_path):
    with pytest.raises(ValueError):
        lhotse.kaldi.load_kaldi_text_mapping(tmp_path / "nonexistent", must_exist=True)


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
        assert os.path.exists("wav.scp")
        assert os.path.exists("segments")
        assert os.path.exists("text")
        assert os.path.exists("utt2spk")
        assert os.path.exists("utt2dur")
        assert os.path.exists("utt2gender")
        assert os.path.exists("reco2dur")

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

    for elem in segments_orig.keys():
        elem_other = utt2spk_orig[elem] + "-" + elem if prefix else elem
        seg_orig = segments_orig[elem].split()
        seg = segments[elem_other].split()
        assert seg_orig[0] == seg[0]
        assert float(seg_orig[1]) == pytest.approx(float(seg[1]))
        assert float(seg_orig[2]) == pytest.approx(float(seg[2]))

    return
