import pytest
import os.path
import contextlib
from pathlib import Path

import lhotse

FIXTURE_PATH = Path(os.path.realpath(__file__)).parent / "fixtures"

MINILIB_PATH = FIXTURE_PATH / "mini_librispeech"


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

    if replace:
        lhotse_dir = "lhotse-" + replace
    else:
        lhotse_dir = "lhotse"

    with working_directory(fixture_path / lhotse_dir):
        recording_set = lhotse.RecordingSet.from_jsonl("recordings.jsonl.gz")
        supervision_set = lhotse.SupervisionSet.from_jsonl("supervisions.jsonl.gz")

    assert out[0] == recording_set
    assert out[1] == supervision_set


def open_and_load(path):
    return lhotse.kaldi.load_kaldi_text_mapping(Path(path))


@pytest.mark.parametrize("replace", [None, "lbi"])
@pytest.mark.parametrize("prefix", [False, True])
def test_kaldi_export(tmp_path, replace, prefix):
    fixture_path = MINILIB_PATH
    output_path = tmp_path
    with working_directory(fixture_path / "lhotse"):
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
