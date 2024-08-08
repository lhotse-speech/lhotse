import logging
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest
import torch
import torchaudio

from lhotse import (
    LilcomFilesWriter,
    MonoCut,
    NumpyFilesWriter,
    Recording,
    compute_num_samples,
    validate,
)
from lhotse.serialization import deserialize_item
from lhotse.testing.dummies import (
    dummy_cut,
    dummy_multi_channel_recording,
    dummy_multi_cut,
    dummy_recording,
    dummy_supervision,
)
from lhotse.testing.random import deterministic_rng


@pytest.mark.parametrize("cut", [dummy_cut(1), dummy_cut(2).pad(300)])
def test_cut_nonexistent_attribute(cut):
    with pytest.raises(AttributeError):
        cut.nonexistent_attribute


def test_cut_load_array():
    """Check that a custom Array attribute is successfully recognized."""
    ivector = np.arange(20).astype(np.float32)
    with TemporaryDirectory() as d, LilcomFilesWriter(d) as writer:
        manifest = writer.store_array(key="utt1", value=ivector)
        cut = MonoCut(id="x", start=0, duration=5, channel=0)
        # Note: MonoCut doesn't normally have an "ivector" attribute,
        #       and a "load_ivector()" method.
        #       We are dynamically extending it.
        cut.ivector = manifest
        restored_ivector = cut.load_ivector()
        np.testing.assert_equal(ivector, restored_ivector)


def test_cut_load_array_truncate():
    """Check that loading a custom Array works after truncation."""
    ivector = np.arange(20).astype(np.float32)
    with TemporaryDirectory() as d, LilcomFilesWriter(d) as writer:
        cut = dummy_cut(0, duration=5.0)
        cut.ivector = writer.store_array(key="utt1", value=ivector)

        cut = cut.truncate(duration=3)

        restored_ivector = cut.load_ivector()
        np.testing.assert_equal(ivector, restored_ivector)


def test_cut_load_array_pad():
    """Check that loading a custom Array works after padding."""
    ivector = np.arange(20).astype(np.float32)
    with TemporaryDirectory() as d, LilcomFilesWriter(d) as writer:
        cut = MonoCut(
            id="x",
            start=0,
            duration=5,
            channel=0,
            recording=dummy_recording(1, duration=5.0),
        )
        cut.ivector = writer.store_array(key="utt1", value=ivector)

        cut = cut.pad(duration=7.6)

        restored_ivector = cut.load_ivector()
        np.testing.assert_equal(ivector, restored_ivector)


def test_cut_custom_attr_serialization():
    """Check that a custom Array attribute is successfully serialized + deserialized."""
    ivector = np.arange(20).astype(np.float32)
    with TemporaryDirectory() as d, LilcomFilesWriter(d) as writer:
        cut = MonoCut(id="x", start=0, duration=5, channel=0)
        cut.ivector = writer.store_array(key="utt1", value=ivector)

        data = cut.to_dict()
        restored_cut = deserialize_item(data)
        assert cut == restored_cut

        restored_ivector = restored_cut.load_ivector()
        np.testing.assert_equal(ivector, restored_ivector)


def test_cut_custom_nonarray_attr_serialization():
    """Check that arbitrary custom fields work with Cuts upon (de)serialization."""
    cut = MonoCut(id="x", start=10, duration=8, channel=0, custom={"SNR": 7.3})

    data = cut.to_dict()
    restored_cut = deserialize_item(data)
    assert cut == restored_cut

    # Note: we extended cuts attributes by setting the "custom" field.
    assert restored_cut.SNR == 7.3


def test_cut_load_temporal_array(deterministic_rng):
    """Check that we can read a TemporalArray from a cut when their durations match."""
    alignment = np.random.randint(500, size=131)
    with TemporaryDirectory() as d, NumpyFilesWriter(d) as writer:
        manifest = writer.store_array(
            key="utt1", value=alignment, frame_shift=0.4, temporal_dim=0
        )
        expected_duration = 52.4  # 131 frames x 0.4s frame shift == 52.4s
        cut = MonoCut(id="x", start=0, duration=expected_duration, channel=0)
        # Note: MonoCut doesn't normally have an "alignment" attribute,
        #       and a "load_alignment()" method.
        #       We are dynamically extending it.
        cut.alignment = manifest
        restored_alignment = cut.load_alignment()
        np.testing.assert_equal(alignment, restored_alignment)


def test_cut_load_temporal_array_truncate(deterministic_rng):
    """Check the array loaded via TemporalArray is truncated along with the cut."""
    with TemporaryDirectory() as d, NumpyFilesWriter(d) as writer:
        expected_duration = 52.4  # 131 frames x 0.4s frame shift == 52.4s
        cut = dummy_cut(0, duration=expected_duration)

        alignment = np.random.randint(500, size=131)
        cut.alignment = writer.store_array(
            key="utt1", value=alignment, frame_shift=0.4, temporal_dim=0
        )
        cut_trunc = cut.truncate(duration=5.0)

        alignment_piece = cut_trunc.load_alignment()
        assert alignment_piece.shape == (13,)  # 5.0 / 0.4 == 12.5 ~= 13
        np.testing.assert_equal(alignment[:13], alignment_piece)


@pytest.mark.parametrize("pad_value", [-1, 0])
def test_cut_load_temporal_array_pad(deterministic_rng, pad_value):
    """Check the array loaded via TemporalArray is padded along with the cut."""
    with TemporaryDirectory() as d, NumpyFilesWriter(d) as writer:
        cut = MonoCut(
            id="x",
            start=0,
            duration=52.4,  # 131 frames x 0.4s frame shift == 52.4s
            channel=0,
            recording=dummy_recording(1),
        )

        alignment = np.random.randint(500, size=131)
        cut.alignment = writer.store_array(
            key="utt1", value=alignment, frame_shift=0.4, temporal_dim=0
        )
        cut_pad = cut.pad(duration=60.0, pad_value_dict={"alignment": pad_value})

        alignment_pad = cut_pad.load_alignment()
        assert alignment_pad.shape == (150,)  # 60.0 / 0.4 == 150
        np.testing.assert_equal(alignment_pad[:131], alignment)
        np.testing.assert_equal(alignment_pad[131:], pad_value)


def test_validate_cut_with_temporal_array(caplog, deterministic_rng):
    # Note: "caplog" is a special variable in pytest that captures logs.
    caplog.set_level(logging.WARNING)
    with TemporaryDirectory() as d, NumpyFilesWriter(d) as writer:
        cut = MonoCut(
            id="cut1",
            start=0,
            duration=4.9,
            channel=0,
            recording=dummy_recording(1),
        )
        alignment = np.random.randint(500, size=131)
        cut.alignment = writer.store_array(
            key="utt1", value=alignment, frame_shift=0.4, temporal_dim=0
        )
        validate(cut)

    assert (
        "MonoCut cut1: possibly mismatched duration between cut (4.9s) "
        "and temporal array in custom field 'alignment' (num_frames=131 "
        "* frame_shift=0.4 == duration=52.400000000000006)" in caplog.text
    )


def test_cut_load_custom_recording(deterministic_rng):
    sampling_rate = 16000
    duration = 52.4
    audio = np.random.randn(1, compute_num_samples(duration, sampling_rate)).astype(
        np.float32
    )
    audio /= np.abs(audio).max()  # normalize to [-1, 1]
    with NamedTemporaryFile(suffix=".wav") as f:
        torchaudio.save(f.name, torch.from_numpy(audio), sampling_rate)
        f.flush()
        os.fsync(f)
        recording = Recording.from_file(f.name)

        # Note: MonoCut doesn't normally have an "alignment" attribute,
        #       and a "load_alignment()" method.
        #       We are dynamically extending it.
        cut = MonoCut(id="x", start=0, duration=duration, channel=0)
        cut.my_favorite_song = recording

        restored_audio = cut.load_my_favorite_song()
        np.testing.assert_allclose(audio, restored_audio, atol=4e-5)


def test_cut_load_custom_recording_truncate(deterministic_rng):
    sampling_rate = 16000
    duration = 52.4
    audio = np.random.randn(1, compute_num_samples(duration, sampling_rate)).astype(
        np.float32
    )
    audio /= np.abs(audio).max()  # normalize to [-1, 1]
    with NamedTemporaryFile(suffix=".wav") as f:
        torchaudio.save(f.name, torch.from_numpy(audio), sampling_rate)
        f.flush()
        os.fsync(f)
        recording = Recording.from_file(f.name)

        # Note: MonoCut doesn't normally have an "alignment" attribute,
        #       and a "load_alignment()" method.
        #       We are dynamically extending it.
        cut = dummy_cut(0, duration=duration)
        cut.my_favorite_song = recording

        cut_trunc = cut.truncate(duration=5.0)

        restored_audio = cut_trunc.load_my_favorite_song()
        assert restored_audio.shape == (1, 80000)

        np.testing.assert_allclose(audio[:, :80000], restored_audio, atol=3e-5)


def test_cut_load_custom_recording_pad_right(deterministic_rng):
    sampling_rate = 16000
    duration = 52.4
    audio = np.random.randn(1, compute_num_samples(duration, sampling_rate)).astype(
        np.float32
    )
    audio /= np.abs(audio).max()  # normalize to [-1, 1]
    with NamedTemporaryFile(suffix=".wav") as f:
        torchaudio.save(f.name, torch.from_numpy(audio), sampling_rate)
        f.flush()
        os.fsync(f)
        recording = Recording.from_file(f.name)

        # Note: MonoCut doesn't normally have an "alignment" attribute,
        #       and a "load_alignment()" method.
        #       We are dynamically extending it.
        cut = MonoCut(
            id="x",
            start=0,
            duration=duration,
            channel=0,
            recording=dummy_recording(0, duration=duration),
        )
        cut.my_favorite_song = recording

        cut_pad = cut.pad(duration=60.0)

        restored_audio = cut_pad.load_my_favorite_song()
        assert restored_audio.shape == (1, 960000)  # 16000 * 60

        np.testing.assert_allclose(
            audio, restored_audio[:, : audio.shape[1]], atol=4e-5
        )
        np.testing.assert_allclose(0, restored_audio[:, audio.shape[1] :], atol=4e-5)


def test_cut_load_custom_recording_pad_left(deterministic_rng):
    sampling_rate = 16000
    duration = 52.4
    audio = np.random.randn(1, compute_num_samples(duration, sampling_rate)).astype(
        np.float32
    )
    audio /= np.abs(audio).max()  # normalize to [-1, 1]
    with NamedTemporaryFile(suffix=".wav") as f:
        torchaudio.save(f.name, torch.from_numpy(audio), sampling_rate)
        f.flush()
        os.fsync(f)
        recording = Recording.from_file(f.name)

        # Note: MonoCut doesn't normally have an "alignment" attribute,
        #       and a "load_alignment()" method.
        #       We are dynamically extending it.
        cut = MonoCut(
            id="x",
            start=0,
            duration=duration,
            channel=0,
            recording=dummy_recording(0, duration=duration),
        )
        cut.my_favorite_song = recording

        cut_pad = cut.pad(duration=60.0, direction="left")

        restored_audio = cut_pad.load_my_favorite_song()
        assert restored_audio.shape == (1, 960000)  # 16000 * 60

        np.testing.assert_allclose(0, restored_audio[:, : -audio.shape[1]], atol=4e-5)
        np.testing.assert_allclose(
            audio, restored_audio[:, -audio.shape[1] :], atol=4e-5
        )


def test_cut_load_custom_recording_pad_both(deterministic_rng):
    sampling_rate = 16000
    duration = 52.4
    audio = np.random.randn(1, compute_num_samples(duration, sampling_rate)).astype(
        np.float32
    )
    audio /= np.abs(audio).max()  # normalize to [-1, 1]
    with NamedTemporaryFile(suffix=".wav") as f:
        torchaudio.save(f.name, torch.from_numpy(audio), sampling_rate)
        f.flush()
        os.fsync(f)
        recording = Recording.from_file(f.name)

        # Note: MonoCut doesn't normally have an "alignment" attribute,
        #       and a "load_alignment()" method.
        #       We are dynamically extending it.
        cut = MonoCut(
            id="x",
            start=0,
            duration=duration,
            channel=0,
            recording=dummy_recording(0, duration=duration),
        )
        cut.my_favorite_song = recording

        cut_pad = cut.pad(duration=duration + 1, direction="left").pad(
            duration=60.0, direction="right"
        )

        restored_audio = cut_pad.load_my_favorite_song()
        assert restored_audio.shape == (1, 960000)  # 16000 * 60

        np.testing.assert_allclose(0, restored_audio[:, :sampling_rate], atol=4e-5)
        np.testing.assert_allclose(
            audio,
            restored_audio[:, sampling_rate : audio.shape[1] + sampling_rate],
            atol=4e-5,
        )
        np.testing.assert_allclose(
            0, restored_audio[:, sampling_rate + audio.shape[1] :], atol=4e-5
        )


def test_cut_attach_tensor(deterministic_rng):
    alignment = np.random.randint(500, size=131)
    expected_duration = 52.4  # 131 frames x 0.4s frame shift == 52.4s
    cut = MonoCut(id="x", start=0, duration=expected_duration, channel=0)
    # Note: MonoCut doesn't normally have an "alignment" attribute,
    #       and a "load_alignment()" method.
    #       We are dynamically extending it.
    cut = cut.attach_tensor("alignment", alignment, frame_shift=0.4, temporal_dim=0)
    restored_alignment = cut.load_alignment()
    np.testing.assert_equal(alignment, restored_alignment)


def test_cut_attach_tensor_temporal():
    ivector = np.arange(20).astype(np.float32)
    cut = MonoCut(id="x", start=0, duration=5, channel=0)
    # Note: MonoCut doesn't normally have an "ivector" attribute,
    #       and a "load_ivector()" method.
    #       We are dynamically extending it.
    cut = cut.attach_tensor("ivector", ivector)
    restored_ivector = cut.load_ivector()
    np.testing.assert_equal(ivector, restored_ivector)


def test_del_attr_supervision():
    sup = dummy_supervision(0)

    with pytest.raises(AttributeError):
        del sup.nonexistent_attribute

    sup.extra_metadata = {"version": "0.1.1"}
    assert "extra_metadata" in sup.custom
    sup.extra_metadata  # does not raise
    del sup.extra_metadata
    with pytest.raises(AttributeError):
        del sup.extra_metadata
    assert "extra_metadata" not in sup.custom


@pytest.mark.parametrize("cut", [dummy_cut(0), dummy_multi_cut(0)])
def test_del_attr_mono_cut(cut):
    with pytest.raises(AttributeError):
        del cut.nonexistent_attribute

    cut.extra_metadata = {"version": "0.1.1"}
    assert "extra_metadata" in cut.custom
    cut.extra_metadata  # does not raise
    del cut.extra_metadata
    with pytest.raises(AttributeError):
        del cut.extra_metadata
    assert "extra_metadata" not in cut.custom


def test_multi_cut_custom_multi_recording_channel_selector():
    cut = dummy_multi_cut(0, channel=[0, 1, 2, 3], with_data=True)
    cut.target_recording = dummy_multi_channel_recording(
        1, channel_ids=[0, 1, 2, 3], with_data=True
    )

    # All input channels
    ref_audio = cut.load_audio()
    assert ref_audio.shape == (4, 16000)

    # Input channel selection
    two_channel_in = cut.with_channels([0, 1])
    audio = two_channel_in.load_audio()
    assert audio.shape == (2, 16000)
    np.testing.assert_allclose(ref_audio[:2], audio)

    # Input channel selection, different channels
    two_channel_in = cut.with_channels([0, 3])
    audio = two_channel_in.load_audio()
    assert audio.shape == (2, 16000)
    np.testing.assert_allclose(ref_audio[::3], audio)

    # All output channels
    ref_tgt_audio = cut.load_target_recording()
    assert ref_tgt_audio.shape == (4, 16000)

    # Output channel selection
    two_channel_out = cut.with_custom("target_recording_channel_selector", [0, 1])
    audio = two_channel_out.load_target_recording()
    assert audio.shape == (2, 16000)
    np.testing.assert_allclose(ref_tgt_audio[:2], audio)

    # Output channel selection, different channels
    two_channel_out = cut.with_custom("target_recording_channel_selector", [0, 3])
    audio = two_channel_out.load_target_recording()
    assert audio.shape == (2, 16000)
    np.testing.assert_allclose(ref_tgt_audio[::3], audio)


def test_padded_cut_custom_recording():
    original_duration = 1.0  # seconds
    padded_duration = 2.0  # seconds

    # prepare cut
    cut = dummy_cut(unique_id=0, with_data=True, duration=original_duration)
    cut.target_recording = dummy_recording(
        unique_id=1, duration=cut.duration, with_data=True
    )
    target_recording = cut.load_target_recording()

    # prepare padded cut (MixedCut)
    padded_cut = cut.pad(duration=padded_duration)

    # check the padded cut (MixedCut) has the custom attribute
    assert padded_cut.has_custom("target_recording")

    # load the audio from the padded cut
    padded_target_recording = padded_cut.load_target_recording()

    # check the non-padded component is matching
    np.testing.assert_allclose(
        padded_target_recording[:, : cut.num_samples], target_recording
    )

    # check the padded component is zero
    assert np.all(padded_target_recording[:, cut.num_samples :] == 0)
