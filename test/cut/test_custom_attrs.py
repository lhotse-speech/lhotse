import logging
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest
import torch

pytest.importorskip("lilcom", reason="Lilcom tests require lilcom.")

from lhotse import (
    LilcomFilesWriter,
    MonoCut,
    NumpyFilesWriter,
    Recording,
    compute_num_samples,
    fastcopy,
    validate,
)
from lhotse.audio import save_audio
from lhotse.cut import MixedCut
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
        save_audio(f.name, audio, sampling_rate=sampling_rate)
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
        save_audio(f.name, audio, sampling_rate=sampling_rate)
        f.flush()
        os.fsync(f)
        recording = Recording.from_file(f.name)

        # Note: MonoCut doesn't normally have a "my_favorite_song" attribute,
        #       and a "load_my_favorite_song()" method.
        #       We are dynamically extending it.
        cut = dummy_cut(0, duration=duration)
        cut.my_favorite_song = recording

        cut_trunc = cut.truncate(duration=5.0)

        restored_audio = cut_trunc.load_my_favorite_song()
        assert restored_audio.shape == (1, 80000)

        np.testing.assert_allclose(audio[:, :80000], restored_audio, atol=3e-4)


def test_cut_load_custom_recording_pad_right(deterministic_rng):
    sampling_rate = 16000
    duration = 52.4
    audio = np.random.randn(1, compute_num_samples(duration, sampling_rate)).astype(
        np.float32
    )
    audio /= np.abs(audio).max()  # normalize to [-1, 1]
    with NamedTemporaryFile(suffix=".wav") as f:
        save_audio(f.name, audio, sampling_rate=sampling_rate)
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
        save_audio(f.name, audio, sampling_rate=sampling_rate)
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
        save_audio(f.name, audio, sampling_rate=sampling_rate)
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


def test_copy_mixed_cut_with_custom_attr():
    cut = dummy_cut(0)
    cut = cut.mix(cut, offset_other_by=0.5)
    cut.some_attribute = "dummy"
    cpy = fastcopy(cut)
    assert cpy == cut


def test_mixed_cut_can_access_custom_directly():
    cut = dummy_cut(0, with_data=True)
    orig_custom = cut.custom
    cut = cut.pad(duration=cut.duration * 2)
    assert isinstance(cut, MixedCut)
    mixed_custom = cut.custom
    assert orig_custom.keys() == mixed_custom.keys()
    assert orig_custom == mixed_custom


@pytest.mark.parametrize("target_sampling_rate", [4000, 8000, 16000])
def test_cut_resample_custom_recording(target_sampling_rate):
    # has both .recording and.custom_recording
    cut = dummy_cut(0, duration=10.0, recording_duration=10.0, with_data=True)
    original_sample_rate = cut.sampling_rate
    original_custom_sample_rate = cut.custom_recording.sampling_rate

    cut_resampled_only_recording = cut.resample(target_sampling_rate)
    assert cut_resampled_only_recording.recording.sampling_rate == target_sampling_rate
    assert (
        cut_resampled_only_recording.custom_recording.sampling_rate
        == original_custom_sample_rate
    )

    cut_resampled_only_recording_explicit = cut.resample(
        target_sampling_rate, recording_field=None
    )
    assert (
        cut_resampled_only_recording_explicit.recording.sampling_rate
        == target_sampling_rate
    )
    assert (
        cut_resampled_only_recording_explicit.custom_recording.sampling_rate
        == original_custom_sample_rate
    )

    cut_resampled_only_custom_recording = cut.resample(
        target_sampling_rate, recording_field="custom_recording"
    )
    assert (
        cut_resampled_only_custom_recording.custom_recording.sampling_rate
        == target_sampling_rate
    )
    assert (
        cut_resampled_only_custom_recording.recording.sampling_rate
        == original_sample_rate
    )

    cut_resampled_both = cut.resample(target_sampling_rate).resample(
        target_sampling_rate, recording_field="custom_recording"
    )
    assert cut_resampled_both.recording.sampling_rate == target_sampling_rate
    assert cut_resampled_both.custom_recording.sampling_rate == target_sampling_rate


@pytest.mark.parametrize("target_sampling_rate", [4000, 8000, 16000])
def test_cut_resample_custom_recording_leaves_original_custom_field_intact(
    target_sampling_rate,
):
    # has both .recording and.custom_recording
    cut = dummy_cut(0, duration=10.0, recording_duration=10.0, with_data=True)
    if cut.sampling_rate == target_sampling_rate:
        pytest.skip("Skipping test because there is no resampling to do")

    cut_resampled = cut.resample(
        target_sampling_rate, recording_field="custom_recording"
    )

    assert (
        cut_resampled.custom_recording != cut.custom_recording
    ), "custom_recording should be different from the original"
    assert (
        cut_resampled.custom != cut.custom
    ), "set of custom fields should be different from the original"


@pytest.mark.parametrize("target_sampling_rate", [4000, 8000, 16000])
def test_cut_resample_custom_recording_fails_when_custom_recording_not_present(
    target_sampling_rate,
):
    cut = dummy_cut(0, duration=10.0, recording_duration=10.0, with_data=True)

    with pytest.raises(KeyError):
        cut.resample(target_sampling_rate, recording_field="nonexistent_recording")


def test_mixed_cut_load_custom_recording_after_append():
    """
    When two MonoCuts each carrying a custom Recording (target_audio)
    are appended, the resulting MixedCut.load_target_audio() should
    return the concatenated audio from both custom Recordings.
    """
    # Source at 16kHz, target_audio at 24kHz (different sampling rates)
    c1 = dummy_cut(0, recording=dummy_recording(0, with_data=True, sampling_rate=16000))
    c1.target_audio = dummy_recording(10, with_data=True, sampling_rate=24000)

    c2 = dummy_cut(1, recording=dummy_recording(1, with_data=True, sampling_rate=16000))
    c2.target_audio = dummy_recording(11, with_data=True, sampling_rate=24000)

    mixed = c1.append(c2)
    assert isinstance(mixed, MixedCut)

    # Source audio (main recording) already works — sanity check
    source_audio = mixed.load_audio()
    assert source_audio.shape == (1, 2 * 16000)  # 2s at 16kHz

    # Custom Recording across multiple tracks — the new feature
    target_audio = mixed.load_target_audio()
    assert target_audio.shape == (1, 2 * 24000)  # 2s at 24kHz

    # Verify the content matches the individual cuts' custom recordings
    t1 = c1.load_target_audio()
    t2 = c2.load_target_audio()
    np.testing.assert_array_equal(target_audio[:, :24000], t1)
    np.testing.assert_array_equal(target_audio[:, 24000:], t2)


def test_mixed_cut_load_custom_recording_after_append_same_sr():
    """Same as above but source and target at the same sampling rate."""
    c1 = dummy_cut(0, recording=dummy_recording(0, with_data=True))
    c1.target_audio = dummy_recording(10, with_data=True)

    c2 = dummy_cut(1, recording=dummy_recording(1, with_data=True))
    c2.target_audio = dummy_recording(11, with_data=True)

    mixed = c1.append(c2)
    target_audio = mixed.load_target_audio()
    assert target_audio.shape == (1, 2 * 16000)

    t1 = c1.load_target_audio()
    t2 = c2.load_target_audio()
    np.testing.assert_array_equal(target_audio[:, :16000], t1)
    np.testing.assert_array_equal(target_audio[:, 16000:], t2)


def test_mixed_cut_has_custom_with_multiple_tracks():
    """has_custom should return True even when multiple tracks carry the attribute."""
    c1 = dummy_cut(0, recording=dummy_recording(0, with_data=True))
    c1.target_audio = dummy_recording(10, with_data=True)

    c2 = dummy_cut(1, recording=dummy_recording(1, with_data=True))
    c2.target_audio = dummy_recording(11, with_data=True)

    mixed = c1.append(c2)
    assert mixed.has_custom("target_audio")


def test_mixed_cut_load_custom_recording_three_appended():
    """Append three cuts, each with a custom Recording."""
    cuts = []
    for i in range(3):
        c = dummy_cut(i, recording=dummy_recording(i, with_data=True))
        c.target_audio = dummy_recording(10 + i, with_data=True)
        cuts.append(c)

    mixed = cuts[0].append(cuts[1]).append(cuts[2])
    target_audio = mixed.load_target_audio()
    assert target_audio.shape == (1, 3 * 16000)

    for i, c in enumerate(cuts):
        t = c.load_target_audio()
        np.testing.assert_array_equal(target_audio[:, i * 16000 : (i + 1) * 16000], t)


def test_mixed_cut_load_custom_recording_single_track_still_works():
    """
    Regression: a single MonoCut padded to a MixedCut should still
    load its custom Recording correctly (existing behavior).
    """
    c = dummy_cut(0, recording=dummy_recording(0, with_data=True))
    c.target_audio = dummy_recording(10, with_data=True)
    mixed = c.pad(duration=c.duration * 2)  # creates a MixedCut with 1 data track
    assert isinstance(mixed, MixedCut)

    target_audio = mixed.load_target_audio()
    # 2s total duration at 16kHz, first 1s is data, second 1s is zero-padding
    assert target_audio.shape == (1, 2 * 16000)

    orig = c.load_target_audio()
    np.testing.assert_array_equal(target_audio[:, :16000], orig)
