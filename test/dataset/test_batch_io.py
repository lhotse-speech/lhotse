import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import numpy as np
import pytest
import torch.testing

from lhotse import CutSet, Fbank, MonoCut
from lhotse.dataset import AudioSamples, OnTheFlyFeatures, PrecomputedFeatures


@pytest.fixture
def libri_cut_set():
    cuts = CutSet.from_json("test/fixtures/libri/cuts.json")
    return CutSet.from_cuts(
        [
            cuts[0],
            cuts[0].with_id("copy-1"),
            cuts[0].with_id("copy-2"),
            cuts[0].append(cuts[0]),
        ]
    )


@pytest.mark.parametrize(
    "batchio", [AudioSamples, PrecomputedFeatures, partial(OnTheFlyFeatures, Fbank())]
)
@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize(
    "executor_type",
    [
        ThreadPoolExecutor,
        partial(ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")),
    ],
)
def test_batch_io(libri_cut_set, batchio, num_workers, executor_type):
    # does not fail / hang / etc.
    read_fn = batchio(num_workers=num_workers, executor_type=executor_type)
    read_fn(libri_cut_set)


def test_audio_samples_with_custom_field(libri_cut_set):
    batchio = AudioSamples()

    def attach_custom_audio(cut):
        """Simulate adding an additional custom recording"""
        cut.my_favorite_song = cut.recording.perturb_volume(factor=1.1)
        return cut

    # Reject mixed cuts (we don't support mixing custom attributes for now) and add custom audio
    cuts = libri_cut_set.filter(lambda c: isinstance(c, MonoCut)).map(
        attach_custom_audio
    )
    # does not fail / hang / etc.
    audio, audio_lens = batchio(cuts, recording_field="my_favorite_song")
    assert audio.shape[0] == 3

    # check that the audio is not the same as in the default 'recording' field
    audio_default, _ = batchio(cuts)
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(audio, audio_default)


def test_audio_samples_with_missing_custom_field(libri_cut_set):
    batchio = AudioSamples()
    with pytest.raises(AssertionError):
        audio, audio_lens = batchio(libri_cut_set, recording_field="my_favorite_song")


def test_audio_samples_equivalent_to_cut_set_load_audio(libri_cut_set):
    batchio = AudioSamples()
    audio, audio_lens = batchio(libri_cut_set)
    audio2, audio_lens2 = libri_cut_set.load_audio(collate=True)
    np.testing.assert_equal(audio2, audio.numpy())
    np.testing.assert_equal(audio_lens2, audio_lens.numpy())


def test_cut_set_load_audio_collate_false(libri_cut_set):
    audio = libri_cut_set.load_audio()
    assert isinstance(audio, list)
