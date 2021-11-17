import random
from math import isclose
from tempfile import NamedTemporaryFile

import pytest
import torch
import numpy as np

from lhotse import CutSet, NumpyHdf5Writer
from lhotse.array import seconds_to_frames
from lhotse.dataset.collation import (
    TokenCollater,
    collate_audio,
    collate_custom_field,
    collate_features,
)
from lhotse.testing.dummies import dummy_cut, dummy_supervision
from lhotse.utils import nullcontext as does_not_raise


@pytest.mark.parametrize("add_bos", [True, False])
@pytest.mark.parametrize("add_eos", [True, False])
def test_token_collater(add_bos, add_eos):
    test_sentences = [
        "Testing the first sentence.",
        "Let's add some more punctuation, shall we?",
        "How about number 42!",
    ]

    cuts = CutSet.from_cuts(
        dummy_cut(idx, idx, supervisions=[dummy_supervision(idx, idx, text=sentence)])
        for idx, sentence in enumerate(test_sentences)
    )

    token_collater = TokenCollater(cuts, add_bos=add_bos, add_eos=add_eos)
    tokens_batch, tokens_lens = token_collater(cuts)

    assert isinstance(tokens_batch, torch.LongTensor)
    assert isinstance(tokens_lens, torch.IntTensor)

    extend = int(add_bos) + int(add_eos)
    expected_len = len(max(test_sentences, key=len)) + extend
    assert tokens_batch.shape == (len(test_sentences), expected_len)
    assert torch.all(
        tokens_lens
        == torch.IntTensor([len(sentence) + extend for sentence in test_sentences])
    )

    reconstructed = token_collater.inverse(tokens_batch, tokens_lens)
    assert reconstructed == test_sentences


def test_collate_audio_padding():
    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    assert len(set(cut.num_samples for cut in cuts)) > 1

    correct_pad = max(cut.num_samples for cut in cuts)
    audio, audio_lens = collate_audio(cuts)

    assert audio.shape[-1] == correct_pad
    assert max(audio_lens).item() == correct_pad


def test_collate_feature_padding():
    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    assert len(set(cut.num_frames for cut in cuts)) > 1

    correct_pad = max(cut.num_frames for cut in cuts)
    features, features_lens = collate_features(cuts)

    assert features.shape[1] == correct_pad
    assert max(features_lens).item() == correct_pad


def test_collate_custom_array():
    EMBEDDING_SIZE = 300

    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    with NamedTemporaryFile(suffix=".h5") as f, NumpyHdf5Writer(f.name) as writer:
        expected_xvectors = []
        for cut in cuts:
            expected_xvectors.append(np.random.randn(EMBEDDING_SIZE).astype(np.float32))
            cut.xvector = writer.store_array(cut.id, expected_xvectors[-1])

        xvectors = collate_custom_field(cuts, "xvector")
        assert isinstance(xvectors, torch.Tensor)
        assert xvectors.dtype == torch.float32
        assert xvectors.shape == (len(cuts), EMBEDDING_SIZE)
        for idx, xvec in enumerate(expected_xvectors):
            torch.testing.assert_allclose(xvectors[idx], xvec)


def test_collate_custom_numbers():
    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    expected_snrs = []
    for cut in cuts:
        expected_snrs.append(random.random() * 20)
        cut.snr = expected_snrs[-1]

    snrs = collate_custom_field(cuts, "snr")
    assert isinstance(snrs, torch.Tensor)
    assert snrs.dtype == torch.float32
    assert snrs.shape == (len(cuts),)
    for idx, snr in enumerate(expected_snrs):
        assert isclose(snrs[idx], snr, abs_tol=1e-5)


@pytest.mark.parametrize(
    ["pad_value", "exception_expectation"],
    [
        (0.0, does_not_raise()),
        (None, pytest.raises(AssertionError)),  # user forgot to specify pad_value
    ],
)
def test_collate_custom_temporal_array_floats(pad_value, exception_expectation):
    VOCAB_SIZE = 500

    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    max_num_frames = max(cut.num_frames for cut in cuts)

    with NamedTemporaryFile(suffix=".h5") as f, NumpyHdf5Writer(f.name) as writer:
        expected_posteriors = []
        for cut in cuts:
            expected_posteriors.append(
                np.random.randn(cut.num_frames, VOCAB_SIZE).astype(np.float32)
            )
            cut.posterior = writer.store_array(
                cut.id,
                expected_posteriors[-1],
                frame_shift=cut.frame_shift,
                temporal_dim=0,
            )

        with exception_expectation:

            posteriors, posterior_lens = collate_custom_field(
                cuts, "posterior", pad_value=pad_value
            )

            assert isinstance(posterior_lens, torch.Tensor)
            assert posterior_lens.dtype == torch.int32
            assert posterior_lens.shape == (len(cuts),)
            assert posterior_lens.tolist() == [c.num_frames for c in cuts]

            assert isinstance(posteriors, torch.Tensor)
            assert posteriors.dtype == torch.float32
            assert posteriors.shape == (len(cuts), max_num_frames, VOCAB_SIZE)
            for idx, post in enumerate(expected_posteriors):
                exp_len = post.shape[0]
                torch.testing.assert_allclose(posteriors[idx, :exp_len], post)
                torch.testing.assert_allclose(
                    posteriors[idx, exp_len:],
                    pad_value * torch.ones_like(posteriors[idx, exp_len:]),
                )


@pytest.mark.parametrize(
    ["pad_value", "exception_expectation"],
    [
        (-100, does_not_raise()),
        (None, pytest.raises(AssertionError)),  # user forgot to specify pad_value
    ],
)
def test_collate_custom_temporal_array_ints(pad_value, exception_expectation):
    CODEBOOK_SIZE = 512
    FRAME_SHIFT = 0.4

    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    max_num_frames = max(seconds_to_frames(cut.duration, FRAME_SHIFT) for cut in cuts)

    with NamedTemporaryFile(suffix=".h5") as f, NumpyHdf5Writer(f.name) as writer:
        expected_codebook_indices = []
        for cut in cuts:
            expected_codebook_indices.append(
                np.random.randint(
                    CODEBOOK_SIZE, size=(seconds_to_frames(cut.duration, FRAME_SHIFT),)
                ).astype(np.int16)
            )
            cut.codebook_indices = writer.store_array(
                cut.id,
                expected_codebook_indices[-1],
                frame_shift=FRAME_SHIFT,
                temporal_dim=0,
            )

        with exception_expectation:

            codebook_indices, codebook_indices_lens = collate_custom_field(
                cuts, "codebook_indices", pad_value=pad_value
            )

            assert isinstance(codebook_indices_lens, torch.Tensor)
            assert codebook_indices_lens.dtype == torch.int32
            assert codebook_indices_lens.shape == (len(cuts),)
            assert codebook_indices_lens.tolist() == [
                seconds_to_frames(c.duration, FRAME_SHIFT) for c in cuts
            ]

            assert isinstance(codebook_indices, torch.Tensor)
            assert codebook_indices.dtype == torch.int16
            assert codebook_indices.shape == (len(cuts), max_num_frames)
            for idx, cbidxs in enumerate(expected_codebook_indices):
                exp_len = cbidxs.shape[0]
                torch.testing.assert_allclose(codebook_indices[idx, :exp_len], cbidxs)
                torch.testing.assert_allclose(
                    codebook_indices[idx, exp_len:],
                    pad_value * torch.ones_like(codebook_indices[idx, exp_len:]),
                )
