import random
from math import isclose
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from lhotse import CutSet, MonoCut, NumpyFilesWriter
from lhotse.array import seconds_to_frames
from lhotse.dataset.collation import (
    TokenCollater,
    collate_audio,
    collate_custom_field,
    collate_features,
)
from lhotse.testing.dummies import (
    dummy_cut,
    dummy_multi_channel_recording,
    dummy_multi_cut,
    dummy_recording,
    dummy_supervision,
)


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


def test_collate_audio_padding_fault_tolerant_return_vals():
    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    assert len(set(cut.num_samples for cut in cuts)) > 1

    correct_pad = max(cut.num_samples for cut in cuts)
    audio, audio_lens, cuts_ok = collate_audio(cuts, fault_tolerant=True)

    assert len(cuts) == len(cuts_ok)
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
    with TemporaryDirectory() as d, NumpyFilesWriter(d) as writer:
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
    "pad_value",
    [0.0, None],  # user forgot to specify pad_value
)
def test_collate_custom_temporal_array_floats(pad_value):
    VOCAB_SIZE = 500

    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    max_num_frames = max(cut.num_frames for cut in cuts)

    with TemporaryDirectory() as d, NumpyFilesWriter(d) as writer:
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
            expected_pad_value = 0 if pad_value is None else pad_value
            torch.testing.assert_allclose(
                posteriors[idx, exp_len:],
                expected_pad_value * torch.ones_like(posteriors[idx, exp_len:]),
            )


@pytest.mark.parametrize(
    "pad_value",
    [-100, None],  # None means user forgot to specify pad_value
)
def test_collate_custom_temporal_array_ints(pad_value):
    CODEBOOK_SIZE = 512
    FRAME_SHIFT = 0.04

    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    max_num_frames = max(seconds_to_frames(cut.duration, FRAME_SHIFT) for cut in cuts)

    with TemporaryDirectory() as d, NumpyFilesWriter(d) as writer:
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
            # PyTorch < 1.9.0 doesn't have an assert_equal function.
            np.testing.assert_equal(codebook_indices[idx, :exp_len].numpy(), cbidxs)
            expected_pad_value = 0 if pad_value is None else pad_value
            np.testing.assert_equal(
                codebook_indices[idx, exp_len:].numpy(), expected_pad_value
            )


@pytest.mark.parametrize(
    "pad_value",
    [-100, None],  # None means user forgot to specify pad_value
)
def test_collate_custom_temporal_array_ints_with_truncate(pad_value):
    CODEBOOK_SIZE = 512
    FRAME_SHIFT = 0.04

    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")

    with TemporaryDirectory() as d, NumpyFilesWriter(d) as writer:
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

        cuts = cuts.truncate(max_duration=1, offset_type="start")
        max_num_frames = max(
            seconds_to_frames(cut.duration, FRAME_SHIFT) for cut in cuts
        )

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
        assert codebook_indices.dtype == torch.int64
        assert codebook_indices.shape == (len(cuts), max_num_frames)
        for idx, cbidxs in enumerate(expected_codebook_indices):
            # PyTorch < 1.9.0 doesn't have an assert_equal function.
            np.testing.assert_equal(
                codebook_indices[idx, :max_num_frames].numpy(), cbidxs[:max_num_frames]
            )


@pytest.mark.parametrize(
    "pad_direction",
    ["right", "left", "both"],
)
def test_collate_custom_temporal_array_ints(pad_direction):
    CODEBOOK_SIZE = 512
    FRAME_SHIFT = 0.04
    EXPECTED_PAD_VALUE = 0

    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    max_num_frames = max(seconds_to_frames(cut.duration, FRAME_SHIFT) for cut in cuts)

    with TemporaryDirectory() as d, NumpyFilesWriter(d) as writer:
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

        codebook_indices, codebook_indices_lens = collate_custom_field(
            cuts, "codebook_indices", pad_direction=pad_direction
        )

        assert isinstance(codebook_indices_lens, torch.Tensor)
        assert codebook_indices_lens.dtype == torch.int32
        assert codebook_indices_lens.shape == (len(cuts),)
        assert codebook_indices_lens.tolist() == [
            seconds_to_frames(c.duration, FRAME_SHIFT) for c in cuts
        ]

        assert isinstance(codebook_indices, torch.Tensor)
        assert (
            codebook_indices.dtype == torch.int64
        )  # the dtype got promoted by default
        assert codebook_indices.shape == (len(cuts), max_num_frames)
        for idx, cbidxs in enumerate(expected_codebook_indices):
            exp_len = cbidxs.shape[0]
            # PyTorch < 1.9.0 doesn't have an assert_equal function.
            if pad_direction == "right":
                np.testing.assert_equal(codebook_indices[idx, :exp_len].numpy(), cbidxs)
                np.testing.assert_equal(
                    codebook_indices[idx, exp_len:].numpy(), EXPECTED_PAD_VALUE
                )
            if pad_direction == "left":
                np.testing.assert_equal(
                    codebook_indices[idx, -exp_len:].numpy(), cbidxs
                )
                np.testing.assert_equal(
                    codebook_indices[idx, :-exp_len].numpy(), EXPECTED_PAD_VALUE
                )
            if pad_direction == "both":
                half = (max_num_frames - exp_len) // 2
                np.testing.assert_equal(
                    codebook_indices[idx, :half].numpy(), EXPECTED_PAD_VALUE
                )
                np.testing.assert_equal(
                    codebook_indices[idx, half : half + exp_len].numpy(), cbidxs
                )
                if half > 0:
                    # indexing like [idx, -0:] would return the whole array rather
                    # than an empty slice.
                    np.testing.assert_equal(
                        codebook_indices[idx, -half:].numpy(), EXPECTED_PAD_VALUE
                    )


def test_collate_custom_attribute_missing():
    cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json")
    with pytest.raises(AttributeError):
        collate_custom_field(cuts, "nonexistent_attribute")


def test_padding_issue_478():
    """
    https://github.com/lhotse-speech/lhotse/issues/478
    """
    with TemporaryDirectory() as d, NumpyFilesWriter(d) as writer:

        # Prepare data for cut 1.
        cut1 = MonoCut(
            "c1", start=0, duration=4.9, channel=0, recording=dummy_recording(1)
        )
        ali1 = np.random.randint(500, size=(121,))
        cut1.label_alignment = writer.store_array(
            "c1", ali1, frame_shift=0.04, temporal_dim=0
        )

        # Prepare data for cut 2.
        cut2 = MonoCut(
            "c2", start=0, duration=4.895, channel=0, recording=dummy_recording(2)
        )
        ali2 = np.random.randint(500, size=(121,))
        cut2.label_alignment = writer.store_array(
            "c2", ali2, frame_shift=0.04, temporal_dim=0
        )

        # Test collation behavior on this cutset.
        cuts = CutSet.from_cuts([cut1, cut2])
        label_alignments, label_alignment_lens = collate_custom_field(
            cuts, "label_alignment"
        )

        np.testing.assert_equal(label_alignments[0].numpy(), ali1)
        np.testing.assert_equal(label_alignments[1].numpy(), ali2)


def test_collate_cut_multi_channel_recording_and_custom_recording_diff_num_channels():
    cut = dummy_multi_cut(0, channel=[0, 1, 2, 3], with_data=True)
    cut.target_recording = dummy_multi_channel_recording(
        1, channel_ids=[0, 1], with_data=True
    )
    cut2 = dummy_multi_cut(2, duration=2.0, channel=[0, 1, 2, 3], with_data=True)
    cut2.target_recording = dummy_multi_channel_recording(
        3, duration=2.0, channel_ids=[0, 1], with_data=True
    )
    cuts = CutSet([cut, cut2])

    expected_lens = torch.tensor([16000, 32000], dtype=torch.int32)

    audio, audio_lens = collate_audio(cuts)
    assert audio.shape == (2, 4, 32000)  # batch x channel x time
    torch.testing.assert_close(audio_lens, expected_lens)

    target_audio, target_audio_lens = collate_audio(
        cuts, recording_field="target_recording"
    )
    assert target_audio.shape == (2, 2, 32000)  # batch x channel x time
    torch.testing.assert_close(audio_lens, expected_lens)
