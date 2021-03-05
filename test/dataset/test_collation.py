import pytest

import torch

from lhotse.dataset.collation import TokenCollater, collate_audio, collate_features
from lhotse import CutSet
from lhotse.testing.dummies import dummy_cut, dummy_supervision


@pytest.mark.parametrize('add_bos', [True, False])
@pytest.mark.parametrize('add_eos', [True, False])
def test_token_collater(add_bos, add_eos):
    test_sentences = [
        "Testing the first sentence.",
        "Let's add some more punctuation, shall we?",
        "How about number 42!"
    ]

    cuts = CutSet.from_cuts(
        dummy_cut(
            idx,
            idx,
            supervisions=[dummy_supervision(
                idx, idx, text=sentence
            )]
        ) for idx, sentence in enumerate(test_sentences)
    )

    token_collater = TokenCollater(cuts, add_bos=add_bos, add_eos=add_eos)
    tokens_batch, tokens_lens = token_collater(cuts)

    assert isinstance(tokens_batch, torch.LongTensor)
    assert isinstance(tokens_lens, torch.IntTensor)

    extend = int(add_bos) + int(add_eos)
    expected_len = len(max(test_sentences, key=len)) + extend
    assert tokens_batch.shape == (len(test_sentences), expected_len)
    assert torch.all(tokens_lens == torch.IntTensor([len(sentence) + extend for sentence in test_sentences]))

    reconstructed = token_collater.inverse(tokens_batch, tokens_lens)
    assert reconstructed == test_sentences


def test_collate_audio_padding():
    cuts = CutSet.from_json('test/fixtures/ljspeech/cuts.json')
    assert len(set(cut.num_samples for cut in cuts)) > 1

    correct_pad = max(cut.num_samples for cut in cuts)
    audio, audio_lens = collate_audio(cuts)

    assert audio.shape[-1] == correct_pad
    assert max(audio_lens).item() == correct_pad


def test_collate_feature_padding():
    cuts = CutSet.from_json('test/fixtures/ljspeech/cuts.json')
    assert len(set(cut.num_frames for cut in cuts)) > 1

    correct_pad = max(cut.num_frames for cut in cuts)
    features, features_lens = collate_features(cuts)

    assert features.shape[1] == correct_pad
    assert max(features_lens).item() == correct_pad
