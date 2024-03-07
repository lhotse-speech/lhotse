from typing import Tuple

import numpy as np
import pytest
import torch

from lhotse import CutSet, Fbank, compute_num_frames
from lhotse.cut import Cut
from lhotse.cut.text import TextExample
from lhotse.dataset import DynamicBucketingSampler, DynamicCutSampler
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.sampling.base import TokenConstraint
from lhotse.testing.dummies import DummyManifest


@pytest.fixture
def text_source():
    def get_text_source():
        while True:
            for item in ("hello world", "example text", "this is my text data"):
                # for this example, "bytes are all you need", could be BPE, etc.
                yield TextExample(item, np.frombuffer(item.encode("utf-8"), np.int8))

    return get_text_source()


def test_text_dynamic_cut_sampler_static_batch_size(text_source):
    sampler = DynamicCutSampler(
        text_source, constraint=TokenConstraint(max_examples=16)
    )
    batch = next(iter(sampler))
    assert len(batch) == 16
    assert isinstance(batch[0], TextExample)
    assert isinstance(batch[0].text, str)


def test_text_dynamic_cut_sampler_dynamic_batch_size(text_source):
    sampler = DynamicCutSampler(text_source, constraint=TokenConstraint(max_tokens=256))
    batch = next(iter(sampler))
    assert isinstance(batch[0], TextExample)
    assert isinstance(batch[0].text, str)
    assert len(batch) == 12


def test_text_dynamic_bucketing_sampler(text_source):
    sampler = DynamicBucketingSampler(
        text_source,
        num_buckets=2,
        constraint=TokenConstraint(max_tokens=256, quadratic_length=128),
    )
    batch = next(iter(sampler))
    assert isinstance(batch[0], TextExample)
    assert isinstance(batch[0].text, str)
    assert len(batch) == 11


class TextDataset(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.Tensor]:
        from lhotse.dataset.collation import collate_vectors

        tokens = collate_vectors(
            [item.tokens.astype(np.int32) for item in cuts], padding_value=-1
        )
        token_lens = torch.LongTensor([item.tokens.shape[0] for item in cuts])
        return tokens, token_lens


def test_text_dataloader_with_dynamic_bucketing_sampler(text_source):
    sampler = DynamicBucketingSampler(
        text_source,
        num_buckets=2,
        constraint=TokenConstraint(max_tokens=256, quadratic_length=128),
    )
    dloader = torch.utils.data.DataLoader(
        TextDataset(), sampler=sampler, batch_size=None
    )
    batch = next(iter(dloader))
    assert isinstance(batch[0], torch.Tensor)
    assert batch[0].shape == (11, 20)  # (batch_size, seq_len)
    assert isinstance(batch[1], torch.Tensor)
    assert batch[1].shape == (11,)


class MixedAudioTextDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.text_dataset = TextDataset()

    def __getitem__(
        self, cuts: CutSet
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        text_cuts = cuts.filter(lambda c: isinstance(c, TextExample)).to_eager()
        if text_cuts:
            tokens, token_lens = self.text_dataset[text_cuts]
        else:
            tokens, token_lens = None, None

        audio_cuts = cuts.filter(lambda c: isinstance(c, Cut)).to_eager()
        if audio_cuts:
            audio, audio_lens = collate_audio(audio_cuts)
        else:
            audio, audio_lens = None, None

        return tokens, token_lens, audio, audio_lens


def _assign_num_tokens_to_cut(cut, frame_shift=0.01):
    cut.num_tokens = compute_num_frames(
        cut.duration, frame_shift=frame_shift, sampling_rate=cut.sampling_rate
    )
    return cut


@pytest.fixture
def audio_source():
    return (
        DummyManifest(CutSet, begin_id=0, end_id=10, with_data=True)
        .map(_assign_num_tokens_to_cut)
        .repeat()
    )


def test_audio_and_text_dataloader_with_dynamic_sampler(text_source, audio_source):
    mixed = CutSet.mux(text_source, audio_source, weights=[0.7, 0.3])
    sampler = DynamicCutSampler(
        mixed,
        constraint=TokenConstraint(max_tokens=1024, quadratic_length=128),
    )
    dloader = torch.utils.data.DataLoader(
        MixedAudioTextDataset(), sampler=sampler, batch_size=None
    )
    batch = next(iter(dloader))
    assert isinstance(batch[0], torch.Tensor)
    assert batch[0].shape == (3, 20)  # (batch_size, seq_len)
    assert isinstance(batch[1], torch.Tensor)
    assert batch[1].shape == (3,)
    assert isinstance(batch[2], torch.Tensor)
    assert batch[2].shape == (2, 16000)  # (batch_size, seq_len)
    assert isinstance(batch[3], torch.Tensor)
    assert batch[3].shape == (2,)
