from concurrent.futures import Executor
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from lhotse import CutSet
from lhotse.cut import Cut, MixedCut


class TokenCollater:
    """Collate list of tokens

    Map sentences to integers. Sentences are padded to equal length.
    Beginning and end-of-sequence symbols can be added.
    Call .inverse(tokens_batch, tokens_lens) to reconstruct batch as string sentences.

    Example:
        >>> token_collater = TokenCollater(cuts)
        >>> tokens_batch, tokens_lens = token_collater(cuts.subset(first=32))
        >>> original_sentences = token_collater.inverse(tokens_batch, tokens_lens)

    Returns:
        tokens_batch: IntTensor of shape (B, L)
            B: batch dimension, number of input sentences
            L: length of the longest sentence
        tokens_lens: IntTensor of shape (B,)
            Length of each sentence after adding <eos> and <bos>
            but before padding.
    """
    def __init__(
        self,
        cuts: CutSet,
        add_eos: bool = True,
        add_bos: bool = True,
        pad_symbol: str = "<pad>",
        bos_symbol: str = "<bos>",
        eos_symbol: str = "<eos>",
        unk_symbol: str = "<unk>",
    ):
        self.pad_symbol = pad_symbol
        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol
        self.unk_symbol = unk_symbol

        self.add_eos = add_eos
        self.add_bos = add_bos

        tokens = {
            char for cut in cuts
            for char in cut.supervisions[0].text
        }
        tokens_unique = (
            [pad_symbol, unk_symbol]
            + ([bos_symbol] if add_bos else [])
            + ([eos_symbol] if add_eos else [])
            + sorted(tokens)
        )

        self.token2idx = {token: idx for idx, token in enumerate(tokens_unique)}
        self.idx2token = [token for token in tokens_unique]

    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.Tensor]:
        token_sequences = [
            " ".join(supervision.text for supervision in cut.supervisions)
            for cut in cuts
        ]
        max_len = len(max(token_sequences, key=len))

        seqs = [
            ([self.bos_symbol] if self.add_bos else [])
            + list(seq)
            + ([self.eos_symbol] if self.add_eos else [])
            + [self.pad_symbol] * (max_len - len(seq))
            for seq in token_sequences
        ]

        tokens_batch = torch.from_numpy(np.array([
            [self.token2idx[token] for token in seq]
            for seq in seqs
        ], dtype=np.int64))

        tokens_lens = torch.IntTensor([
            len(seq) + int(self.add_eos) + int(self.add_bos)
            for seq in token_sequences
        ])

        return tokens_batch, tokens_lens

    def inverse(self, tokens_batch: torch.LongTensor, tokens_lens: torch.IntTensor) -> List[str]:
        start = 1 if self.add_bos else 0
        sentences = [
            "".join([
                self.idx2token[idx]
                for idx in tokens_list[start:end - int(self.add_eos)]
            ])
            for tokens_list, end in zip(tokens_batch, tokens_lens)
        ]
        return sentences


def collate_features(
        cuts: CutSet,
        pad_direction: str = 'right',
        executor: Optional[Executor] = None,
) -> Tuple[torch.Tensor, torch.IntTensor]:
    """
    Load features for all the cuts and return them as a batch in a torch tensor.
    The output shape is ``(batch, time, features)``.
    The cuts will be padded with silence if necessary.

    :param cuts: a :class:`CutSet` used to load the features.
    :param pad_direction: where to apply the padding (``right``, ``left``, or ``both``).
    :param executor: an instance of ThreadPoolExecutor or ProcessPoolExecutor; when provided,
        we will use it to read the features concurrently.
    :return: a tuple of tensors ``(features, features_lens)``.
    """
    assert all(cut.has_features for cut in cuts)
    features_lens = torch.tensor([cut.num_frames for cut in cuts], dtype=torch.int)
    cuts = maybe_pad(cuts, num_frames=max(features_lens).item(), direction=pad_direction)
    first_cut = next(iter(cuts))
    features = torch.empty(len(cuts), first_cut.num_frames, first_cut.num_features)
    if executor is None:
        for idx, cut in enumerate(cuts):
            features[idx] = _read_features(cut)
    else:
        for idx, example_features in enumerate(executor.map(_read_features, cuts)):
            features[idx] = example_features
    return features, features_lens


def collate_audio(
        cuts: CutSet,
        pad_direction: str = 'right',
        executor: Optional[Executor] = None,
) -> Tuple[torch.Tensor, torch.IntTensor]:
    """
    Load audio samples for all the cuts and return them as a batch in a torch tensor.
    The output shape is ``(batch, time)``.
    The cuts will be padded with silence if necessary.

    :param cuts: a :class:`CutSet` used to load the audio samples.
    :param pad_direction: where to apply the padding (``right``, ``left``, or ``both``).
    :param executor: an instance of ThreadPoolExecutor or ProcessPoolExecutor; when provided,
        we will use it to read audio concurrently.
    :return: a tuple of tensors ``(audio, audio_lens)``.
    """
    assert all(cut.has_recording for cut in cuts)
    audio_lens = torch.tensor([cut.num_samples for cut in cuts], dtype=torch.int32)
    cuts = maybe_pad(cuts, num_samples=max(audio_lens).item(), direction=pad_direction)
    first_cut = next(iter(cuts))
    audio = torch.empty(len(cuts), first_cut.num_samples)
    if executor is None:
        for idx, cut in enumerate(cuts):
            audio[idx] = _read_audio(cut)
    else:
        for idx, example_audio in enumerate(executor.map(_read_audio, cuts)):
            audio[idx] = example_audio
    return audio, audio_lens


def collate_multi_channel_features(cuts: CutSet) -> torch.Tensor:
    """
    Load features for all the cuts and return them as a batch in a torch tensor.
    The cuts have to be of type ``MixedCut`` and their tracks will be interpreted as individual channels.
    The output shape is ``(batch, channel, time, features)``.
    The cuts will be padded with silence if necessary.
    """
    assert all(cut.has_features for cut in cuts)
    assert all(isinstance(cut, MixedCut) for cut in cuts)
    cuts = maybe_pad(cuts)
    # Output tensor shape: (B, C, T, F) -> (batch_size, num_channels, num_frames, num_features)
    first_cut = next(iter(cuts))
    # TODO: make MixedCut more friendly to use with multi channel audio;
    #  discount PaddingCuts in "tracks" when specifying the number of channels
    features = torch.empty(len(cuts), len(first_cut.tracks), first_cut.num_frames, first_cut.num_features)
    for idx, cut in enumerate(cuts):
        features[idx] = torch.from_numpy(cut.load_features(mixed=False))
    return features


def collate_multi_channel_audio(cuts: CutSet) -> torch.Tensor:
    """
    Load audio samples for all the cuts and return them as a batch in a torch tensor.
    The cuts have to be of type ``MixedCut`` and their tracks will be interpreted as individual channels.
    The output shape is ``(batch, channel, time)``.
    The cuts will be padded with silence if necessary.
    """
    assert all(cut.has_recording for cut in cuts)
    assert all(isinstance(cut, MixedCut) for cut in cuts)
    cuts = maybe_pad(cuts)
    first_cut = next(iter(cuts))
    audio = torch.empty(len(cuts), len(first_cut.tracks), first_cut.num_samples)
    for idx, cut in enumerate(cuts):
        audio[idx] = torch.from_numpy(cut.load_audio())
    return audio


def collate_vectors(
        tensors: Iterable[Union[torch.Tensor, np.ndarray]],
        padding_value: Union[int, float] = CrossEntropyLoss().ignore_index,
        matching_shapes: bool = False
) -> torch.Tensor:
    """
    Convert an iterable of 1-D tensors (of possibly various lengths)
    into a single stacked tensor.

    :param tensors: an iterable of 1-D tensors.
    :param padding_value: the padding value inserted to make all tensors have the same length.
    :param matching_shapes: when ``True``, will fail when input tensors have different shapes.
    :return: a tensor with shape ``(B, L)`` where ``B`` is the number of input tensors and
        ``L`` is the number of items in the longest tensor.
    """
    tensors = [t if isinstance(t, torch.Tensor) else torch.from_numpy(t) for t in tensors]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D input tensors."
    longest = max(tensors, key=lambda t: t.shape[0])
    if matching_shapes:
        assert all(t.shape == longest.shape for t in tensors), \
            "All tensors must have the same shape when matching_shapes is set to True."
    result = longest.new_ones(len(tensors), longest.shape[0]) * padding_value
    for i, t in enumerate(tensors):
        result[i, :t.shape[0]] = t
    return result


def collate_matrices(
        tensors: Iterable[Union[torch.Tensor, np.ndarray]],
        padding_value: Union[int, float] = 0,
        matching_shapes: bool = False
) -> torch.Tensor:
    """
    Convert an iterable of 2-D tensors (of possibly various first dimension, but consistent second dimension)
    into a single stacked tensor.

    :param tensors: an iterable of 2-D tensors.
    :param padding_value: the padding value inserted to make all tensors have the same length.
    :param matching_shapes: when ``True``, will fail when input tensors have different shapes.
    :return: a tensor with shape ``(B, L, F)`` where ``B`` is the number of input tensors,
        ``L`` is the largest found shape[0], and ``F`` is equal to shape[1].
    """
    tensors = [t if isinstance(t, torch.Tensor) else torch.from_numpy(t) for t in tensors]
    assert all(len(t.shape) == 2 for t in tensors), "Expected only 2-D input tensors."
    longest = max(tensors, key=lambda t: t.shape[0])
    if matching_shapes:
        assert all(t.shape == longest.shape for t in tensors), \
            "All tensors must have the same shape when matching_shapes is set to True."
    result = longest.new_ones(len(tensors), *longest.shape) * padding_value
    for i, t in enumerate(tensors):
        result[i, :t.shape[0]] = t
    return result


def maybe_pad(
        cuts: CutSet,
        duration: int = None,
        num_frames: int = None,
        num_samples: int = None,
        direction: str = 'right'
) -> CutSet:
    """Check if all cuts' durations are equal and pad them to match the longest cut otherwise."""
    if len(set(c.duration for c in cuts)) == 1:
        # All cuts are of equal duration: nothing to do
        return cuts
    # Non-equal durations: silence padding
    return cuts.pad(
        duration=duration,
        num_frames=num_frames,
        num_samples=num_samples,
        direction=direction
    )


"""
Helper functions to dispatch jobs to the concurrent executors.
"""


def read_audio_from_cuts(
        cuts: Iterable[Cut], executor: Optional[Executor] = None
) -> List[torch.Tensor]:
    map_fn = map if executor is None else executor.map
    return list(map_fn(_read_audio, cuts))


def read_features_from_cuts(
        cuts: Iterable[Cut], executor: Optional[Executor] = None
) -> List[torch.Tensor]:
    map_fn = map if executor is None else executor.map
    return list(map_fn(_read_features, cuts))


def _read_audio(cut: Cut) -> torch.Tensor:
    return torch.from_numpy(cut.load_audio()[0])


def _read_features(cut: Cut) -> torch.Tensor:
    return torch.from_numpy(cut.load_features())
