import warnings
from concurrent.futures import Executor
from functools import partial
from itertools import repeat
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from lhotse import CutSet, Recording
from lhotse.audio import suppress_audio_loading_errors
from lhotse.audio.utils import suppress_video_loading_errors
from lhotse.cut import Cut, MixedCut
from lhotse.utils import DEFAULT_PADDING_VALUE, Seconds, compute_num_samples


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

        tokens = {char for cut in cuts for char in cut.supervisions[0].text}
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

        tokens_batch = torch.from_numpy(
            np.array(
                [[self.token2idx[token] for token in seq] for seq in seqs],
                dtype=np.int64,
            )
        )

        tokens_lens = torch.IntTensor(
            [
                len(seq) + int(self.add_eos) + int(self.add_bos)
                for seq in token_sequences
            ]
        )

        return tokens_batch, tokens_lens

    def inverse(
        self, tokens_batch: torch.LongTensor, tokens_lens: torch.IntTensor
    ) -> List[str]:
        start = 1 if self.add_bos else 0
        sentences = [
            "".join(
                [
                    self.idx2token[idx]
                    for idx in tokens_list[start : end - int(self.add_eos)]
                ]
            )
            for tokens_list, end in zip(tokens_batch, tokens_lens)
        ]
        return sentences


def collate_features(
    cuts: CutSet,
    pad_direction: str = "right",
    executor: Optional[Executor] = None,
    features_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    cuts = cuts.pad(num_frames=max(features_lens).item(), direction=pad_direction)
    first_cut = next(iter(cuts))
    features = torch.empty(
        len(cuts), first_cut.num_frames, first_cut.num_features, dtype=features_dtype
    )
    if executor is None:
        for idx, cut in enumerate(cuts):
            features[idx] = _read_features(cut)
    else:
        for idx, example_features in enumerate(executor.map(_read_features, cuts)):
            features[idx] = example_features
    return features, features_lens


def collate_audio(
    cuts: CutSet,
    pad_direction: str = "right",
    executor: Optional[Executor] = None,
    fault_tolerant: bool = False,
    recording_field: Optional[str] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, CutSet]
]:
    """
    Load audio samples for all the cuts and return them as a batch in a torch tensor.
    The output shape is ``(batch, time)``.
    The cuts will be padded with silence if necessary.

    :param cuts: a :class:`CutSet` used to load the audio samples.
    :param pad_direction: where to apply the padding (``right``, ``left``, or ``both``).
    :param executor: an instance of ThreadPoolExecutor or ProcessPoolExecutor; when provided,
        we will use it to read audio concurrently.
    :param fault_tolerant: when ``True``, the cuts for which audio loading failed
        will be skipped. Setting this parameter will cause the function to return a 3-tuple,
        where the third element is a CutSet for which the audio data were sucessfully read.
    :param recording_field: when specified, we will try to load recordings from a custom field with this name
        (i.e., ``cut.load_<recording_field>()`` instead of default ``cut.load_audio()``).
    :return: a tuple of tensors ``(audio, audio_lens)``, or ``(audio, audio_lens, cuts)``.
    """
    for cut in cuts:
        if recording_field is None:
            assert cut.has_recording, f"Missing recording in cut {cut.id}"
        else:
            assert cut.has_custom(
                recording_field
            ), f"Missing custom recording field {recording_field} in cut {cut.id}"

    # Remember how many samples were there in each cut (later, we might remove cuts that fail to load).
    sample_counts = []
    for cut in cuts:
        if recording_field is None:
            num_samples = cut.num_samples
        else:
            num_samples = compute_num_samples(
                cut.duration, sampling_rate=getattr(cut, recording_field).sampling_rate
            )
        sample_counts.append(num_samples)

    cuts = cuts.pad(
        duration=max(cut.duration for cut in cuts),
        direction=pad_direction,
        preserve_id=True,
    )

    # Note: returned "cuts" may be a subset of the original "cuts" if fault_tolerant=True.
    audios, cuts, sample_counts = read_audio_from_cuts(
        cuts,
        executor,
        suppress_errors=fault_tolerant,
        recording_field=recording_field,
        filter_aux_iter=sample_counts,
    )

    if len(audios[0].shape) == 1:
        audios = collate_vectors(audios, padding_value=0.0)
    else:
        audios = collate_matrices(
            [a.transpose(0, 1) for a in audios], padding_value=0.0
        ).transpose(1, 2)
    audio_lens = torch.tensor(sample_counts, dtype=torch.int32)

    if fault_tolerant:
        return audios, audio_lens, cuts
    else:
        return audios, audio_lens


collate_multi_channel_audio = collate_audio  # alias for backwards compatibility


def collate_video(
    cuts: CutSet,
    pad_direction: str = "right",
    executor: Optional[Executor] = None,
    fault_tolerant: bool = False,
):
    """
    Load video and audio for all cuts and return them as a batch in torch tensors.
    The output video shape is ``(batch, time, channel, height, width)``.
    The output audio shape is ``(batch, channel, time)``.
    The cuts will be padded with silence if necessary.

    .. note:: We expect each video to contain audio and the same number of audio channels.
        We may support padding missing channels at a later time.

    :param cuts: a :class:`CutSet` used to load the audio samples.
    :param pad_direction: where to apply the padding (``right``, ``left``, or ``both``).
    :param executor: an instance of ThreadPoolExecutor or ProcessPoolExecutor; when provided,
        we will use it to read video concurrently.
    :param fault_tolerant: when ``True``, the cuts for which video/audio loading failed
        will be skipped. Setting this parameter will cause the function to return a 5-tuple,
        where the fifth element is a CutSet for which the audio data were sucessfully read.
    :return: a tuple of tensors ``(video, video_lens, audio, audio_lens)``,
        or ``(video, video_lens, audio, audio_lens, cuts)``.
    """
    assert all(cut.has_video for cut in cuts)

    # Remember how many samples were there in each cut (later, we might remove cuts that fail to load).
    id2lens = {}
    for cut in cuts:
        id2lens[cut.id] = (cut.num_samples, cut.video.num_frames)

    cuts = cuts.pad(
        duration=max(c.duration for c in cuts),
        direction=pad_direction,
        preserve_id=True,
    )

    # Note: returned "cuts" may be a subset of the original "cuts" if fault_tolerant=True.
    videos, audios, cuts = read_video_from_cuts(
        cuts, executor, suppress_errors=fault_tolerant
    )

    videos = torch.stack(videos)  # B x T x C x H x W
    audios = torch.stack(audios)  # B x C x T
    audio_lens = torch.tensor([id2lens[cut.id][0] for cut in cuts], dtype=torch.int32)
    video_lens = torch.tensor([id2lens[cut.id][1] for cut in cuts], dtype=torch.int32)

    if fault_tolerant:
        return videos, video_lens, audios, audio_lens, cuts
    else:
        return videos, video_lens, audios, audio_lens


def collate_custom_field(
    cuts: CutSet,
    field: str,
    pad_value: Union[None, int, float] = None,
    pad_direction: str = "right",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load custom arrays for all the cuts and return them as a batch in a torch tensor.
    The output shapes are:

        - ``(batch, d0, d1, d2, ...)`` for :class:`lhotse.array.Array` of shape ``(d0, d1, d2, ...)``.
            Note: all arrays have to be of the same shape, as we expect these represent fixed-size
            embeddings.

        - ``(batch, d0, pad_dt, d1, ...)`` for :class:`lhotse.array.TemporalArray` of shape
            ``(d0, dt, d1, ...)`` where ``dt`` indicates temporal dimension (variable-sized),
            and ``pad_dt`` indicates temporal dimension after padding (equal-sized for all cuts).
            We expect these represent temporal data, such as alignments, posteriors, features, etc.

        - ``(batch, )`` for anything else, such as int or float: we will simply stack them into
            a list and tensorize it.

    .. note:: This function disregards the ``frame_shift`` attribute of
        :class:`lhotse.array.TemporalArray` when padding; it simply pads all the arrays
        to the longest one found in the mini-batch. Because of that, the function
        will work correctly even if the user supplied inconsistent meta-data.

    .. note:: Temporal arrays of integer type that are smaller than torch.int64,
        will be automatically promoted to torch.int64.

    :param cuts: a :class:`CutSet` used to load the features.
    :param field: name of the custom field to be retrieved.
    :param pad_value: value to be used for padding the temporal arrays.
        Ignored for non-temporal array and non-array attributes.
    :param pad_direction: where to apply the padding (``right``, ``left``, or ``both``).
    :return: a collated data tensor, or a tuple of tensors ``(collated_data, sequence_lens)``.
    """
    from lhotse.array import Array, TemporalArray

    first_manifest = getattr(cuts[0], field)
    if isinstance(first_manifest, Array):
        # Expected data type: fixed-size embeddings.
        # Simply stack across a new dimension inserted at 0.
        assert all(getattr(c, field).shape == first_manifest.shape for c in cuts), (
            "Cannot collate manifests of type Array with different shapes, "
            "because we don't know which dimension must be padded. "
            "Use TemporalArray manifests and try again."
        )
        return torch.stack([torch.from_numpy(c.load_custom(field)) for c in cuts])
    elif isinstance(first_manifest, TemporalArray):
        # Expected data type: variable-sized tensors (along only one dimension).
        # Pad across that dimension, then stack at dimension 0.
        if pad_value is None:
            warnings.warn(
                f"Argument 'pad_value' not passed -- we will pad field '{field}' "
                f"with {DEFAULT_PADDING_VALUE}."
            )
            pad_value = DEFAULT_PADDING_VALUE
        temporal_dim = first_manifest.temporal_dim

        # We avoid cuts.pad() because the users might be defining frame_shift differently
        # that we typically do in Lhotse. This may result in extra padding where they
        # expected none to happen. See: https://github.com/lhotse-speech/lhotse/issues/478
        #   cuts = cuts.pad(direction=pad_direction, pad_value_dict={field: pad_value})
        #   tensors = torch.stack([torch.from_numpy(c.load_custom(field)) for c in cuts])

        # Instead, we're going to load everything and pad to the longest sequence.
        arrs = [torch.from_numpy(c.load_custom(field)) for c in cuts]
        arr_lens = torch.tensor(
            [a.shape[temporal_dim] for a in arrs], dtype=torch.int32
        )
        largest_arr = max(arrs, key=torch.numel)
        maxlen = largest_arr.shape[temporal_dim]
        collated_shape = (len(arrs), *largest_arr.shape)
        dtype = largest_arr.dtype
        if any(d == dtype for d in (torch.uint8, torch.int8, torch.int16, torch.int32)):
            dtype = torch.int64
        tensors = pad_value * torch.ones(collated_shape, dtype=dtype)
        for aidx, a in enumerate(arrs):
            alen = a.shape[temporal_dim]
            # Construct an index expression such as tensors[:, :alen, :, :] programmatically;
            # All indices are set to ':', besides temporal dim which is determined on pad_direction.
            if pad_direction == "right":
                temporal_slice = slice(0, alen)
            elif pad_direction == "left":
                temporal_slice = slice(maxlen - alen, maxlen)
            elif pad_direction == "both":
                half = (maxlen - alen) // 2
                temporal_slice = slice(half, maxlen - half)
            else:
                raise ValueError(
                    f"Unexpected pad_direction argument: '{pad_direction}'"
                )
            indices = (aidx,) + tuple(
                temporal_slice if i == temporal_dim else slice(None, None, None)
                for i in range(len(a.shape))
            )
            tensors[indices] = a

        return tensors, arr_lens
    else:
        # Expected data type: int, float, string, etc.
        # Get a list of them and convert to a tensor.
        return torch.tensor([getattr(c, field) for c in cuts])


def collate_multi_channel_features(cuts: CutSet) -> torch.Tensor:
    """
    Load features for all the cuts and return them as a batch in a torch tensor.
    The cuts have to be of type ``MixedCut`` and their tracks will be interpreted as individual channels.
    The output shape is ``(batch, channel, time, features)``.
    The cuts will be padded with silence if necessary.
    """
    assert all(cut.has_features for cut in cuts)
    assert all(isinstance(cut, MixedCut) for cut in cuts)
    cuts = cuts.pad(cuts)
    # Output tensor shape: (B, C, T, F) -> (batch_size, num_channels, num_frames, num_features)
    first_cut = next(iter(cuts))
    # TODO: make MixedCut more friendly to use with multi channel audio;
    #  discount PaddingCuts in "tracks" when specifying the number of channels
    features = torch.empty(
        len(cuts), len(first_cut.tracks), first_cut.num_frames, first_cut.num_features
    )
    for idx, cut in enumerate(cuts):
        features[idx] = torch.from_numpy(cut.load_features(mixed=False))
    return features


def collate_vectors(
    tensors: Iterable[Union[torch.Tensor, np.ndarray]],
    padding_value: Union[int, float] = CrossEntropyLoss().ignore_index,
    matching_shapes: bool = False,
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
    tensors = [
        t if isinstance(t, torch.Tensor) else torch.from_numpy(t) for t in tensors
    ]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D input tensors."
    longest = max(tensors, key=lambda t: t.shape[0])
    if matching_shapes:
        assert all(
            t.shape == longest.shape for t in tensors
        ), "All tensors must have the same shape when matching_shapes is set to True."
    result = longest.new_ones(len(tensors), longest.shape[0]) * padding_value
    for i, t in enumerate(tensors):
        result[i, : t.shape[0]] = t
    return result


def collate_matrices(
    tensors: Iterable[Union[torch.Tensor, np.ndarray]],
    padding_value: Union[int, float] = 0,
    matching_shapes: bool = False,
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
    tensors = [
        t if isinstance(t, torch.Tensor) else torch.from_numpy(t) for t in tensors
    ]
    assert all(len(t.shape) == 2 for t in tensors), "Expected only 2-D input tensors."
    longest = max(tensors, key=lambda t: t.shape[0])
    if matching_shapes:
        assert all(
            t.shape == longest.shape for t in tensors
        ), "All tensors must have the same shape when matching_shapes is set to True."
    result = longest.new_ones(len(tensors), *longest.shape) * padding_value
    for i, t in enumerate(tensors):
        result[i, : t.shape[0]] = t
    return result


"""
Helper functions to dispatch jobs to the concurrent executors.
"""


def read_audio_from_cuts(
    cuts: Iterable[Cut],
    executor: Optional[Executor] = None,
    suppress_errors: bool = False,
    recording_field: Optional[str] = None,
    filter_aux_iter: Optional[Iterable] = None,
) -> Union[Tuple[List[torch.Tensor], CutSet], Tuple[List[torch.Tensor], CutSet, List]]:
    """
    Loads audio data from an iterable of cuts.

    :param cuts: a CutSet or iterable of cuts.
    :param executor: optional Executor (e.g., ThreadPoolExecutor or ProcessPoolExecutor)
        to perform the audio reads in parallel.
    :param suppress_errors: when set to ``True``, will enable fault-tolerant data reads;
        we will skip the cuts and audio data for the instances that failed (and emit a warning).
        When ``False`` (default), the errors will not be suppressed.
    :param recording_field: when specified, we will try to load recordings from a custom field with this name
        (i.e., ``cut.load_<recording_field>()`` instead of default ``cut.load_audio()``).
    :param filter_aux_iter: when specified, we will iterate over this iterator and discard the elements
        for which a corresponding cut failed to load audio, if ``suppress_errors`` is set to ``True``.
        This iterator is expected to be of the same length as ``cuts``.
    :return: a tuple of two items: a list of audio tensors (with different shapes),
        and a list of cuts for which we read the data successfully.
        If ``filter_aux_iter`` is specified, it returns a 3-tuple where the third element is
        the filtered auxiliary iterator.
    """
    aux_requested = True
    if filter_aux_iter is None:
        filter_aux_iter = repeat([None])
        aux_requested = False
    map_fn = map if executor is None else executor.map
    audios = []
    ok_cuts = []
    aux_iter_out = []
    for idx, (cut, maybe_audio, aux_item) in enumerate(
        zip(
            cuts,
            map_fn(
                partial(
                    _read_audio,
                    suppress_errors=suppress_errors,
                    recording_field=recording_field,
                ),
                cuts,
            ),
            filter_aux_iter,
        )
    ):
        if maybe_audio is None:
            continue
        else:
            audios.append(maybe_audio)
            ok_cuts.append(cut)
            aux_iter_out.append(aux_item)
    ans = (audios, CutSet.from_cuts(ok_cuts))
    if aux_requested:
        ans = ans + (aux_iter_out,)
    return ans


def read_video_from_cuts(
    cuts: Iterable[Cut],
    executor: Optional[Executor] = None,
    suppress_errors: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], CutSet]:
    """
    Loads audio data from an iterable of cuts.

    :param cuts: a CutSet or iterable of cuts.
    :param executor: optional Executor (e.g., ThreadPoolExecutor or ProcessPoolExecutor)
        to perform the audio reads in parallel.
    :param suppress_errors: when set to ``True``, will enable fault-tolerant data reads;
        we will skip the cuts and audio data for the instances that failed (and emit a warning).
        When ``False`` (default), the errors will not be suppressed.
    :return: a tuple of two items: a list of audio tensors (with different shapes),
        and a list of cuts for which we read the data successfully.
    """
    map_fn = map if executor is None else executor.map
    videos = []
    audios = []
    ok_cuts = []
    for idx, (cut, maybe_ans) in enumerate(
        zip(
            cuts,
            map_fn(
                partial(
                    _read_video,
                    suppress_errors=suppress_errors,
                ),
                cuts,
            ),
        )
    ):
        if maybe_ans is None:
            continue
        else:
            video, audio = maybe_ans
            videos.append(video)
            audios.append(audio)
            ok_cuts.append(cut)
    return videos, audios, CutSet.from_cuts(ok_cuts)


def read_features_from_cuts(
    cuts: Iterable[Cut], executor: Optional[Executor] = None
) -> List[torch.Tensor]:
    map_fn = map if executor is None else executor.map
    return list(map_fn(_read_features, cuts))


def _read_audio(
    cut: Cut, suppress_errors: bool = False, recording_field: Optional[str] = None
) -> Optional[torch.Tensor]:
    """
    Loads audio data from cut, or returns None if there was an error
    and ``suppress_errors`` was set to ``True``.
    """
    with suppress_audio_loading_errors(enabled=suppress_errors):
        if recording_field is None:
            audio = cut.load_audio()
        else:
            attr = getattr(cut, recording_field)
            assert isinstance(
                attr, Recording
            ), f"Expected 'getattr(cut, {recording_field})' to yield Recording, got {type(attr)}"
            audio = cut.load_custom(recording_field)
        if audio.shape[0] == 1:
            audio = audio.squeeze(0)  # collapse channel dim if mono
        return torch.from_numpy(audio)


def _read_features(cut: Cut) -> torch.Tensor:
    return torch.from_numpy(cut.load_features())


def _read_video(
    cut: Cut, suppress_errors: bool = False
) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Loads video + audio data from cut, or returns None if there was an error
    and ``suppress_errors`` was set to ``True``.
    """
    with suppress_video_loading_errors(enabled=suppress_errors):
        return cut.load_video()
