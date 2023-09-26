import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

from lhotse import RecordingSet, Seconds, compute_num_samples, validate
from lhotse.audio.backend import torchaudio_supports_ffmpeg
from lhotse.audio.utils import suppress_audio_loading_errors
from lhotse.augmentation import AugmentFn
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_audio, collate_features, collate_matrices
from lhotse.features import FeatureExtractor
from lhotse.features.kaldi.layers import _get_strided_batch_streaming


class UnsupervisedDataset(torch.utils.data.Dataset):
    """
    Dataset that contains no supervision - it only provides the features extracted from recordings.

    .. code-block::

        {
            'features': (B x T x F) tensor
            'features_lens': (B, ) tensor
        }
    """

    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        self._validate(cuts)
        features, features_lens = collate_features(cuts)
        return {
            "cuts": cuts,
            "features": features,
            "features_lens": features_lens,
        }

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_features for cut in cuts)


class UnsupervisedWaveformDataset(UnsupervisedDataset):
    """
    A variant of UnsupervisedDataset that provides waveform samples instead of features.
    The output is a tensor of shape (C, T), with C being the number of channels and T the number of audio samples.
    In this implementation, there will always be a single channel.

    Returns:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'audio_lens': (B, ) int tensor
        }
    """

    def __init__(self, collate: bool = True) -> None:
        super().__init__()
        self.collate = collate

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        if self.collate:
            audio, audio_lens = collate_audio(cuts)
            return {
                "cuts": cuts,
                "audio": audio,
                "audio_lens": audio_lens,
            }
        else:
            remain_cuts = []
            remain_audios = []
            for c in cuts:
                with suppress_audio_loading_errors():
                    remain_audios.append(c.load_audio())
                    remain_cuts.append(c)
            return {"cuts": CutSet.from_cuts(remain_cuts), "audio": remain_audios}

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)


class DynamicUnsupervisedDataset(UnsupervisedDataset):
    """
    An example dataset that shows how to use on-the-fly feature extraction in Lhotse.
    It accepts two additional inputs - a FeatureExtractor and an optional WavAugmenter for time-domain data augmentation..
    The output is approximately the same as that of the ``UnsupervisedDataset`` -
    there might be slight differences for ``MixedCut``s, because this dataset mixes them in the time domain,
    and ``UnsupervisedDataset`` does that in the feature domain.
    Cuts that are not mixed will yield identical results in both dataset classes.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        augment_fn: Optional[AugmentFn] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.augment_fn = augment_fn

    def __getitem__(self, cuts: CutSet) -> torch.Tensor:
        self._validate(cuts)

        def generate_cut(cuts: CutSet):
            for cut in cuts:
                with suppress_audio_loading_errors():
                    yield cut.compute_features(
                        extractor=self.feature_extractor,
                        augment_fn=self.augment_fn,
                    )

        features = collate_matrices(generate_cut(cuts))
        return features

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)


class RecordingChunkIterableDataset(IterableDataset):
    """
    This dataset iterates over chunks of a recording, for each recording provided.
    It supports setting a chunk_shift < chunk_size to run model predictions on
    overlapping audio chunks.

    The format of yielded items is the following::
        {
            "recording_id": str
            "begin_time": tensor with dtype=float32 shape=(1,)
            "end_time": tensor with dtype=float32 shape=(1,)
            "audio": tensor with dtype=float32 shape=(chunk_size_in_samples,)
        }

    Unlike most other datasets in Lhotse, this dataset does not yield batched items,
    and should be used like the following::

        >>> recordings = RecordingSet.from_file("my-recordings.jsonl.gz")
        ... dataset = RecordingChunkIterableDataset(recordings, chunk_size=30.0, chunk_shift=25.0)
        ... dloader = torch.utils.data.DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     collate_fn=audio_chunk_collate,
        ...     worker_init_fn=audio_chunk_worker_init_fn,
        ... )

    """

    def __init__(
        self, recordings: RecordingSet, chunk_size: Seconds, chunk_shift: Seconds
    ) -> None:
        self.recordings = list(recordings)
        self.chunk_size = chunk_size
        self.chunk_shift = chunk_shift
        self.start = 0
        self.end = len(self.recordings)

        self.validate()

    def validate(self) -> None:
        if not torchaudio_supports_ffmpeg():
            raise RuntimeError(
                "Using FFMPEG streamer backend for reading is supported only "
                "with PyTorch 1.12+ and torchaudio 0.12+"
            )

        for r in self.recordings:
            assert (
                len(r.sources) == 1
            ), f"We currently don't support multi-source audio in this dataset (got {len(r.sources)} sources in recording {r.id})."
            assert (
                r.sources[0].type == "file"
            ), f"We currently only support 'file' AudioSource type in this dataset (got: {r.sources[0].type} in recording {r.id})."
            assert (
                r.num_channels == 1
            ), f"We currently only support single-channel audio in this dataset (got {r.num_channels} channels in recording {r.id})."

    def __iter__(self):
        import torchaudio

        for r in self.recordings[self.start : self.end]:
            chunk_size = compute_num_samples(self.chunk_size, r.sampling_rate)
            chunk_shift = compute_num_samples(self.chunk_shift, r.sampling_rate)

            streamer = torchaudio.io.StreamReader(src=r.sources[0].source)
            assert streamer.num_src_streams == 1, (
                "Lhotse doesn't support files with more than one FFMPEG source stream yet "
                "(not to be confused with multi-channel)."
            )
            streamer.add_basic_audio_stream(frames_per_chunk=chunk_size)

            begin_time = 0
            end_time = self.chunk_size
            buffer = ShiftingBuffer(chunk_size=chunk_size, chunk_shift=chunk_shift)
            for (incoming_audio,) in streamer.stream():
                buffer.push(incoming_audio.squeeze())
                for chunk in buffer.get_chunks():
                    yield {
                        "recording_id": r.id,
                        "begin_time": torch.as_tensor(begin_time, dtype=torch.float32),
                        "end_time": torch.as_tensor(end_time, dtype=torch.float32),
                        "audio": chunk,
                    }
                    begin_time += self.chunk_shift
                    end_time = begin_time + self.chunk_size
            remainder = buffer.flush()
            if remainder.shape[0] > 0:
                yield {
                    "recording_id": r.id,
                    "begin_time": torch.as_tensor(begin_time, dtype=torch.float32),
                    "end_time": torch.as_tensor(end_time, dtype=torch.float32),
                    "audio": remainder,
                }


class ShiftingBuffer:
    """
    Utility for iterating over streaming audio chunks that supports chunk_shift < chunk_size.
    It is useful when running model predictions on overlapping chunks of audio data.
    """

    def __init__(self, chunk_size: int, chunk_shift: int):
        self.buf = torch.empty(1, 0)
        self.chunk_size = chunk_size
        self.chunk_shift = chunk_shift

    def push(self, audio: torch.Tensor) -> None:
        """Add new chunk of audio to the buffer. Expects shape (num_samples, )."""
        self.buf = torch.cat([self.buf, audio.unsqueeze(0)], dim=1)

    def get_chunks(self) -> torch.Tensor:
        """
        Retrieve chunks accumulated so far, adjusted for chunk_shift.
        For chunk_shift < chunk_size, there will typically be more chunks
        returned from this function than were pushed into the buffer because
        of overlap.
        The returned shape is (num_chunks, chunk_size).
        """
        out, self.buf = _get_strided_batch_streaming(
            self.buf,
            window_shift=self.chunk_shift,
            window_length=self.chunk_size,
            snip_edges=True,
        )
        return out.squeeze(0)

    def flush(self) -> torch.Tensor:
        """
        Flush out the remainder chunk from the buffer.
        Typically it will be shorter than chunk_size.
        The returned shape is (remainder_size, ).
        """
        out = self.buf.squeeze(0)
        self.buf = torch.empty(1, 0)
        return out


def audio_chunk_collate(batch: List[Dict]):
    from torch.utils.data import default_collate

    audios = [d.pop("audio") for d in batch]
    out = default_collate(batch)

    maxlen = max(a.shape[0] for a in audios)
    audio = torch.zeros((len(audios), maxlen))
    for i, a in enumerate(audios):
        audio[i, : a.shape[0]] = torch.as_tensor(a)
    out["audio"] = audio

    return out


def audio_chunk_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(
        math.ceil((overall_end - overall_start) / float(worker_info.num_workers))
    )
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
