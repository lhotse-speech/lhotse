"""
Note: this module is very heavily based on a torchaudio tutorial about forced
alignment with Wav2Vec2 created by Moto Hira.

Link: https://pytorch.org/audio/stable/pipelines.html
"""
from typing import Generator, List, NamedTuple, Sequence

import torch
import torchaudio

from lhotse import CutSet, MonoCut
from lhotse.supervision import AlignmentItem


def align_with_torchaudio(
    cuts: CutSet, bundle_name: str = "WAV2VEC2_ASR_BASE_960H", device: str = "cpu"
) -> Generator[MonoCut, None, None]:
    """
    Use a pretrained model from torchaudio (such as Wav2Vec2) to perform forced
    word-level alignment of a CutSet.

    This means that for every SupervisionSegment with a transcript, we will find the
    start and end timestamps for each of the words.

    We support cuts with multiple supervisions -- the forced alignment will be done
    for every supervision region separately, and attached to the relevant supervision.

    Note: this is an experimental feature of Lhotse, and is not guaranteed to yield
    high quality of data.

    See torchaudio's documentation and tutorials for more details:
    - https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
    - https://pytorch.org/audio/stable/pipelines.html

    :param cuts: input CutSet.
    :param bundle_name: name of the selected pretrained model from torchaudio.
        By default, we use WAV2VEC2_ASR_BASE_960H.
    :param device: device on which to run the computation.
    :return: a generator of cuts that have the "alignment" field set in each of
        their supervisions.
    """
    bundle = getattr(torchaudio.pipelines, bundle_name)
    sampling_rate = bundle.sample_rate
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    device = torch.device(device)
    dictionary = {c: i for i, c in enumerate(labels)}

    for cut in cuts:
        assert not cut.has_overlapping_supervisions, (
            f"We don't support forced alignment of cuts with overlapping supervisions "
            f"(cut ID: '{cut.id}')"
        )

        for idx, subcut in enumerate(cut.trim_to_supervisions(keep_overlapping=False)):
            sup = subcut.supervisions[0]
            waveform = torch.as_tensor(
                subcut.resample(sampling_rate).load_audio(), device=device
            )
            # TODO: smarter text norm
            transcript = sup.text.replace(" ", "|")
            # TODO: handle OOV symbols gracefully
            tokens = [dictionary[c] for c in transcript]

            with torch.inference_mode():
                emissions, _ = model(waveform)
                emissions = torch.log_softmax(emissions, dim=-1)
            emission = emissions[0].cpu()

            trellis = _get_trellis(emission, tokens)

            path = _backtrack(trellis, emission, tokens)

            segments = _merge_repeats(path, transcript)

            word_segments = _merge_words(segments)

            # Ratio of number of samples to number of frames
            ratio = waveform.size(1) / emission.size(0)
            alignment = [
                AlignmentItem.new(
                    symbol=ws.label,
                    start=round(int(ratio * ws.start) / sampling_rate, ndigits=8),
                    duration=round(
                        int(ratio * (ws.end - ws.start)) / sampling_rate, ndigits=8
                    ),
                    score=ws.score,
                )
                for ws in word_segments
            ]
            sup = sup.with_alignment(kind="word", alignment=alignment)

            cut.supervisions[idx] = sup

        yield cut


def _get_trellis(
    emission: torch.Tensor, tokens: Sequence[int], blank_id: int = 0
) -> torch.Tensor:
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


class Point(NamedTuple):
    token_index: int
    time_index: int
    score: float


def _backtrack(
    trellis: torch.Tensor,
    emission: torch.Tensor,
    tokens: Sequence[int],
    blank_id: int = 0,
) -> List[Point]:
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


class Segment(NamedTuple):
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def _merge_repeats(path: List[Point], transcript: str) -> List[Segment]:
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def _merge_words(segments: List[Segment], separator: str = "|") -> List[Segment]:
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(
                    seg.length for seg in segs
                )
                words.append(
                    Segment(word, segments[i1].start, segments[i2 - 1].end, score)
                )
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words
