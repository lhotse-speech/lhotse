import re
from typing import List, NamedTuple, Sequence

import torch

from lhotse.supervision import AlignmentItem

from .base import FailedToAlign, ForcedAligner


class ASRForcedAligner(ForcedAligner):
    def __init__(
        self, bundle_name: str = "WAV2VEC2_ASR_BASE_960H", device: str = "cpu", **kwargs
    ):
        import torchaudio

        super().__init__(device=device)
        self.bundle_name = bundle_name
        self.bundle = getattr(torchaudio.pipelines, bundle_name)
        self.model = self.bundle.get_model().to(device)
        self.labels = self.bundle.get_labels()
        self.dictionary = {c: i for i, c in enumerate(self.labels)}
        self.discard_symbols = _make_discard_symbols_regex(self.labels)

    @property
    def sample_rate(self) -> int:
        return self.bundle.sample_rate

    def normalize_text(self, text: str, **kwargs) -> str:
        return _normalize_text(text, self.discard_symbols)

    def align(self, audio: torch.Tensor, transcript: str) -> List[AlignmentItem]:
        tokens = [self.dictionary[c] for c in transcript]

        with torch.inference_mode():
            emissions, _ = self.model(audio)
            emissions = torch.log_softmax(emissions, dim=-1)
        emission = emissions[0].cpu()

        trellis = _get_trellis(emission, tokens)
        path = _backtrack(trellis, emission, tokens)

        segments = _merge_repeats(path, transcript)

        word_segments = _merge_words(segments)

        # Ratio of number of samples to number of frames
        ratio = audio.size(1) / emission.size(0)
        return [
            AlignmentItem(
                symbol=ws.label,
                start=round(int(ratio * ws.start) / self.sample_rate, ndigits=8),
                duration=round(
                    int(ratio * (ws.end - ws.start)) / self.sample_rate,
                    ndigits=8,
                ),
                score=ws.score,
            )
            for ws in word_segments
        ]


def _make_discard_symbols_regex(labels: Sequence[str]) -> re.Pattern:
    return re.compile(rf"[^{' '.join(labels)}]")


def _normalize_text(text: str, discard_symbols: re.Pattern) -> str:
    from lhotse.utils import is_module_available

    assert is_module_available(
        "num2words"
    ), "To align with torchaudio, please run 'pip install num2words' for number to word normalization."
    from num2words import num2words

    text = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), text)
    return re.sub(discard_symbols, "", text.upper().replace(" ", "|"))


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
        raise FailedToAlign()
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
