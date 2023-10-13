import re
from typing import List, NamedTuple, Sequence

import torch
import torchaudio

from lhotse.supervision import AlignmentItem
from lhotse.utils import is_module_available

from .base import FailedToAlign, ForcedAligner


class MMSForcedAligner(ForcedAligner):
    def __init__(self, bundle_name: str, device: str = "cpu"):
        super().__init__(device=device)
        assert bundle_name == "MMS_FA", "MMSForcedAligner only supports MMS_FA bundle"

        if not is_module_available("uroman"):
            raise ImportError(
                "MMSForcedAligner requires the 'uroman' module to be installed. "
                "Please install it with 'pip install uroman-python'."
            )

        from torchaudio.pipelines import MMS_FA as bundle
        from uroman import uroman

        self.bundle = bundle
        self.model = bundle.get_model().to(device)
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        labels = list(set(bundle.get_labels()) - set("-"))
        self.discard_regex = re.compile(rf"[^{' '.join(labels)}]")
        self._uroman = uroman

    @property
    def sample_rate(self) -> int:
        return self.bundle.sample_rate

    def normalize_text(self, text: str, language=None) -> List[str]:
        romanized = self._uroman(text, language=language)
        romanized_l = romanized.lower().replace("’", "'")
        romanized_no_punct = re.sub(self.discard_regex, "", romanized_l)
        return romanized_no_punct.strip().split()

    def align(self, audio: torch.Tensor, transcript: List[str]) -> List[AlignmentItem]:
        with torch.inference_mode():
            emission, _ = self.model(audio)
            token_spans = self.aligner(emission[0], self.tokenizer(transcript))

        ratio = audio.shape[1] / emission.shape[1] / self.sample_rate
        return [
            AlignmentItem(
                symbol=word,
                start=round(ratio * t_spans[0].start, ndigits=8),
                duration=round(ratio * (t_spans[-1].end - t_spans[0].start), ndigits=8),
                score=_merge_score(t_spans),
            )
            for t_spans, word in zip(token_spans, transcript)
        ]


def _merge_score(tspans):
    return sum(sp.score * (sp.end - sp.start) for sp in tspans) / sum(
        (sp.end - sp.start) for sp in tspans
    )
