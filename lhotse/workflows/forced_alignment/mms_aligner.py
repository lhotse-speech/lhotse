import logging
import re
from typing import List, Optional

import torch
import torchaudio

from lhotse.supervision import AlignmentItem
from lhotse.utils import is_module_available

from .base import FailedToAlign, ForcedAligner

# Note: Korean _does_ use spaces, but not so straightforwardly as Indo-European languages.
# We still need to tokenize it into morphemes to get proper alignment.
LANGUAGES_WITHOUT_SPACES = ["zh", "ja", "ko", "th", "my", "km", "lo"]


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
        self.discard_regex = re.compile(rf"[^{' '.join(bundle.get_labels())}]")
        self._uroman = uroman

    @property
    def sample_rate(self) -> int:
        return self.bundle.sample_rate

    def normalize_text(self, text: str, language=None) -> List[str]:
        # Add spaces between words for languages which do not have them
        text = _spacify(text, language)

        romanized = self._uroman(text, language=language)
        romanized_l = romanized.lower().replace("’", "'")
        romanized_no_punct = re.sub(self.discard_regex, "", romanized_l)
        words = romanized_no_punct.strip().split()

        # Remove standalone dashes - aligner doesn't like them
        return [w for w in words if w != "-"]

    def align(self, audio: torch.Tensor, transcript: List[str]) -> List[AlignmentItem]:
        try:
            with torch.inference_mode():
                emission, _ = self.model(audio)
                token_spans = self.aligner(emission[0], self.tokenizer(transcript))
        except Exception as e:
            raise FailedToAlign from e

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


def _spacify(text: str, language: Optional[str] = None) -> str:
    """
    Add spaces between words for languages which do not have them.
    """

    # TODO: maybe add some simplistic auto-language detection?
    # many dataset recipes might not provide proper language tags to supervisions
    if language is None:
        return text

    language = _normalize_language(language)
    if language not in LANGUAGES_WITHOUT_SPACES:
        return text

    if language == "zh":
        if not is_module_available("jieba"):
            raise ImportError(
                "MMSForcedAligner requires the 'jieba' module to be installed to align Chinese text."
                "Please install it with 'pip install jieba'."
            )

        import jieba

        return " ".join(jieba.cut(text))

    elif language == "ja":
        if not is_module_available("nagisa"):
            raise ImportError(
                "MMSForcedAligner requires the 'nagisa' module to be installed to align Japanese text."
                "Please install it with 'pip install nagisa'."
            )

        import nagisa

        return " ".join(nagisa.tagging(text).words)

    elif language == "ko":
        if not is_module_available("kss"):
            raise ImportError(
                "MMSForcedAligner requires the 'kss' module to be installed to align Korean text."
                "Please install it with 'pip install kss'."
            )

        import kss

        return " ".join(kss.split_morphemes(text, return_pos=False))

    elif language == "th":
        # `pythainlp` is alive and much better, but it is a huge package bloated with dependencies
        if not is_module_available("tltk"):
            raise ImportError(
                "MMSForcedAligner requires the 'tltk' module to be installed to align Thai text."
                "Please install it with 'pip install tltk'."
            )

        from tltk import nlp

        pieces = nlp.pos_tag(text)
        words = [
            word if word != "<s/>" else " " for piece in pieces for word, _ in piece
        ]
        return " ".join(words)

    else:
        logging.warning(
            f"Language `{language}` does not have spaces between words, "
            f"but proper word tokenization for it is not supported yet."
            f"Proceeding with character-level alignment."
        )
        return " ".join(text)


def _normalize_language(language: str) -> str:
    """
    Returns top-level 2-letters language code for any language code
    or language name in English.
    """
    from langcodes import Language, tag_parser

    try:
        # Try to parse the language tag first
        return Language.get(language).language
    except tag_parser.LanguageTagError:
        # If it fails, try to parse the language name.
        return Language.find(language).language
