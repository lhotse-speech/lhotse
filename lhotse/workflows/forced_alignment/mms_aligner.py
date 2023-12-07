import logging
import re
from typing import List, Optional, Tuple

import torch

from lhotse.supervision import AlignmentItem
from lhotse.utils import is_module_available

from .base import FailedToAlign, ForcedAligner

# Note: Korean _does_ use spaces, but not so straightforwardly as Indo-European languages.
# We still need to tokenize it into morphemes to get proper alignment.
LANGUAGES_WITHOUT_SPACES = ["zh", "ja", "ko", "th", "my", "km", "lo"]


class MMSForcedAligner(ForcedAligner):
    def __init__(self, device: str = "cpu", check_language: bool = True, **kwargs):
        super().__init__(device=device)

        if not is_module_available("uroman"):
            raise ImportError(
                "MMSForcedAligner requires the 'uroman' module to be installed. "
                "Please install it with 'pip install uroman-python'."
            )

        from torchaudio.pipelines import MMS_FA as bundle
        from uroman import uroman

        self.check_language = check_language
        self.bundle = bundle
        self.model = bundle.get_model().to(device)
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        self.discard_regex = re.compile(rf"[^{' '.join(bundle.get_labels())}]")
        self._uroman = uroman

    @property
    def sample_rate(self) -> int:
        return self.bundle.sample_rate

    def normalize_text(self, text: str, language=None) -> List[Tuple[str, str]]:
        # Split text into words (possibly with adjacent punctuation)
        language = _normalize_language(language)

        if language is None and self.check_language:
            logging.warning(
                "Language tag is not provided for the supervision text."
                "MMSForceAligner might not behave properly, especially for languages without spaces,"
                "such as Chinese or Japanese."
                "Provide `--dont-check-language` (for CLI) or `check_language=False` (for Python API) to suppress this warning"
            )

        orig_words = _word_tokenize(text, language)

        sep = _safe_separator(text)
        romanized_words = self._uroman(sep.join(orig_words), language=language).split(
            sep
        )
        romanized_l = [w.lower().replace("’", "'") for w in romanized_words]
        norm_words = [re.sub(self.discard_regex, "", w).strip() for w in romanized_l]
        word_pairs = list(zip(orig_words, norm_words))

        # Remove empty words and standalone dashes (aligner doesn't like them)
        return [(orig, norm) for orig, norm in word_pairs if norm != "" and norm != "-"]

    def align(
        self, audio: torch.Tensor, transcript: List[Tuple[str, str]]
    ) -> List[AlignmentItem]:
        try:
            with torch.inference_mode():
                emission, _ = self.model(audio)
                token_spans = self.aligner(
                    emission[0], self.tokenizer([p[1] for p in transcript])
                )
        except Exception as e:
            raise FailedToAlign from e

        ratio = audio.shape[1] / emission.shape[1] / self.sample_rate
        return [
            AlignmentItem(
                symbol=orig_word,
                start=round(ratio * t_spans[0].start, ndigits=8),
                duration=round(ratio * (t_spans[-1].end - t_spans[0].start), ndigits=8),
                score=_merge_score(t_spans),
            )
            for t_spans, (orig_word, _) in zip(token_spans, transcript)
        ]


def _merge_score(tspans):
    return sum(sp.score * (sp.end - sp.start) for sp in tspans) / sum(
        (sp.end - sp.start) for sp in tspans
    )


def _word_tokenize(text: str, language: Optional[str] = None) -> List[str]:
    """
    Add spaces between words for languages which do not have them.
    """

    # TODO: maybe add some simplistic auto-language detection?
    # many dataset recipes might not provide proper language tags to supervisions
    if language is None:
        return text.split()

    language = _normalize_language(language)
    if language not in LANGUAGES_WITHOUT_SPACES:
        return text.split()

    if language == "zh":
        if not is_module_available("jieba"):
            raise ImportError(
                "MMSForcedAligner requires the 'jieba' module to be installed to align Chinese text."
                "Please install it with 'pip install jieba'."
            )

        import jieba

        return jieba.lcut(text)

    elif language == "ja":
        if not is_module_available("nagisa"):
            raise ImportError(
                "MMSForcedAligner requires the 'nagisa' module to be installed to align Japanese text."
                "Please install it with 'pip install nagisa'."
            )

        import nagisa

        return nagisa.tagging(text).words

    elif language == "ko":
        if not is_module_available("kss"):
            raise ImportError(
                "MMSForcedAligner requires the 'kss' module to be installed to align Korean text."
                "Please install it with 'pip install kss'."
            )

        import kss

        return kss.split_morphemes(text, return_pos=False)

    elif language == "th":
        if not is_module_available("attacut"):
            raise ImportError(
                "MMSForcedAligner requires the 'attacut' module to be installed to align Thai text."
                "Please install it with 'pip install attacut'."
            )

        import attacut

        return attacut.tokenize(text)

    elif language == "my":
        if not is_module_available("pyidaungsu"):
            raise ImportError(
                "MMSForcedAligner requires the 'pyidaungsu' module to be installed to align Burmese text."
                "Please install it with 'pip install pyidaungsu'."
            )

        import pyidaungsu as pds

        return pds.tokenize(text, form="word")

    elif language == "km":
        if not is_module_available("khmernltk"):
            raise ImportError(
                "MMSForcedAligner requires the 'khmernltk' module to be installed to align Khmer text."
                "Please install it with 'pip install khmer-nltk'."
            )

        import khmernltk

        return khmernltk.word_tokenize(text)

    else:
        logging.warning(
            f"Language `{language}` does not have spaces between words, "
            f"but proper word tokenization for it is not supported yet."
            f"Proceeding with character-level alignment."
        )
        return " ".join(text)


def _normalize_language(language: Optional[str]) -> Optional[str]:
    """
    Returns top-level 2-letters language code for any language code
    or language name in English.
    """
    if language is None:
        return None

    from langcodes import Language, tag_parser

    try:
        # Try to parse the language tag first
        return Language.get(language).language
    except tag_parser.LanguageTagError:
        # If it fails, try to parse the language name. But we need the `language_data`
        # package to do that.
        if not is_module_available("language_data"):
            raise ImportError(
                f"Language `{language}` is not a valid language tag, and MMSForcedAligner requires"
                "the 'language_data' module to be installed to parse language names."
                "Please install it with 'pip install language_data'."
            )

        return Language.find(language).language


def _safe_separator(text):
    """
    Returns a separator that is not present in the text.
    """
    special_symbols = "#$%^&~_"
    i = 0
    while special_symbols[i] in text and i < len(special_symbols):
        i += 1

    # better use space than just fail
    return special_symbols[i] if i < len(special_symbols) else " "
