from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

from lhotse import CutSet, FeatureSet, load_manifest
from lhotse.audio import RecordingSet
from lhotse.supervision import SupervisionSet
from lhotse.utils import Pathlike

DEFAULT_DETECTED_MANIFEST_TYPES = ("recordings", "supervisions")

TYPES_TO_CLASSES = {
    "recordings": RecordingSet,
    "supervisions": SupervisionSet,
    "features": FeatureSet,
    "cuts": CutSet,
}


def read_manifests_if_cached(
    dataset_parts: Optional[Sequence[str]],
    output_dir: Optional[Pathlike],
    prefix: str = "",
    suffix: Optional[str] = "jsonl.gz",
    types: Iterable[str] = DEFAULT_DETECTED_MANIFEST_TYPES,
    lazy: bool = False,
) -> Optional[Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    Loads manifests from the disk, or a subset of them if only some exist.
    The manifests are searched for using the pattern ``output_dir / f'{prefix}_{manifest}_{part}.json'``,
    where `manifest` is one of ``["recordings", "supervisions"]`` and ``part`` is specified in ``dataset_parts``.
    This function is intended to speedup data preparation if it has already been done before.

    :param dataset_parts: Names of dataset pieces, e.g. in LibriSpeech: ``["test-clean", "dev-clean", ...]``.
    :param output_dir: Where to look for the files.
    :param prefix: Optional common prefix for the manifest files (underscore is automatically added).
    :param suffix: Optional common suffix for the manifest files ("json" by default).
    :param types: Which types of manifests are searched for (default: 'recordings' and 'supervisions').
    :return: A dict with manifest (``d[dataset_part]['recording'|'manifest']``) or ``None``.
    """
    if isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]
    if output_dir is None:
        return None
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"
    if suffix.startswith("."):
        suffix = suffix[1:]
    if lazy and not suffix.startswith("jsonl"):
        raise ValueError(
            f"Only JSONL manifests can be opened lazily (got suffix: '{suffix}')"
        )
    manifests = defaultdict(dict)
    output_dir = Path(output_dir)
    for part in dataset_parts:
        for manifest in types:
            path = output_dir / f"{prefix}{manifest}_{part}.{suffix}"
            if not path.is_file():
                continue
            if lazy:
                manifests[part][manifest] = TYPES_TO_CLASSES[manifest].from_jsonl_lazy(
                    path
                )
            else:
                manifests[part][manifest] = load_manifest(path)
    return dict(manifests)


def manifests_exist(
    part: str,
    output_dir: Optional[Pathlike],
    types: Iterable[str] = DEFAULT_DETECTED_MANIFEST_TYPES,
    prefix: str = "",
    suffix: str = "jsonl.gz",
) -> bool:
    if output_dir is None:
        return False
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"
    if suffix.startswith("."):
        suffix = suffix[1:]
    output_dir = Path(output_dir)
    for name in types:
        path = output_dir / f"{prefix}{name}_{part}.{suffix}"
        if not path.is_file():
            return False
    return True


def normalize_text_alimeeting(text: str, normalize: str = "m2met") -> str:
    """
    Text normalization similar to M2MeT challenge baseline.
    See: https://github.com/yufan-aslp/AliMeeting/blob/main/asr/local/text_normalize.pl
    """
    if normalize == "none":
        return text
    elif normalize == "m2met":
        import re

        text = text.replace("<sil>", "")
        text = text.replace("<%>", "")
        text = text.replace("<->", "")
        text = text.replace("<$>", "")
        text = text.replace("<#>", "")
        text = text.replace("<_>", "")
        text = text.replace("<space>", "")
        text = text.replace("`", "")
        text = text.replace("&", "")
        text = text.replace(",", "")
        if re.search("[a-zA-Z]", text):
            text = text.upper()
        text = text.replace("Ａ", "A")
        text = text.replace("ａ", "A")
        text = text.replace("ｂ", "B")
        text = text.replace("ｃ", "C")
        text = text.replace("ｋ", "K")
        text = text.replace("ｔ", "T")
        text = text.replace("，", "")
        text = text.replace("丶", "")
        text = text.replace("。", "")
        text = text.replace("、", "")
        text = text.replace("？", "")
        return text


def normalize_text_ami(text: str, normalize: str = "upper") -> str:
    """
    Text normalization similar to Kaldi's AMI recipe.
    """
    if normalize == "none":
        return text
    elif normalize == "upper":
        return text.upper()
    elif normalize == "kaldi":
        # Kaldi style text normalization
        import re

        # convert text to uppercase
        text = text.upper()
        # remove punctuations
        text = re.sub(r"[^A-Z0-9']+", " ", text)
        # remove multiple spaces
        text = re.sub(r"\s+", " ", text)
        # apply few exception for dashed phrases, Mm-Hmm, Uh-Huh, OK etc. those are frequent in AMI
        # and will be added to dictionary
        text = re.sub(r"MM HMM", "MM-HMM", text)
        text = re.sub(r"UH HUH", "UH-HUH", text)
        text = re.sub(r"(\b)O K(\b)", r"\g<1>OK\g<2>", text)
        text = re.sub(r"(\b)O_K(\b)", r"\g<1>OK\g<2>", text)
        return text.strip()


def normalize_text_chime6(text: str, normalize: str = "upper") -> str:
    """
    Text normalization similar to Kaldi's CHiME-6 recipe.
    """
    if normalize == "none":
        return text
    elif normalize == "upper":
        return text.upper()
    elif normalize == "kaldi":
        # Kaldi style text normalization
        import re

        if "[redacted]" in text:
            return ""

        # convert text to lowercase
        text = text.lower()
        # remove punctuations: " . ? , : ; !
        text = re.sub(r"[.?,:;!]", "", text)
        # remove multiple spaces
        text = re.sub(r"\s+", " ", text)
        # replace multiple consecutive [inaudible] with a single one
        text = re.sub(r"\[inaudible[- 0-9]*\]", "[inaudible]", text)
        # replace stranded dash with space
        text = re.sub(r" - ", " ", text)
        # replace mm- with mm
        text = re.sub(r"mm-", "mm", text)
        return text


def normalize_text_tedlium(text: str, normalize: str = "upper") -> str:
    """
    Text normalization similar to Kaldi's TEDLIUM-3 recipe.
    """
    if normalize == "none":
        return text
    elif normalize == "upper":
        return text.upper()
    elif normalize == "kaldi":
        # Kaldi style text normalization
        import re

        # remove tokens such as "[NOISE]"
        text = re.sub(r"\[[^\]]+\]", "", text)
        # remove "<unk>"
        text = re.sub(r"<unk>", "", text)
        # join suffixes with words, e.g. they 're -> they're
        text = re.sub(r"(\w+) '(\w+)", r"\1'\2", text)
        # join dangling "'" with next word, e.g. ' 60s -> '60s, ' cause -> 'cause
        text = re.sub(r"' (\w+)", r"'\1", text)

        return text.strip()


class TimeFormatConverter:
    @staticmethod
    def hms_to_seconds(time: str) -> float:
        """Converts time in HH:MM:SS.mmm format to seconds"""
        h, m, s = time.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    @staticmethod
    def seconds_to_hms(time: float) -> str:
        """Converts time in seconds to HH:MM:SS.mmm format"""
        h = int(time // 3600)
        m = int((time % 3600) // 60)
        s = time % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"
