"""
╔═════════════════════════════════════════════╗
║ Export/Import CutSet to HuggingFace Dataset ║
╚═════════════════════════════════════════════╝
"""
from hashlib import md5
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from lhotse import Recording, SupervisionSegment
from lhotse.cut import CutSet, MonoCut
from lhotse.utils import Pathlike, is_module_available


def contains_only_mono_cuts(cutset: CutSet) -> bool:
    return all(isinstance(cut, MonoCut) for cut in cutset)


def has_one_supervision_per_cut(cutset: CutSet) -> bool:
    return all(len(cut.supervisions) == 1 for cut in cutset)


def has_one_audio_source(cutset: CutSet) -> bool:
    return all(len(cut.recording.sources) == 1 for cut in cutset)


def convert_cuts_info_to_hf(cutset: CutSet) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Converts the cut information into a dictionary compatible with HuggingFace datasets format.

    :param cutset: A CutSet object.
    :return: A tuple where the first element is a dictionary
        representing the cut attributes and the second element is a dictionary describing the
        format of the HuggingFace dataset.
    """
    from datasets import Audio, Value

    cut_info = {
        "id": [cut.id for cut in cutset],
        "audio": [cut.recording.sources[0].source for cut in cutset],
        "duration": [cut.duration for cut in cutset],
        "num_channels": [len(cut.recording.channel_ids) for cut in cutset],
    }
    cut_info_description = {
        "id": Value("string"),
        "audio": Audio(mono=False),
        "duration": Value("float"),
        "num_channels": Value("uint16"),
    }
    return cut_info, cut_info_description


def convert_supervisions_info_to_hf(
    cutset: CutSet,
    exclude_attributes: Optional[Union[List[str], Set[str]]] = None,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Converts cut supervisions into a dictionary compatible with HuggingFace datasets format.

    :param cutset: A CutSet object.
    :param exclude_attributes: A list|set of attributes to exclude from the supervisions dicts.
    :return: A tuple where the first element is a dictionary
        representing the cut attributes and the second element is a dictionary describing the
        format of the HuggingFace dataset.
    """

    from datasets import Features, Sequence, Value

    has_speaker = any(
        (
            hasattr(cut.supervisions[0], "speaker")
            and cut.supervisions[0].speaker is not None
        )
        for cut in cutset
    )
    has_language = any(
        (
            hasattr(cut.supervisions[0], "language")
            and cut.supervisions[0].language is not None
        )
        for cut in cutset
    )
    alignment_types = [
        s.alignment.keys()
        for c in cutset
        for s in c.supervisions
        if s.alignment is not None
    ]
    alignment_types = set([item for sublist in alignment_types for item in sublist])

    sup_dicts = []
    for c in cutset:
        cut_sup_dicts = []
        for s in c.supervisions:
            sup_dict = {
                "text": s.text,
            }

            if exclude_attributes is None or "start" not in exclude_attributes:
                sup_dict["start"] = s.start

            if exclude_attributes is None or "end" not in exclude_attributes:
                sup_dict["end"] = s.end

            if exclude_attributes is None or "channel" not in exclude_attributes:
                if isinstance(s.channel, list):
                    sup_dict["channel"] = ",".join(map(str, s.channel))
                else:
                    sup_dict["channel"] = str(s.channel)

            if has_speaker and (
                exclude_attributes is None or "speaker" not in exclude_attributes
            ):
                sup_dict["speaker"] = str(s.speaker)

            if has_language and (
                exclude_attributes is None or "language" not in exclude_attributes
            ):
                sup_dict["language"] = str(s.language)

            if alignment_types and (
                exclude_attributes is None or "alignments" not in exclude_attributes
            ):
                alignments = {}
                for alignment_type in alignment_types:
                    alignments[alignment_type + "_alignment"] = list(
                        map(
                            lambda item: {
                                "symbol": item.symbol,
                                "start": item.start,
                                "end": item.end,
                            },
                            s.alignment[alignment_type],
                        )
                    )

                sup_dict = {**sup_dict, **alignments}

            cut_sup_dicts.append(sup_dict)
        sup_dicts.append(cut_sup_dicts)

    sup_dicts_info = {"text": Value("string")}

    if exclude_attributes is None or "start" not in exclude_attributes:
        sup_dicts_info["start"] = Value("float")

    if exclude_attributes is None or "end" not in exclude_attributes:
        sup_dicts_info["end"] = Value("float")

    if exclude_attributes is None or "channel" not in exclude_attributes:
        sup_dicts_info["channel"] = Value("string")

    if has_speaker and (
        exclude_attributes is None or "speaker" not in exclude_attributes
    ):
        sup_dicts_info["speaker"] = Value("string")

    if has_language and (
        exclude_attributes is None or "language" not in exclude_attributes
    ):
        sup_dicts_info["language"] = Value("string")

    if alignment_types and (
        exclude_attributes is None or "alignments" not in exclude_attributes
    ):
        alignment_info = {
            "symbol": Value("string"),
            "start": Value("float"),
            "end": Value("float"),
        }
        for alignment_type in alignment_types:
            sup_dicts_info[alignment_type + "_alignment"] = Sequence(
                Features(**alignment_info)
            )

    return sup_dicts, sup_dicts_info


def lod_to_dol(lod: List[Dict[str, Any]]) -> Dict[str, List]:
    """
    Converts List of Dicts to Dict of Lists.
    """
    return {k: [d[k] for d in lod] for k in lod[0].keys()}


def export_cuts_to_hf(cutset: CutSet):
    """
    Converts a CutSet to a HuggingFace Dataset. Currently, only MonoCut with one recording source is supported.
    Other cut types will be supported in the future.

    Currently, two formats are supported:
        1. If each cut has one supervision (e.g. LibriSpeech), each cut is represented as a single row (entry)
           in the HuggingFace dataset with all the supervision information stored along the cut information.
           The final HuggingFace dataset format is:
               ╔═══════════════════╦═══════════════════════════════╗
               ║      Feature      ║            Type               ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║        id         ║ Value(dtype='string')         ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║      audio        ║ Audio()                       ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║     duration      ║ Value(dtype='float32')        ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║   num_channels    ║ Value(dtype='uint16')         ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║       text        ║ Value(dtype='string')         ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║     speaker       ║ Value(dtype='string')         ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║     language      ║ Value(dtype='string')         ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║   {x}_alignment   ║ Sequence(Alignment)           ║
               ╚═══════════════════╩═══════════════════════════════╝
           where x stands for the alignment type (commonly used: "word", "phoneme").

           Alignment is represented as:
               ╔═══════════════════╦═══════════════════════════════╗
               ║      Feature      ║            Type               ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║      symbol       ║ Value(dtype='string')         ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║       start       ║ Value(dtype='float32')        ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║        end        ║ Value(dtype='float32')        ║
               ╚═══════════════════╩═══════════════════════════════╝


        2. If each cut has multiple supervisions (e.g. AMI), each cut is represented as a single row (entry)
           while all the supervisions are stored in a separate list of dictionaries under the 'segments' key.
           The final HuggingFace dataset format is:
               ╔══════════════╦════════════════════════════════════╗
               ║   Feature    ║                 Type               ║
               ╠══════════════╬════════════════════════════════════╣
               ║      id      ║ Value(dtype='string')              ║
               ╠══════════════╬════════════════════════════════════╣
               ║    audio     ║ Audio()                            ║
               ╠══════════════╬════════════════════════════════════╣
               ║   duration   ║ Value(dtype='float32')             ║
               ╠══════════════╬════════════════════════════════════╣
               ║ num_channels ║ Value(dtype='uint16')              ║
               ╠══════════════╬════════════════════════════════════╣
               ║   segments   ║ Sequence(Segment)                  ║
               ╚══════════════╩════════════════════════════════════╝
           where one Segment is represented as:
               ╔═══════════════════╦═══════════════════════════════╗
               ║      Feature      ║            Type               ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║        text       ║ Value(dtype='string')         ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║       start       ║ Value(dtype='float32')        ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║        end        ║ Value(dtype='float32')        ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║      channel      ║ Value(dtype='string')         ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║      speaker      ║ Value(dtype='string')         ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║      language     ║ Value(dtype='string')         ║
               ╠═══════════════════╬═══════════════════════════════╣
               ║   {x}_alignment   ║ Sequence(Alignment)           ║
               ╚═══════════════════╩═══════════════════════════════╝

    :param cutset: A CutSet object.
    :return: A HuggingFace Dataset.
    """

    assert has_one_audio_source(
        cutset
    ), "Only CutSets with one audio source per cut are supported. MultiSource cuts coming soon."

    if not is_module_available("datasets"):
        raise ImportError(
            "Please install the 'datasets' package (pip install datasets)."
        )
    from datasets import Dataset, Features, Sequence

    # We don't need start and end attribute if we have only one supervision/segment per cut,
    #  as start=0 and end=duration.
    cut_info, cut_info_description = convert_cuts_info_to_hf(cutset)
    sup_dicts, sup_dicts_info = convert_supervisions_info_to_hf(
        cutset,
        exclude_attributes={"start", "end", "channel"}
        if has_one_supervision_per_cut(cutset)
        else None,
    )

    if has_one_supervision_per_cut(cutset):
        dataset_dict = {
            **cut_info,
            **lod_to_dol([x[0] for x in sup_dicts]),
        }
        dataset_info = Features(
            **cut_info_description,
            **sup_dicts_info,
        )
    else:
        dataset_dict = {
            **cut_info,
            "segments": sup_dicts,
        }
        dataset_info = Features(
            segments=Sequence(Features(**sup_dicts_info)),
            **cut_info_description,
        )

    return Dataset.from_dict(dataset_dict, features=dataset_info)


class LazyHFDatasetIterator:
    """
    Thin wrapper on top of HF datasets objects that allows to interact with them through a Lhotse CutSet.
    It can be initialized with an existing HF dataset, or args/kwargs passed on to ``datasets.load_dataset()``.

    Use ``audio_key``, ``text_key``, ``lang_key`` and ``gender_key`` options to indicate which keys in dict examples
    returned from HF Dataset should be looked up for audio, transcript, language, and gender respectively.
    The remaining keys in HF dataset examples will be stored inside ``cut.custom`` dictionary.

    Example with existing HF dataset::

        >>> import datasets
        ... dataset = datasets.load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")
        ... dataset = dataset.map(some_transform)
        ... cuts_it = LazyHFDatasetIterator(dataset)
        ... for cut in cuts_it:
        ...     pass

    Example providing HF dataset init args/kwargs::

        >>> import datasets
        ... cuts_it = LazyHFDatasetIterator("mozilla-foundation/common_voice_11_0", "hi", split="test")
        ... for cut in cuts_it:
        ...     pass

    """

    def __init__(
        self,
        *dataset_args,
        audio_key: str = "audio",
        text_key: str = "sentence",
        lang_key: str = "language",
        gender_key: str = "gender",
        **dataset_kwargs
    ):
        assert is_module_available("datasets")
        self.audio_key = audio_key
        self.text_key = text_key
        self.lang_key = lang_key
        self.gender_key = gender_key
        self.dataset_args = dataset_args
        self.dataset_kwargs = dataset_kwargs

    def __iter__(self):
        from datasets import (
            Audio,
            Dataset,
            DatasetDict,
            IterableDataset,
            IterableDatasetDict,
            load_dataset,
        )

        if len(self.dataset_args) == 1 and isinstance(
            self.dataset_args[0],
            (Dataset, IterableDataset, DatasetDict, IterableDatasetDict),
        ):
            dataset = self.dataset_args[0]
        else:
            dataset = load_dataset(*self.dataset_args, **self.dataset_kwargs)
        dataset = dataset.cast_column(self.audio_key, Audio(decode=False))
        for item in dataset:
            audio_data = item.pop(self.audio_key)
            recording = Recording.from_bytes(
                audio_data["bytes"], recording_id=md5(audio_data["bytes"]).hexdigest()
            )
            supervision = SupervisionSegment(
                id=recording.id,
                recording_id=recording.id,
                start=0.0,
                duration=recording.duration,
                text=item.pop(self.text_key, None),
                language=item.pop(self.lang_key, None),
                gender=item.pop(self.gender_key, None),
            )
            cut = recording.to_cut()
            cut.supervisions = [supervision]
            cut.custom = item
            yield cut
