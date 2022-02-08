import contextlib
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Type

from lhotse import AudioSource
from lhotse.array import Array, TemporalArray
from lhotse.audio import Recording, RecordingSet
from lhotse.cut import CutSet, MonoCut
from lhotse.features import FeatureSet, Features
from lhotse.manipulation import Manifest
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import fastcopy


@contextlib.contextmanager
def as_lazy(manifest):
    """
    Context manager for converting eager manifests to lazy manifests.
    Intended for testing.
    """
    with NamedTemporaryFile(suffix=".jsonl.gz") as f:
        manifest.to_file(f.name)
        f.flush()
        yield type(manifest).from_jsonl_lazy(f.name)


# noinspection PyPep8Naming
def DummyManifest(type_: Type, *, begin_id: int, end_id: int) -> Manifest:
    if type_ == RecordingSet:
        return RecordingSet.from_recordings(
            dummy_recording(idx) for idx in range(begin_id, end_id)
        )
    if type_ == SupervisionSet:
        return SupervisionSet.from_segments(
            dummy_supervision(idx) for idx in range(begin_id, end_id)
        )
    if type_ == FeatureSet:
        # noinspection PyTypeChecker
        return FeatureSet.from_features(
            dummy_features(idx) for idx in range(begin_id, end_id)
        )
    if type_ == CutSet:
        # noinspection PyTypeChecker
        return CutSet.from_cuts(
            dummy_cut(idx, supervisions=[dummy_supervision(idx)])
            for idx in range(begin_id, end_id)
        )


def dummy_recording(unique_id: int, duration: float = 1.0) -> Recording:
    return Recording(
        id=f"dummy-recording-{unique_id:04d}",
        sources=[
            AudioSource(type="command", channels=[0], source='echo "dummy waveform"')
        ],
        sampling_rate=16000,
        num_samples=16000,
        duration=duration,
    )


def dummy_alignment(
    text: str = "irrelevant", start: float = 0.0, duration: float = 1.0
) -> AlignmentItem:
    subwords = [
        text[i : i + 3] for i in range(0, len(text), 3)
    ]  # Create subwords of 3 chars
    dur = duration / len(subwords)
    alignment = [
        AlignmentItem(symbol=sub, start=start + i * dur, duration=dur)
        for i, sub in enumerate(subwords)
    ]
    return {"subword": alignment}


def dummy_supervision(
    unique_id: int,
    start: float = 0.0,
    duration: float = 1.0,
    text: str = "irrelevant",
    alignment: Optional[Dict[str, List[AlignmentItem]]] = dummy_alignment(),
) -> SupervisionSegment:
    return SupervisionSegment(
        id=f"dummy-segment-{unique_id:04d}",
        recording_id=f"dummy-recording-{unique_id:04d}",
        start=start,
        duration=duration,
        text=text,
        speaker="irrelevant",
        language="irrelevant",
        gender="irrelevant",
        custom={"custom_field": "irrelevant"},
        alignment=alignment,
    )


def dummy_features(
    unique_id: int, start: float = 0.0, duration: float = 1.0
) -> Features:
    return Features(
        recording_id=f"dummy-recording-{unique_id:04d}",
        channels=0,
        start=start,
        duration=duration,
        type="fbank",
        num_frames=100,
        num_features=23,
        frame_shift=0.01,
        sampling_rate=16000,
        storage_type="lilcom_files",
        storage_path="test/fixtures/dummy_feats/storage",
        storage_key="dbf9a0ec-f79d-4eb8-ae83-143a6d5de64d.llc",
    )


def dummy_temporal_array(start: float = 0.0) -> TemporalArray:
    return TemporalArray(
        array=Array(
            storage_type="lilcom_files",
            storage_path="test/fixtures/dummy_feats/storage",
            storage_key="dbf9a0ec-f79d-4eb8-ae83-143a6d5de64d.llc",
            shape=[100, 23],
        ),
        temporal_dim=0,
        start=start,
        frame_shift=0.01,
    )


def dummy_cut(
    unique_id: int,
    start: float = 0.0,
    duration: float = 1.0,
    recording: Recording = None,
    features: Features = None,
    supervisions=None,
):
    return MonoCut(
        id=f"dummy-cut-{unique_id:04d}",
        start=start,
        duration=duration,
        channel=0,
        recording=recording if recording else dummy_recording(unique_id),
        features=features if features else dummy_features(unique_id),
        supervisions=supervisions if supervisions is not None else [],
    )


def remove_spaces_from_segment_text(segment):
    if segment.text is None:
        return segment
    return fastcopy(segment, text=segment.text.replace(" ", ""))
