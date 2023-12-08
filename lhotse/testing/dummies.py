import contextlib
from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Type, Union

import numpy as np
import torch

from lhotse.array import Array, TemporalArray
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.cut import CutSet, MonoCut, MultiCut
from lhotse.features import Features, FeatureSet
from lhotse.features.io import MemoryRawWriter
from lhotse.manipulation import Manifest
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import compute_num_frames, compute_num_samples, fastcopy


@contextlib.contextmanager
def as_lazy(manifest, suffix=".jsonl.gz"):
    """
    Context manager for converting eager manifests to lazy manifests.
    Intended for testing.
    """
    with NamedTemporaryFile(suffix=suffix) as f:
        manifest.to_file(f.name)
        f.flush()
        yield type(manifest).from_jsonl_lazy(f.name)


# noinspection PyPep8Naming
def DummyManifest(
    type_: Type, *, begin_id: int, end_id: int, with_data: bool = False
) -> Manifest:
    if type_ == RecordingSet:
        return RecordingSet.from_recordings(
            dummy_recording(idx, with_data=with_data) for idx in range(begin_id, end_id)
        )
    if type_ == SupervisionSet:
        return SupervisionSet.from_segments(
            dummy_supervision(idx) for idx in range(begin_id, end_id)
        )
    if type_ == FeatureSet:
        # noinspection PyTypeChecker
        return FeatureSet.from_features(
            dummy_features(idx, with_data=with_data) for idx in range(begin_id, end_id)
        )
    if type_ == CutSet:
        # noinspection PyTypeChecker
        return CutSet.from_cuts(
            dummy_cut(idx, supervisions=[dummy_supervision(idx)], with_data=with_data)
            for idx in range(begin_id, end_id)
        )


def dummy_recording(
    unique_id: int,
    duration: float = 1.0,
    sampling_rate: int = 16000,
    with_data: bool = False,
) -> Recording:
    num_samples = compute_num_samples(duration, sampling_rate)
    return Recording(
        id=f"dummy-recording-{unique_id:04d}",
        sources=[
            dummy_audio_source(
                sampling_rate=sampling_rate,
                num_samples=num_samples,
                with_data=with_data,
            )
        ],
        sampling_rate=sampling_rate,
        num_samples=num_samples,
        duration=duration,
    )


def dummy_audio_source(
    num_samples: int = 16000,
    sampling_rate: int = 16000,
    channels: Optional[List[int]] = None,
    with_data: bool = False,
) -> AudioSource:
    if channels is None:
        channels = [0]
    if not with_data:
        return AudioSource(
            type="command", channels=channels, source='echo "dummy waveform"'
        )
    else:
        import torchaudio

        # 1kHz sine wave
        data = torch.sin(2 * np.pi * 1000 * torch.arange(num_samples)).unsqueeze(0)
        if len(channels) > 1:
            data = data.expand(len(channels), -1)
        binary_data = BytesIO()
        torchaudio.save(binary_data, data, sample_rate=sampling_rate, format="wav")
        binary_data.seek(0)
        return AudioSource(
            type="memory", channels=channels, source=binary_data.getvalue()
        )


def dummy_multi_channel_recording(
    unique_id: int,
    duration: float = 1.0,
    channel_ids: Optional[List[int]] = None,
    source_per_channel: bool = False,
    with_data: bool = False,
) -> Recording:
    if channel_ids is None:
        channel_ids = [0, 1]
    if source_per_channel:
        sources = [dummy_audio_source(channels=channel_ids, with_data=with_data)]
    else:
        sources = [
            dummy_audio_source(channels=[i], with_data=with_data) for i in channel_ids
        ]
    return Recording(
        id=f"dummy-multi-channel-recording-{unique_id:04d}",
        sources=sources,
        sampling_rate=16000,
        num_samples=16000,
        duration=duration,
    )


def dummy_alignment(
    text: str = "irrelevant", start: float = 0.0, duration: float = 1.0
) -> Dict[str, List[AlignmentItem]]:
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
    channel: Union[int, List[int]] = 0,
    text: str = "irrelevant",
    alignment: Optional[Dict[str, List[AlignmentItem]]] = dummy_alignment(),
) -> SupervisionSegment:
    return SupervisionSegment(
        id=f"dummy-segment-{unique_id:04d}",
        recording_id=f"dummy-recording-{unique_id:04d}",
        start=start,
        duration=duration,
        channel=channel,
        text=text,
        speaker="irrelevant",
        language="irrelevant",
        gender="irrelevant",
        custom={"custom_field": "irrelevant"},
        alignment=alignment,
    )


def dummy_features(
    unique_id: int, start: float = 0.0, duration: float = 1.0, with_data: bool = False
) -> Features:
    # Note: with_data is ignored as this always has real data attached
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


def dummy_in_memory_features(
    unique_id: int,
    start: float = 0.0,
    duration: float = 1.0,
    sampling_rate: int = 16000,
    frame_shift: float = 0.01,
) -> Features:
    num_frames = compute_num_frames(duration, frame_shift, sampling_rate)
    num_features = 23
    data = np.random.rand(num_frames, num_features).astype(np.float32)
    bindata = MemoryRawWriter().write("dummy-features", data)
    return Features(
        recording_id=f"dummy-recording-{unique_id:04d}",
        channels=0,
        start=start,
        duration=duration,
        type="fbank",
        num_frames=num_frames,
        num_features=num_features,
        frame_shift=frame_shift,
        sampling_rate=sampling_rate,
        storage_type=MemoryRawWriter.name,
        storage_path="",
        storage_key=bindata,
    )


def dummy_multi_channel_features(
    unique_id: int,
    start: float = 0.0,
    duration: float = 1.0,
    channels: Optional[List[int]] = None,
) -> Features:
    if channels is None:
        channels = [0, 1]
    return Features(
        recording_id=f"dummy-multi-channel-recording-{unique_id:04d}",
        channels=channels,
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


def dummy_temporal_array(
    start: float = 0.0,
    num_frames: int = 100,
    num_features: int = 23,
    frame_shift: float = 0.01,
) -> TemporalArray:
    data = np.random.rand(num_frames, num_features).astype(np.float32)
    return MemoryRawWriter().store_array(
        key="temporal-array-float32",
        value=data,
        frame_shift=frame_shift,
        temporal_dim=0,
        start=start,
    )


def dummy_array() -> Array:
    data = np.random.rand(128).astype(np.float32)
    return MemoryRawWriter().store_array("vector-float32", data)


def dummy_temporal_array_uint8(
    start: float = 0.0, num_frames: int = 100, frame_shift: float = 0.01
) -> TemporalArray:
    data = np.random.randint(0, 255, num_frames, dtype=np.uint8)
    return MemoryRawWriter().store_array(
        "temporal-array-int8",
        data,
        frame_shift=frame_shift,
        temporal_dim=0,
        start=start,
    )


def dummy_cut(
    unique_id: int,
    start: float = 0.0,
    duration: float = 1.0,
    recording: Recording = None,
    features: Features = None,
    supervisions=None,
    with_data: bool = False,
):
    custom = {
        "custom_attribute": "dummy-value",
        "custom_attribute_other": "dummy-value-other",
    }
    if with_data:
        custom.update(
            {
                "custom_embedding": dummy_array(),
                "custom_features": dummy_temporal_array(start),
                "custom_recording": dummy_recording(
                    unique_id, duration=duration, with_data=True
                ),
                "custom_indexes": dummy_temporal_array_uint8(start=start),
            }
        )
    return MonoCut(
        id=f"dummy-mono-cut-{unique_id:04d}",
        start=start,
        duration=duration,
        channel=0,
        recording=recording
        if recording
        else dummy_recording(unique_id, with_data=with_data),
        features=features
        if features
        else dummy_features(unique_id, with_data=with_data),
        supervisions=supervisions if supervisions is not None else [],
        custom=custom,
    )


def dummy_multi_cut(
    unique_id: int,
    start: float = 0.0,
    duration: float = 1.0,
    recording: Recording = None,
    features: Features = None,
    supervisions: SupervisionSet = None,
    channel: Optional[List[int]] = None,
    source_per_channel: bool = False,
    with_data: bool = False,
):
    if channel is None:
        channel = [0, 1]
    return MultiCut(
        id=f"dummy-multi-cut-{unique_id:04d}",
        start=start,
        duration=duration,
        channel=channel,
        recording=recording
        if recording
        else dummy_multi_channel_recording(
            unique_id,
            channel_ids=channel,
            with_data=with_data,
            source_per_channel=source_per_channel,
        ),
        features=features
        if features
        else dummy_multi_channel_features(unique_id, channels=channel),
        supervisions=supervisions if supervisions is not None else [],
    )


def remove_spaces_from_segment_text(segment):
    if segment.text is None:
        return segment
    return fastcopy(segment, text=segment.text.replace(" ", ""))
