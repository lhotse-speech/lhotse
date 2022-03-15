from collections import defaultdict
import logging
from math import isclose
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from lhotse.array import Array, TemporalArray
from lhotse.audio import Recording, RecordingSet
from lhotse.cut import Cut, CutSet, MixedCut, PaddingCut
from lhotse.features import FeatureSet, Features
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import compute_num_frames, overlaps

_VALIDATORS: Dict[str, Callable] = {}


def validate(obj: Any, read_data: bool = False) -> None:
    """
    Validate a Lhotse manifest object.
    It checks for conditions such as positive duration, matching channels, ids, etc.
    It raises AssertionError when it finds some mismatch.

    Optionally it can load the audio/feature data from disk and inspect whether the
    num samples/frames/features declared in the manifests are matching the actual data.

    This function determines the passed object's type and automatically calls
    the proper validator for that object.
    """
    validator = None
    for registered_type in _VALIDATORS:
        if isinstance(obj, registered_type):
            validator = _VALIDATORS[registered_type]
            break
    if validator is None:
        raise ValueError(
            f"Object of unknown type passed to validate() (T = {type(obj)}, known types = {list(_VALIDATORS)}"
        )
    validator(obj, read_data=read_data)


def fix_manifests(
    recordings: RecordingSet, supervisions: SupervisionSet
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Fix a pair of :class:`~lhotse.audio.RecordingSet` and :class:`~lhotse.supervision.SupervisionSet`,
    which is conceptually similar to how Kaldi's ``utils/fix_data_dir.sh`` works.

    We will:
        - remove all supervisions without a corresponding recording;
        - remove all recordings without a corresponding supervision;
        - remove all supervisions that exceed the duration of a recording;
        - trim supervisions that exceed the duration of a recording but start before its end;
        - and possibly other operations in the future.

    :param recordings: a :class:`~lhotse.audio.RecordingSet` instance.
    :param supervisions: a corresponding :class:`~lhotse.supervision.SupervisionSet` instance.
    :return: a pair of ``recordings`` and ``supervisions`` that were fixed:
        the original manifests are not modified.
    """
    recordings, supervisions = remove_missing_recordings_and_supervisions(
        recordings, supervisions
    )
    if len(recordings) == 0 or len(supervisions) == 0:
        raise ValueError(
            "There are no matching recordings and supervisions in the input manifests."
        )
    supervisions = trim_supervisions_to_recordings(recordings, supervisions)
    if len(supervisions) == 0:
        raise ValueError("All supervisions exceed the recordings duration.")
    return recordings, supervisions


def validate_recordings_and_supervisions(
    recordings: Union[RecordingSet, Recording],
    supervisions: Union[SupervisionSet, SupervisionSegment],
    read_data: bool = False,
) -> None:
    """
    Validate the recording and supervision manifests separately,
    and then check if they are consistent with each other.

    This method will emit warnings, instead of errors, when some recordings or supervisions
    are missing their counterparts.
    These items will be discarded by default when creating a CutSet.
    """
    if isinstance(recordings, Recording):
        recordings = RecordingSet.from_recordings([recordings])
    if isinstance(supervisions, SupervisionSegment):
        supervisions = SupervisionSet.from_segments([supervisions])

    if recordings.is_lazy:
        recordings = RecordingSet.from_recordings(iter(recordings))
    if supervisions.is_lazy:
        supervisions = SupervisionSet.from_segments(iter(supervisions))

    validate(recordings, read_data=read_data)
    validate(supervisions)
    # Errors
    for s in supervisions:
        r = recordings[s.recording_id]
        assert -1e-3 <= s.start <= s.end <= r.duration + 1e-3, (
            f"Supervision {s.id}: exceeded the bounds of its corresponding recording "
            f"(supervision spans [{s.start}, {s.end}]; recording spans [0, {r.duration}])"
        )
        assert s.channel in r.channel_ids, (
            f"Supervision {s.id}: channel {s.channel} does not exist in its corresponding Recording "
            f"(recording channels: {r.channel_ids})"
        )
    # Warnings
    recording_ids = frozenset(r.id for r in recordings)
    recording_ids_in_sups = frozenset(s.recording_id for s in supervisions)
    only_in_recordings = recording_ids - recording_ids_in_sups
    if only_in_recordings:
        logging.warning(
            f"There are {len(only_in_recordings)} recordings that "
            f"do not have any corresponding supervisions in the SupervisionSet."
        )
    only_in_supervisions = recording_ids_in_sups - recording_ids
    if only_in_supervisions:
        logging.warning(
            f"There are {len(only_in_supervisions)} supervisions that "
            f"are missing their corresponding recordings in the RecordingSet."
        )


def remove_missing_recordings_and_supervisions(
    recordings: RecordingSet,
    supervisions: SupervisionSet,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Fix the recording and supervision manifests by removing all entries that
    miss their counterparts.

    :param recordings: a :class:`RecordingSet` object.
    :param supervisions: a :class:`RecordingSet` object.
    :return: A pair of :class:`RecordingSet` and :class:`SupervisionSet` with removed entries.
    """
    recording_ids = frozenset(r.id for r in recordings)
    recording_ids_in_sups = frozenset(s.recording_id for s in supervisions)
    only_in_recordings = recording_ids - recording_ids_in_sups
    if only_in_recordings:
        recordings = recordings.filter(lambda r: r.id not in only_in_recordings)
        logging.warning(
            f"Removed {len(only_in_recordings)} recordings with no corresponding supervisions."
        )
    only_in_supervisions = recording_ids_in_sups - recording_ids
    if only_in_supervisions:
        n_orig_sups = len(supervisions)
        supervisions = supervisions.filter(
            lambda s: s.recording_id not in only_in_supervisions
        )
        n_removed_sups = n_orig_sups - len(supervisions)
        logging.warning(
            f"Removed {n_removed_sups} supervisions with no corresponding recordings "
            f"(for a total of {len(only_in_supervisions)} recording IDs)."
        )
    return recordings, supervisions


def trim_supervisions_to_recordings(
    recordings: RecordingSet, supervisions: SupervisionSet
) -> SupervisionSet:
    """
    Return a new :class:`~lhotse.supervision.SupervisionSet` with supervisions that are
    not exceeding the duration of their corresponding :class:`~lhotse.audio.Recording`.
    """
    if recordings.is_lazy:
        recordings = RecordingSet.from_recordings(iter(recordings))

    sups = []
    removed = 0
    trimmed = 0
    for s in supervisions:
        end = recordings[s.recording_id].duration
        if s.start > end:
            removed += 1
            continue
        if s.end > end:
            trimmed += 1
            s = s.trim(recordings[s.recording_id].duration)
        sups.append(s)
    if removed:
        logging.warning(
            f"Removed {removed} supervisions starting after the end of the recording."
        )
    if trimmed:
        logging.warning(
            f"Trimmed {trimmed} supervisions exceeding the end of the recording."
        )
    return SupervisionSet.from_segments(sups)


def register_validator(fn):
    """
    Decorator registers the function to be invoked inside ``validate()``
    when the first argument's type is matching.
    """
    # Check the first function argument's type
    first_arg_type = next(iter(fn.__annotations__.values()))
    # Register the function to be called when an object of that type is passed to validate()
    _VALIDATORS[first_arg_type] = fn
    return fn


@register_validator
def validate_recording(r: Recording, read_data: bool = False) -> None:
    assert (
        r.duration > 0
    ), f"Recording {r.id}: duration has to be greater than 0 (is {r.duration})"
    expected_duration = r.num_samples / r.sampling_rate
    assert r.num_channels > 0, f"Recording {r.id}: no channels available"
    assert isclose(expected_duration, r.duration), (
        f"Recording {r.id}: mismatched declared duration ({r.duration}) with "
        f"num_samples / sampling_rate ({expected_duration})."
    )
    if read_data:
        samples = r.load_audio()
        n_ch, n_s = samples.shape
        assert (
            r.num_channels == n_ch
        ), f"Recording {r.id}: expected {r.num_channels} channels, got {n_ch}"
        assert (
            r.num_samples == n_s
        ), f"Recording {r.id}: expected {r.num_samples} samples, got {n_s}"


@register_validator
def validate_supervision(
    s: SupervisionSegment, read_data: bool = False, **kwargs
) -> None:
    assert (
        s.duration > 0
    ), f"Supervision {s.id}: duration has to be greater than 0 (is {s.duration})"

    # Conditions related to custom fields
    if s.custom is not None:
        assert isinstance(
            s.custom, dict
        ), f"SupervisionSegment {s.id}: custom field has to be set to a dict or None."
        for key, value in s.custom.items():
            if isinstance(value, Array):
                validate_array(value, read_data=read_data)
            elif isinstance(value, TemporalArray):
                validate_temporal_array(value, read_data=read_data)
                if not isclose(s.duration, value.duration):
                    logging.warning(
                        f"SupervisionSegment {s.id}: possibly mismatched "
                        f"duration between supervision ({s.duration}s) and temporal array "
                        f"in custom field '{key}' (num_frames={value.num_frames} * "
                        f"frame_shift={value.frame_shift} == duration={value.duration})."
                    )


@register_validator
def validate_features(
    f: Features, read_data: bool = False, feats_data: Optional[np.ndarray] = None
) -> None:
    assert f.start >= 0, f"Features: start has to be greater than 0 (is {f.start})"
    assert (
        f.duration > 0
    ), f"Features: duration has to be greater than 0 (is {f.duration})"
    assert (
        f.num_frames > 0
    ), f"Features: num_frames has to be greater than 0 (is {f.num_frames})"
    assert (
        f.num_features > 0
    ), f"Features: num_features has to be greater than 0 (is {f.num_features})"
    assert (
        f.sampling_rate > 0
    ), f"Features: sampling_rate has to be greater than 0 (is {f.sampling_rate})"
    assert (
        f.frame_shift > 0
    ), f"Features: frame_shift has to be greater than 0 (is {f.frame_shift})"
    window_hop = round(f.frame_shift * f.sampling_rate, ndigits=12)
    assert float(int(window_hop)) == window_hop, (
        f"Features: frame_shift of {f.frame_shift} is incorrect because it is physically impossible; "
        f"multiplying it by a sampling rate of {f.sampling_rate} results in a fractional window hop "
        f"of {window_hop} samples."
    )
    expected_num_frames = compute_num_frames(
        duration=f.duration, frame_shift=f.frame_shift, sampling_rate=f.sampling_rate
    )
    assert expected_num_frames == f.num_frames, (
        f"Features: manifest is inconsistent: declared num_frames is {f.num_frames}, "
        f"but duration ({f.duration}s) / frame_shift ({f.frame_shift}s) results in {expected_num_frames} frames. "
        f"If you're using a custom feature extractor, you might need to ensure that it preserves "
        f"this relationship between duration, frame_shift and num_frames (use rounding up if needed - "
        f"see lhotse.utils.compute_num_frames)."
    )
    if read_data or feats_data is not None:
        if read_data:
            feats_data = f.load()
        n_fr, n_ft = feats_data.shape
        assert (
            f.num_frames == n_fr
        ), f"Features: expected num_frames: {f.num_frames}, actual: {n_fr}"
        assert (
            f.num_features == n_ft
        ), f"Features: expected num_features: {f.num_features}, actual: {n_ft}"


@register_validator
def validate_array(arr: Array, read_data: bool = False) -> None:
    if read_data:
        data = arr.load()
        assert data.shape == arr.shape


@register_validator
def validate_temporal_array(arr: TemporalArray, read_data: bool = False) -> None:
    assert arr.temporal_dim >= 0, "TemporalArray: temporal_dim cannot be negative."
    assert arr.temporal_dim < arr.ndim, (
        f"TemporalArray: temporal_dim {arr.temporal_dim} "
        f"canot be greater than ndim {arr.ndim}."
    )
    assert arr.frame_shift > 0, "TemporalArray: frame_shift must be positive."
    assert arr.start >= 0, "TemporalArray: start must be non-negative."
    if read_data:
        data = arr.load()
        assert data.shape == arr.shape


@register_validator
def validate_cut(c: Cut, read_data: bool = False) -> None:
    # Validate MixedCut
    if isinstance(c, MixedCut):
        assert (
            len(c.tracks) > 0
        ), f"MonoCut {c.id}: Mixed cut must have at least one track."
        for idx, track in enumerate(c.tracks):
            validate_cut(track.cut, read_data=read_data)
            assert (
                track.offset >= 0
            ), f"MonoCut: {c.id}: track {idx} has a negative offset."
        return

    # Validate MonoCut and PaddingCut
    assert c.start >= 0, f"MonoCut {c.id}: start must be 0 or greater (got {c.start})"
    assert (
        c.duration > 0
    ), f"MonoCut {c.id}: duration must be greater than 0 (got {c.duration})"
    assert (
        c.sampling_rate > 0
    ), f"MonoCut {c.id}: sampling_rate must be greater than 0 (got {c.sampling_rate})"
    assert (
        c.has_features or c.has_recording
    ), f"MonoCut {c.id}: must have either Features or Recording attached."

    # The rest pertains only to regular Cuts
    if isinstance(c, PaddingCut):
        return

    # Conditions related to features
    if c.has_features:
        validate_features(c.features)
        assert c.channel == c.features.channels
        if read_data:
            # We are not passing "read_data" to "validate_features" to avoid loading feats twice;
            # we'll just validate the subset of the features relevant for the cut.
            feats = c.load_features()
            n_fr, n_ft = feats.shape
            assert (
                c.num_frames == n_fr
            ), f"MonoCut {c.id}: expected num_frames: {c.num_frames}, actual: {n_fr}"
            assert (
                c.num_features == n_ft
            ), f"MonoCut {c.id}: expected num_features: {c.num_features}, actual: {n_ft}"

    # Conditions related to recording
    if c.has_recording:
        validate_recording(c.recording)
        assert c.channel in c.recording.channel_ids
        if read_data:
            # We are not passing "read_data" to "validate_recording" to avoid loading audio twice;
            # we'll just validate the subset of the recording relevant for the cut.
            samples = c.load_audio()
            assert (
                c.num_samples == samples.shape[1]
            ), f"MonoCut {c.id}: expected {c.num_samples} samples, got {samples.shape[1]}"

    # Conditions related to supervisions
    for s in c.supervisions:
        validate_supervision(s)
        assert s.recording_id == c.recording_id, (
            f"MonoCut {c.id}: supervision {s.id} has a mismatched recording_id "
            f"(expected {c.recording_id}, supervision has {s.recording_id})"
        )
        assert s.channel == c.channel, (
            f"MonoCut {c.id}: supervision {s.id} has a mismatched channel "
            f"(expected {c.channel}, supervision has {s.channel})"
        )

    # Conditions related to custom fields
    if c.custom is not None:
        assert isinstance(
            c.custom, dict
        ), f"MonoCut {c.id}: custom field has to be set to a dict or None."
        for key, value in c.custom.items():
            if isinstance(value, Array):
                validate_array(value, read_data=read_data)
            elif isinstance(value, TemporalArray):
                validate_temporal_array(value, read_data=read_data)
                if not isclose(c.duration, value.duration):
                    logging.warning(
                        f"MonoCut {c.id}: possibly mismatched "
                        f"duration between cut ({c.duration}s) and temporal array "
                        f"in custom field '{key}' (num_frames={value.num_frames} * "
                        f"frame_shift={value.frame_shift} == duration={value.duration})."
                    )
                assert overlaps(c, value), (
                    f"MonoCut {c.id}: TemporalArray at custom field '{key}' "
                    "seems to have incorrect start time (the array with time span "
                    f"[{value.start}s - {value.end}s] does not overlap with cut "
                    f"with time span [{c.start}s - {c.end}s])."
                )


@register_validator
def validate_recording_set(recordings: RecordingSet, read_data: bool = False) -> None:
    rates = set()
    for r in recordings:
        validate_recording(r, read_data=read_data)
        rates.add(r.sampling_rate)
    if len(rates) > 1:
        logging.warning(
            f"RecordingSet contains recordings with different sampling rates ({rates}). "
            f"Make sure that this was intended."
        )


@register_validator
def validate_supervision_set(supervisions: SupervisionSet, **kwargs) -> None:
    for s in supervisions:
        validate_supervision(s)

    # Catch errors in data preparation:
    # - more than one supervision for a given recording starts at 0 (in a given channel)
    supervisions._index_by_recording_id_and_cache()
    for rid, sups in supervisions._segments_by_recording_id.items():
        cntr_per_channel = defaultdict(int)
        for s in sups:
            cntr_per_channel[s.channel] += int(s.start == 0)
        for channel, count in cntr_per_channel.items():
            if count > 1:
                logging.warning(
                    f"SupervisionSet contains {count} supervisions that start at 0 for recording {rid} "
                    f"(channel {channel}). Did you forget to set supervision start times?"
                )


@register_validator
def validate_feature_set(features: FeatureSet, read_data: bool = False) -> None:
    first = next(iter(features))
    sampling_rate = first.sampling_rate
    num_features = first.num_features
    features_type = first.type
    for idx, f in enumerate(features):
        validate_features(f, read_data=read_data)
        assert f.sampling_rate == sampling_rate, (
            f"FeatureSet: mismatched sampling rate (the first Features manifest had {sampling_rate}, "
            f"got {f.sampling_rate} in Features at index {idx})"
        )
        assert f.num_features == num_features, (
            f"FeatureSet: mismatched num_features (the first Features manifest had {num_features}, "
            f"got {f.num_features} in Features at index {idx})"
        )
        assert f.type == features_type, (
            f"FeatureSet: mismatched feature_type (the first Features manifest had {features_type}, "
            f"got {f.type} in Features at index {idx})"
        )


@register_validator
def validate_cut_set(cuts: CutSet, read_data: bool = False) -> None:
    for c in cuts:
        validate_cut(c, read_data=read_data)
