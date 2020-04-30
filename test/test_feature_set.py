from lhotse.audio import AudioSet
from lhotse.features import FeatureSet, FeatureExtractor, FeatureSegment
from lhotse.supervision import SupervisionSet

other_params = {}
some_augmentation = None


def test_feature_set():
    # Use case #1 - pre-computed features, stored somewhere, specified by an existing manifest
    feature_set = FeatureSet.from_yaml('...')

    # Use case #2
    # - feature extraction
    # - with recordings delivered by AudioSet,
    # - with specified feature parameters (mfcc/fbank/..., frame shapes, etc.),
    # - with specified optional segmentation (default: extract feats for whole recordings)
    # Note: introduce FeatureExtractor as a FeatureSet builder for separation of concerns (loading and manipulating feauters vs extracting them)

    audio_set = AudioSet.from_yaml('...')

    # Variant A: whole recordings
    whole_recording_feature_set: FeatureSet = (
        FeatureExtractor()
            .with_audio_set(audio_set)
            .with_augmentation(some_augmentation)
            .with_algorithm(method='mfcc', frame_size=0.025, frame_shift=0.01, **other_params)
            .extract()
    )

    # Variant B: Exact segments from SupervisionSet
    supervision_set = SupervisionSet.from_yaml('...')

    whole_recording_feature_set: FeatureSet = (
        FeatureExtractor()
            .with_audio_set(audio_set)
            .with_augmentation(some_augmentation)
            .with_algorithm(method='mfcc', frame_size=0.025, frame_shift=0.01, **other_params)
            .with_segmentation(supervision_set, extra_left_seconds=0.0, extra_right_seconds=0.0)
            .extract()
    )

    # Variant C: custom segmentation (perhaps redundant)
    segmentation = [
        #               recording-id  channel  start  duration
        FeatureSegment('recording-1', 0, 0.5, 0.5),
        FeatureSegment('recording-1', 1, 1.3, 1.5),
        FeatureSegment('recording-1', 0, 2.4, 1.1),
        FeatureSegment('recording-2', 0, 0.0, 30.)
    ]
    segment_feature_set: FeatureSet = (
        FeatureExtractor()
            .with_audio_set(audio_set)
            .with_augmentation(some_augmentation)
            .with_algorithm(method='mfcc', frame_size=0.025, frame_shift=0.01, **other_params)
            .with_segmentation(segmentation)
            .extract()
    )
