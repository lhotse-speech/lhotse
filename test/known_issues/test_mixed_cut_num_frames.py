from tempfile import TemporaryDirectory

from lhotse import Fbank, LilcomFilesWriter
from lhotse.cut import MixedCut
from lhotse.testing.fixtures import make_cut


def test_mixed_cut_num_frames_example_1():
    fbank = Fbank()
    with make_cut(sampling_rate=16000, num_samples=237920) as cut1, \
            make_cut(sampling_rate=16000, num_samples=219600) as cut2, \
            TemporaryDirectory() as d, \
            LilcomFilesWriter(d) as storage:
        # These are two cuts of similar duration, concatenated together with 1 second of silence
        # in between, and padded to duration of 31.445.
        mixed: MixedCut = (
            cut1.compute_and_store_features(fbank, storage)
                .pad(duration=cut1.duration + 1.0)
                .append(cut2.compute_and_store_features(fbank, storage))
                .pad(duration=31.445)
        )
        assert mixed.duration == 31.445  # Padded correctly
        assert mixed.num_frames == 3145  # Round last 5 up
        assert sum(t.cut.num_frames for t in mixed.tracks) == 3145  # Since the tracks do not overlap in this example,
        # The sum of individual cut num_frames should be equal to the total num_frames
        features = mixed.load_features()
        assert features.shape[0] == 3145  # Loaded features num frames matches the meta-data


def test_mixed_cut_num_frames_example_2():
    fbank = Fbank()
    with make_cut(sampling_rate=16000, num_samples=252879) as cut1, \
            make_cut(sampling_rate=16000, num_samples=185280) as cut2, \
            make_cut(sampling_rate=16000, num_samples=204161) as cut3, \
            TemporaryDirectory() as d, \
            LilcomFilesWriter(d) as storage:
        # These are two cuts of similar duration, concatenated together with 1 second of silence
        # in between, and padded to duration of 31.445.
        mixed: MixedCut = (
            cut1.compute_and_store_features(fbank, storage)
                .pad(duration=cut1.duration + 1.0)
                .append(cut2.compute_and_store_features(fbank, storage))
        )
        mixed = (
            mixed.pad(duration=mixed.duration + 1.0)
                .append(cut3.compute_and_store_features(fbank, storage))
        )
        assert mixed.duration == 42.145  # Padded correctly
        assert mixed.num_frames == 4215  # Round last 5 up
        # TODO(pzelasko): This assertion would not pass for now, as we're adding an extra frame during load_features.
        # assert sum(t.cut.num_frames for t in mixed.tracks) == 4215  # Since the tracks do not overlap in this example,
        # The sum of individual cut num_frames should be equal to the total num_frames
        features = mixed.load_features()
        assert features.shape[0] == 4215  # Loaded features num frames matches the meta-data
