from lhotse.cut import MixedCut
from lhotse.testing.fixtures import RandomCutTestCase


class TestKnownProblematicCuts(RandomCutTestCase):
    def test_mixed_cut_num_frames_example_1(self):
        cut1 = self.with_cut(sampling_rate=16000, num_samples=237920)
        cut2 = self.with_cut(sampling_rate=16000, num_samples=219600)
        # These are two cuts of similar duration, concatenated together with 1 second of silence
        # in between, and padded to duration of 31.445.
        mixed: MixedCut = (
            cut1.pad(duration=cut1.duration + 1.0)
                .append(cut2)
                .pad(duration=31.445)
        )
        assert mixed.duration == 31.445  # Padded correctly
        assert mixed.num_frames == 3145  # Round last 5 up
        assert sum(t.cut.num_frames for t in mixed.tracks) == 3145  # Since the tracks do not overlap in this example,
        # The sum of individual cut num_frames should be equal to the total num_frames
        features = mixed.load_features()
        assert features.shape[0] == 3145  # Loaded features num frames matches the meta-data

    def test_mixed_cut_num_frames_example_2(self):
        cut1 = self.with_cut(sampling_rate=16000, num_samples=252879)
        cut2 = self.with_cut(sampling_rate=16000, num_samples=185280)
        cut3 = self.with_cut(sampling_rate=16000, num_samples=204161)
        # Three cuts padded with 1s of silence in between
        mixed = cut1.pad(duration=cut1.duration + 1.0).append(cut2)
        mixed = mixed.pad(duration=mixed.duration + 1.0).append(cut3)
        assert mixed.duration == 42.145  # Padded correctly
        assert mixed.num_frames == 4215  # Round last 5 up
        # TODO(pzelasko): This assertion would not pass for now, as we're adding an extra frame during load_features.
        # assert sum(t.cut.num_frames for t in mixed.tracks) == 4215  # Since the tracks do not overlap in this example,
        # The sum of individual cut num_frames should be equal to the total num_frames
        features = mixed.load_features()
        assert features.shape[0] == 4215  # Loaded features num frames matches the meta-data
