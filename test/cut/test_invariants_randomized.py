from hypothesis import given, settings
from hypothesis import strategies as st

from lhotse.testing.fixtures import RandomCutTestCase


class TestMixedCutNumFramesNumSamplesRandomized(RandomCutTestCase):
    @settings(deadline=None, print_blob=True)
    @given(
        st.one_of(st.just(8000), st.just(16000), st.just(44100), st.just(48000)),
        st.data(),
    )
    def test_invariants_mix(
            self,
            sampling_rate: int,
            data
    ):
        # Generate 2 - 10 cut durations in numbers of samples
        nums_samples = data.draw(
            st.lists(
                st.integers(
                    round(sampling_rate * 0.1),
                    round(sampling_rate * 5.0)
                ),
                min_size=2,
                max_size=10
            ),
            label='Cuts numbers of samples'
        )
        # Generate random cuts
        cuts = [
            self.with_cut(sampling_rate=sampling_rate, num_samples=num_samples)
            for num_samples in nums_samples
        ]
        # Mix them with random offsets
        mixed = cuts[0]
        for idx, cut in enumerate(cuts[1:]):
            offset = data.draw(st.floats(min_value=0.1, max_value=mixed.duration), label=f'Offset for cut {idx + 1}')
            mixed = mixed.mix(cut, offset_other_by=offset)
        # Truncate somewhere around the middle
        end = data.draw(st.floats(mixed.duration * 0.4, mixed.duration * 0.6), label='Truncated duration')
        mixed = mixed.truncate(duration=end)
        # Test the invariants
        feats = mixed.load_features()
        samples = mixed.load_audio()
        assert mixed.has_features
        assert feats.shape[0] == mixed.num_frames
        assert feats.shape[1] == mixed.num_features
        assert mixed.has_recording
        assert samples.shape[0] == 1
        assert samples.shape[1] == mixed.num_samples
        self.cleanup()

    @settings(deadline=None, print_blob=True)
    @given(
        st.one_of(st.just(8000), st.just(16000), st.just(44100), st.just(48000)),
        st.data(),
    )
    def test_invariants_append(
            self,
            sampling_rate: int,
            data
    ):
        # Generate 2 - 10 cut durations in numbers of samples
        nums_samples = data.draw(
            st.lists(
                st.integers(
                    round(sampling_rate * 0.1),
                    round(sampling_rate * 5.0)
                ),
                min_size=2,
                max_size=10
            ),
            label='Cuts numbers of samples'
        )
        # Generate random cuts
        cuts = [
            self.with_cut(sampling_rate=sampling_rate, num_samples=num_samples)
            for num_samples in nums_samples
        ]
        # Append the cuts
        mixed = cuts[0]
        for idx, cut in enumerate(cuts[1:]):
            mixed = mixed.append(cut)
        # Test the invariants
        feats = mixed.load_features()
        samples = mixed.load_audio()
        assert mixed.has_features
        assert feats.shape[0] == mixed.num_frames
        assert feats.shape[1] == mixed.num_features
        assert mixed.has_recording
        assert samples.shape[0] == 1
        assert samples.shape[1] == mixed.num_samples
        self.cleanup()
