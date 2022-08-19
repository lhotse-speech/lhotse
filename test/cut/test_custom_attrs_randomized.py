from hypothesis import given, reproduce_failure, settings
from hypothesis import strategies as st

from lhotse.array import seconds_to_frames
from lhotse.testing.fixtures import RandomCutTestCase

EXAMPLE_TIMEOUT_MS = 3000
MAX_EXAMPLES = 200


class TestCustomAttrPaddingRandomized(RandomCutTestCase):
    @settings(deadline=EXAMPLE_TIMEOUT_MS, max_examples=MAX_EXAMPLES, print_blob=True)
    @given(
        st.one_of(
            st.just(8000),
            st.just(16000),
            st.just(22050),
            st.just(44100),
            st.just(48000),
        ),
        st.one_of(st.just(160), st.just(200), st.just(256)),
        st.one_of(st.just("left"), st.just("right"), st.just("both")),
        st.data(),
    )
    def test_invariants_pad(
        self, sampling_rate: int, window_hop: int, pad_direction: str, rand_gen
    ):
        # Generate cut duration in numbers of samples
        num_samples = rand_gen.draw(
            st.integers(round(sampling_rate * 0.46), round(sampling_rate * 1.9)),
            label="Number of audio samples in a cut.",
        )
        # Generate random cut
        frame_shift = window_hop / sampling_rate
        cut = self.with_cut(
            sampling_rate=sampling_rate,
            num_samples=num_samples,
            frame_shift=frame_shift,
            features=False,
            custom_field=True,
        )
        # Pad with random duration
        duration = rand_gen.draw(
            st.floats(
                min_value=cut.duration + 0.03 * cut.duration, max_value=cut.duration * 2
            ),
            label=f"Padded cut duration",
        )
        padded = cut.pad(
            duration=duration,
            direction=pad_direction,
            pad_value_dict={"codebook_indices": -1},
        )
        # Test the invariants
        array = padded.load_codebook_indices()
        assert array.ndim == padded.codebook_indices.ndim
        expected_num_frames = seconds_to_frames(
            padded.duration, padded.codebook_indices.frame_shift
        )
        assert array.shape[0] == expected_num_frames
        self.cleanup()
