from math import isclose

import hypothesis.strategies as st
from hypothesis import given, settings

from lhotse.testing.fixtures import RandomCutTestCase


class TestResample(RandomCutTestCase):
    @settings(deadline=None, print_blob=True)
    @given(
        st.one_of(
            st.just(8000),
            st.just(16000),
            st.just(22050),
            st.just(44100),
            st.just(48000),
        ),
        st.one_of(
            st.just(8000),
            st.just(16000),
            st.just(22050),
            st.just(44100),
            st.just(48000),
        ),
        st.data(),
    )
    def test_resample(self, source_sampling_rate, target_sampling_rate, randgen):
        # Draw a number of samples between 0.9 - 1.1 times the sampling rate
        num_samples = randgen.draw(
            st.integers(
                round(source_sampling_rate * 0.9), round(source_sampling_rate * 1.1)
            ),
            label="Numbers of samples for Recordings",
        )
        # Generate random recording
        rec = self.with_recording(
            sampling_rate=source_sampling_rate, num_samples=num_samples
        )
        # Actual test
        rec_rs = rec.resample(target_sampling_rate)
        assert rec_rs.id == rec.id
        # Tolerance of one sample in the resampled domain
        assert isclose(rec_rs.duration, rec.duration, abs_tol=1 / target_sampling_rate)
        samples = rec_rs.load_audio()
        assert samples.shape[0] == rec_rs.num_channels
        assert samples.shape[1] == rec_rs.num_samples
        # Cleanup open file handles
        self.cleanup()
