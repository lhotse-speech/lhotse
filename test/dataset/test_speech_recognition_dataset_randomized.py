from hypothesis import given, settings
from hypothesis import strategies as st
from torch.utils.data import DataLoader

from lhotse import CutSet
from lhotse.dataset import K2SpeechRecognitionDataset
from lhotse.dataset.cut_transforms import CutConcatenate
from lhotse.dataset.sampling import SimpleCutSampler
from lhotse.testing.fixtures import RandomCutTestCase


class TestCollationRandomized(RandomCutTestCase):
    @settings(deadline=None, print_blob=True)
    @given(
        st.one_of(st.just(8000), st.just(16000), st.just(44100)),
        st.data(),
    )
    def test_no_off_by_one_errors_in_dataset_batch_collation(
        self, sampling_rate: int, data
    ):
        ### Test data preparation ###
        # Generate 10 - 20 cut durations in numbers of samples
        nums_samples = data.draw(
            st.lists(
                st.integers(round(sampling_rate * 0.1), round(sampling_rate * 5.0)),
                min_size=10,
                max_size=20,
            ),
            label="Cuts numbers of samples",
        )
        # Generate random cuts
        cuts = [
            self.with_cut(
                sampling_rate=sampling_rate, num_samples=num_samples, supervision=True
            )
            for num_samples in nums_samples
        ]
        # Mix them with random offsets
        mixed_cuts = CutSet.from_cuts(
            lhs.mix(
                rhs,
                # Sample the offset in terms of number of samples, and then divide by the sampling rate
                # to obtain "realistic" offsets
                offset_other_by=data.draw(
                    st.integers(
                        min_value=int(0.1 * sampling_rate),
                        max_value=int(lhs.duration * sampling_rate),
                    ),
                    label=f"Offset for pair {idx + 1}",
                )
                / sampling_rate,
            )
            for idx, (lhs, rhs) in enumerate(zip(cuts, cuts[1:]))
        )
        # Create an ASR dataset
        dataset = K2SpeechRecognitionDataset(
            return_cuts=True,
            cut_transforms=[CutConcatenate(duration_factor=3.0)],
        )
        sampler = SimpleCutSampler(
            mixed_cuts,
            shuffle=False,
            max_cuts=10000,
        )
        dloader = DataLoader(dataset, batch_size=None, sampler=sampler)
        ### End of test data preparation ###
        # Test the invariants
        for batch in dloader:
            sups = batch["supervisions"]
            cuts = sups["cut"]
            for idx, cut in enumerate(cuts):
                assert (
                    sups["start_frame"][idx] + sups["num_frames"][idx] <= cut.num_frames
                ), f"Error at index {idx}"
                # assert sups['start_sample'][idx] + sups['num_samples'][
                #     idx] <= cut.num_samples, f"Error at index {idx}"
        # Need to call cleanup manually to free the file handles, otherwise the test may crash
        self.cleanup()
