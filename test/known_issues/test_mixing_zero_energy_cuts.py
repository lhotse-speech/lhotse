import pytest

from lhotse import CutSet
from lhotse.dataset.collation import collate_audio
from lhotse.testing.fixtures import RandomCutTestCase
from lhotse.utils import NonPositiveEnergyError


class TestMixZeroEnergyCuts(RandomCutTestCase):
    @pytest.mark.parametrize("snr", [None, 10])
    def test_mix_zero_energy_cut_raises(self, snr):
        sr = 16000
        zero_cut = self.with_cut(
            sampling_rate=sr, num_samples=sr, features=False, use_zeroes=True
        )
        rand_cut = self.with_cut(sampling_rate=sr, num_samples=sr, features=False)

        mixed = zero_cut.mix(rand_cut, snr=snr)

        mix_cut_samples = mixed.load_audio()
        assert mix_cut_samples.shape[1] == sr

    @pytest.mark.parametrize("snr", [None, 10])
    def test_fault_tolerant_loading_skips_cut(self, snr):
        sr = 16000
        zero_cut = self.with_cut(
            sampling_rate=sr, num_samples=sr, features=False, use_zeroes=True
        )
        rand_cut = self.with_cut(sampling_rate=sr, num_samples=sr, features=False)

        zero_mixed = zero_cut.mix(rand_cut, snr=snr)
        rand_mixed = rand_cut.mix(rand_cut, snr=snr)

        cuts_all = CutSet.from_cuts([zero_mixed, rand_mixed])

        audio, audio_lens, cuts_ok = collate_audio(cuts_all, fault_tolerant=True)
        assert cuts_ok[0] == zero_mixed
        assert cuts_ok[1] == rand_mixed
