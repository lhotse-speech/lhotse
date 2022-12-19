import pytest

import numpy as np
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
        rand_cut = self.with_cut(
            sampling_rate=sr, num_samples=sr, features=False)

        mixed = zero_cut.mix(rand_cut, snr=snr)

        mix_cut_samples = mixed.load_audio()
        assert mix_cut_samples.shape[1] == sr
        assert np.testing.assert_equal(rand_cut.load_audio(), mix_cut_samples)
