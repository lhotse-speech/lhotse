import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from tempfile import TemporaryDirectory

import pytest

from lhotse import CutSet, Fbank
from lhotse.testing.fixtures import RandomCutTestCase

torchaudio = pytest.importorskip('torchaudio', minversion='0.7.1')


class TestAugmentationWithExecutor(RandomCutTestCase):
    @pytest.mark.parametrize(
        'exec_type',
        [
            # Multithreading works
            ThreadPoolExecutor,
            # Multiprocessing works, but only when using the "spawn" context (in testing)
            pytest.param(
                partial(ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")),
                marks=pytest.mark.skipif(
                    sys.version_info[0] == 3 and sys.version_info[1] < 7,
                    reason="The mp_context argument is introduced in Python 3.7"
                )
            ),
        ]
    )
    def test_wav_augment_with_executor(self, exec_type):
        cut = self.with_cut(sampling_rate=16000, num_samples=16000)
        with TemporaryDirectory() as d, \
                exec_type(max_workers=4) as ex:
            cut_set_speed = CutSet.from_cuts(
                cut.with_id(str(i)) for i in range(100)
            ).perturb_speed(1.1)  # perturb_speed uses torchaudio SoX effect that could hang
            # Just test that it runs and does not hang.
            cut_set_speed_feats = cut_set_speed.compute_and_store_features(
                extractor=Fbank(),
                storage_path=d,
                executor=ex
            )

    @pytest.mark.parametrize(
        'exec_type',
        [
            # Multithreading works
            ThreadPoolExecutor,
            # Multiprocessing works, but only when using the "spawn" context (in testing)
            pytest.param(
                partial(ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")),
                marks=pytest.mark.skipif(
                    sys.version_info[0] == 3 and sys.version_info[1] < 7,
                    reason="The mp_context argument is introduced in Python 3.7"
                )
            ),
        ]
    )
    def test_wav_augment_with_executor(self, exec_type):
        cut = self.with_cut(sampling_rate=16000, num_samples=16000)
        with TemporaryDirectory() as d, \
                exec_type(max_workers=4) as ex:
            cut_set_volume = CutSet.from_cuts(
                cut.with_id(str(i)) for i in range(100)
            ).perturb_volume(0.125)  # perturb_volume uses torchaudio SoX effect that could hang
            # Just test that it runs and does not hang.
            cut_set_volume_feats = cut_set_volume.compute_and_store_features(
                extractor=Fbank(),
                storage_path=d,
                executor=ex
            )
