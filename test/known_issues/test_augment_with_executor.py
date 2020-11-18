import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from tempfile import TemporaryDirectory

import pytest

from lhotse import CutSet, Fbank, LilcomFilesWriter, SoxEffectTransform, speed
from lhotse.testing.fixtures import RandomCutTestCase

torchaudio = pytest.importorskip('torchaudio', minversion='0.6')


class TestAugmentationWithExecutor(RandomCutTestCase):
    @pytest.mark.parametrize(
        'exec_type',
        [
            # Multithreading works
            ThreadPoolExecutor,
            # Multiprocessing works, but only when using the "spawn" context
            partial(ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")),
        ]
    )
    def test_wav_augment_with_executor(self, exec_type):
        cut = self.with_cut(sampling_rate=16000, num_samples=16000)
        with TemporaryDirectory() as d, \
                LilcomFilesWriter(storage_path=d) as storage, \
                exec_type(max_workers=4) as ex:
            cut_set = CutSet.from_cuts(
                cut.with_id(str(i)) for i in range(100)
            )
            # Just test that it runs and does not hang.
            cut_set_feats = cut_set.compute_and_store_features(
                extractor=Fbank(),
                storage=storage,
                augment_fn=SoxEffectTransform(speed(16000)),
                executor=ex
            )
