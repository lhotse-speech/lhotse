from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tempfile import TemporaryDirectory

import pytest

from lhotse import CutSet, Fbank, LilcomFilesWriter, SoxEffectTransform, speed
from test.known_issues.utils import make_cut

torchaudio = pytest.importorskip('torchaudio', minversion='0.6')


@pytest.mark.parametrize('exec_type', [ProcessPoolExecutor, ThreadPoolExecutor])
def test_wav_augment_with_executor(exec_type):
    with make_cut(sampling_rate=16000, num_samples=16000) as cut, \
            TemporaryDirectory() as d, \
            LilcomFilesWriter(storage_path=d) as storage, \
            exec_type(1) as ex:
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
