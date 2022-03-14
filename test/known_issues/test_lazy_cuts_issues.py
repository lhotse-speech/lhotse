from tempfile import NamedTemporaryFile, TemporaryDirectory

from lhotse import CutSet, combine, load_manifest_lazy
from lhotse.testing.dummies import DummyManifest


def test_lazy_cuts_combine_split_issue():
    # Test for lack of exception
    cuts = DummyManifest(CutSet, begin_id=0, end_id=1000)
    with TemporaryDirectory() as d, NamedTemporaryFile(suffix=".jsonl.gz") as f:
        cuts.to_file(f.name)
        f.flush()

        cuts_lazy = load_manifest_lazy(f.name)
        cuts_lazy = combine(cuts_lazy, cuts_lazy.perturb_speed(0.9))
        cuts_lazy.split_lazy(d, chunk_size=100)
