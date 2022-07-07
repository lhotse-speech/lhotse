from .audio import (
    AudioSource,
    Recording,
    RecordingSet,
    set_audio_duration_mismatch_tolerance,
)
from .caching import is_caching_enabled, set_caching_enabled
from .cut import CutSet, MonoCut, create_cut_set_eager, create_cut_set_lazy
from .features import *
from .kaldi import load_kaldi_data_dir
from .manipulation import combine, split_parallelize_combine, to_manifest
from .qa import fix_manifests, validate, validate_recordings_and_supervisions
from .serialization import load_manifest, load_manifest_lazy, store_manifest
from .supervision import SupervisionSegment, SupervisionSet
from .tools.env import add_tools_to_path as _add_tools_to_path
from .utils import (
    Decibels,
    Seconds,
    add_durations,
    compute_num_frames,
    compute_num_samples,
    fastcopy,
    fix_random_seed,
    measure_overlap,
    streaming_shuffle,
)

try:
    # Try to get Lhotse's version (should be created during running pip install / python setup.py ...)
    from .version import __version__
except:
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # NOTE: REMEMBER TO UPDATE THE ACTUAL VERSION IN setup.py WHEN RELEASING #
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # Use a default placeholder when the version is unavailable...
    __version__ = "1.4.0+missing.version.file"

from . import augmentation, dataset, features, recipes

_add_tools_to_path()
