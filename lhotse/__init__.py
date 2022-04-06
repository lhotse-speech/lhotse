from .audio import (
    AudioSource,
    Recording,
    RecordingSet,
    set_audio_duration_mismatch_tolerance,
)
from .augmentation import *
from .caching import is_caching_enabled, set_caching_enabled
from .cut import CutSet, MonoCut, create_cut_set_eager, create_cut_set_lazy
from .features import *
from .kaldi import load_kaldi_data_dir
from .manipulation import combine, split_parallelize_combine, to_manifest
from .qa import fix_manifests, validate, validate_recordings_and_supervisions
from .serialization import load_manifest, load_manifest_lazy, store_manifest
from .supervision import SupervisionSegment, SupervisionSet
from .tools.env import add_tools_to_path as _add_tools_to_path

try:
    # Try to get Lhotse's version (should be created during running pip install / python setup.py ...)
    from .version import __version__
except:
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # NOTE: REMEMBER TO UPDATE THE ACTUAL VERSION IN setup.py WHEN RELEASING #
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # Use a default placeholder when the version is unavailable...
    __version__ = "1.0.0+missing.version.file"

from . import augmentation
from . import dataset
from . import features
from . import recipes


_add_tools_to_path()
