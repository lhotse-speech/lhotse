from .audio import (
    AudioSource,
    Recording,
    RecordingSet,
    set_audio_duration_mismatch_tolerance,
    set_ffmpeg_torchaudio_info_enabled,
)
from .caching import is_caching_enabled, set_caching_enabled
from .cut import CutSet, MonoCut, MultiCut, create_cut_set_eager, create_cut_set_lazy
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
from .workflows import *

try:
    # Try to get Lhotse's version (should be created during running pip install / python setup.py ...)
    from .version import __version__
except:
    # Use a default placeholder when the version is unavailable...
    from os import environ as _environ
    from pathlib import Path as _Path

    _base_version_path = _Path(".").parent / "VERSION"
    if _base_version_path.is_file():
        _base_version = open(_base_version_path).read().strip()
        _dev_marker = ""
        if not _environ.get("LHOTSE_PREPARING_RELEASE", False):
            _dev_marker = ".dev"
        __version__ = f"{_base_version}{_dev_marker}+missing.version.file"
    else:
        __version__ = f"0.0.0+unknown.version"

from . import augmentation, dataset, features, recipes

_add_tools_to_path()
