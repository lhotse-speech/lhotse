import warnings

with warnings.catch_warnings():
    # Those torchaudio warnings are a real nuisance
    warnings.simplefilter("ignore")
    # noinspection PyUnresolvedReferences
    import torchaudio

from .audio import AudioSource, Recording, RecordingSet
from .augmentation import *
from .cut import MonoCut, CutSet
from .features import *
from .kaldi import load_kaldi_data_dir
from .manipulation import combine, to_manifest, split_parallelize_combine
from .serialization import load_manifest, store_manifest
from .supervision import SupervisionSegment, SupervisionSet
from .tools.env import add_tools_to_path as _add_tools_to_path
from .qa import validate, validate_recordings_and_supervisions, fix_manifests

try:
    # Try to get Lhotse's version (should be created during running pip install / python setup.py ...)
    from .version import __version__
except:
    # Use a default placeholder when the version is unavailable...
    __version__ = '0.0.0.dev'

from . import augmentation
from . import dataset
from . import features
from . import recipes


_add_tools_to_path()
