import warnings

with warnings.catch_warnings():
    # Those torchaudio warnings are a real nuisance
    warnings.simplefilter('ignore')
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

from . import augmentation
from . import dataset
from . import features
from . import recipes


# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
# (this snippet is borrowed from scikit-learn's __init__.py)
#
# Generic release markers:
#   X.Y.0   # For first release after an increment in Y
#   X.Y.Z   # For bugfix releases
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
# >>>>>>>>>>>>>>>>>>> CAUTION <<<<<<<<<<<<<<<<<<<<<
# > For public releases, remove the '.dev' suffix <
# > For non-public releases, add a '.dev' suffix  <
# >>>>>>>>>>>>>>>>>>> CAUTION <<<<<<<<<<<<<<<<<<<<<

__version__ = "0.8.0"


_add_tools_to_path()
