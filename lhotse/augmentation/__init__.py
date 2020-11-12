import torchaudio as _torchaudio
from packaging.version import parse as _parse

from .common import AugmentFn
from .wavaugment import *

# Note: we cannot directly compare the Version objects return from _parse because
# Anaconda torchaudio has a version string '0.7.0a0+ac17b64' that is interpreted as
# lesser than 0.7.0.
_ta_version = _parse(_torchaudio.__version__)
_req_version = _parse('0.7')
if _ta_version.major >= _req_version.major and _ta_version.minor >= _req_version.minor:
    from .torchaudio import *
