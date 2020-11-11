import torchaudio as _torchaudio

from .common import AugmentFn
from .wavaugment import *

if _torchaudio.__version__ >= '0.7.0':
    from .torchaudio import *
