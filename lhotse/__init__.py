import warnings

with warnings.catch_warnings():
    # Those torchaudio warnings are a real nuisance
    warnings.simplefilter('ignore')
    # noinspection PyUnresolvedReferences
    import torchaudio

from .audio import AudioSource, Recording, RecordingSet
from .augmentation import *
from .cut import Cut, CutSet
from .features import *
from .kaldi import load_kaldi_data_dir
from .manipulation import combine, load_manifest, to_manifest
from .supervision import SupervisionSegment, SupervisionSet
from .qa import validate, validate_recordings_and_supervisions

from . import recipes
