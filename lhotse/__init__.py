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
from .manipulation import combine, to_manifest
from .serialization import load_manifest, store_manifest
from .supervision import SupervisionSegment, SupervisionSet
from .qa import validate, validate_recordings_and_supervisions, fix_manifests

from . import augmentation
from . import dataset
from . import features
from . import recipes
