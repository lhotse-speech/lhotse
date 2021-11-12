from tempfile import NamedTemporaryFile
import numpy as np
import pytest
from lhotse import Recording, OpenSmileWrapper, OpenSmileConfig
from lhotse.utils import is_module_available

@pytest.fixture()
def recording():
    return Recording.from_file("test/fixtures/libri/libri-1088-134315-0000.wav")
    
    
@pytest.mark.skipif(
    not is_module_available("opensmile"), reason="Opensmile is an optional dependency."
)
def test_extract_feats_default_config(recording):
    feature_extractor = OpenSmileWrapper()
    y = feature_extractor.extract(recording.load_audio(), recording.sampling_rate)
    assert(np.shape(y) == (1600,65))


@pytest.mark.skipif(
    not is_module_available("opensmile"), reason="Opensmile is an optional dependency."
)
def test_extract_feats_opensmile_config(recording):
    import opensmile    
    config = OpenSmileConfig(
                feature_set = opensmile.FeatureSet.ComParE_2016, 
                feature_level = opensmile.FeatureLevel.LowLevelDescriptors,
                sampling_rate = recording.sampling_rate                
            )
    feature_extractor = OpenSmileWrapper(config=config)
    y = feature_extractor.extract(recording.load_audio(), recording.sampling_rate)
    assert(np.shape(y) == (1600,65))


@pytest.mark.parametrize(
    "feature_set,feature_level,expected_shape",
    [   
        ('ComParE_2016','lld',(1600,65)),
        ('ComParE_2016','lld_de',(1602,65)),
        ('ComParE_2016','func',(1,6373)),
        #('GeMAPSv01a','lld',(1600,18)), #deprecated
        #('GeMAPSv01a','func',(1,62)),  #deprecated
        ('GeMAPSv01b','lld',(1600,18)),
        ('GeMAPSv01b','func',(1,62)),
        #('eGeMAPSv01a','lld',(1600,23)), #deprecated
        #('eGeMAPSv01a','func',(1,88)), #deprecated
        #('eGeMAPSv01b','lld',(1600,23)), #deprecated
        #('eGeMAPSv01b','func',(1,88)), #deprecated
        ('eGeMAPSv02','lld',(1600,25)),
        ('eGeMAPSv02','func',(1,88)),
        ('emobase','lld',(1602,26)),
        ('emobase','lld_de',(1604,26)),
        ('emobase','func',(1,988)),
    ],
)
@pytest.mark.skipif(
    not is_module_available("opensmile"), reason="Opensmile is an optional dependency."
)
def test_extract_feats_config_defined_by_string(recording,feature_set, feature_level, expected_shape):
    import opensmile    
    config = OpenSmileConfig(
                feature_set = feature_set, 
                feature_level = feature_level,
                sampling_rate = recording.sampling_rate                
            )
    feature_extractor = OpenSmileWrapper(config=config)
    y = feature_extractor.extract(recording.load_audio(), recording.sampling_rate)
    assert(np.shape(y) == expected_shape)