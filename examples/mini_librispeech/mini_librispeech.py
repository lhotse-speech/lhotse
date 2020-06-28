import os

from pathlib import Path

import matplotlib.pyplot as plt

from lhotse.recipes.mini_librispeech import download_and_untar, prepare_mini_librispeech
from lhotse.features import FeatureSetBuilder, FeatureExtractor
from lhotse.cut import make_cuts_from_features
from lhotse.dataset.speech_recognition import SpeechRecognitionDataset

root_dir = Path('env')
corpus_dir = root_dir / 'LibriSpeech'
output_dir = root_dir / 'mini_librispeech_nb'

# Download and untar
download_and_untar(root_dir)

# Prepare audio and supervision manifests
mini_librispeech_manifests = prepare_mini_librispeech(corpus_dir, output_dir)

# Extract features
for partition, manifests in mini_librispeech_manifests.items():
    feature_set_builder = FeatureSetBuilder(
        feature_extractor=FeatureExtractor(type='mfcc'),
        output_dir=f'{output_dir}/feats_{partition}'
    )
    feature_set = feature_set_builder.process_and_store_recordings(
        recordings=manifests['audio'],
        num_jobs=os.cpu_count()
    )
    mini_librispeech_manifests[partition]['feats'] = feature_set

    cut_set = make_cuts_from_features(feature_set, manifests['supervisions'])
    mini_librispeech_manifests[partition]['cuts'] = cut_set
    cut_set.to_yaml(output_dir / f'cuts_{partition}.yml')

# Dataset
dataset = SpeechRecognitionDataset(mini_librispeech_manifests['dev-clean-2']['cuts'])

# Illustation
sample = dataset[0]
print('transcript: {}'.format(sample['text']))
plt.matshow(sample['feature'].transpose(0, 1).flip(0))
plt.savefig(output_dir / 'example-feature.png')
