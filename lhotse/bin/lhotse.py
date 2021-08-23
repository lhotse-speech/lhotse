#!/usr/bin/env python3
"""
Use this script like:

$ lhotse --help
$ lhotse make-feats --help
$ lhotse make-feats --compressed recording_manifest.yml mfcc_dir/
$ lhotse write-default-feature-config feat-conf.yml
$ lhotse kaldi import data/train 16000 train_manifests/
$ lhotse split 3 audio.yml split_manifests/
$ lhotse combine feature.1.yml feature.2.yml combined_feature.yml
$ lhotse recipe --help
$ lhotse recipe librimix-dataprep path/to/librimix.csv output_manifests_dir/
$ lhotse recipe librimix-obtain target_dir/
$ lhotse recipe mini-librispeech-dataprep corpus_dir/ output_manifests_dir/
$ lhotse recipe mini-librispeech-obtain target_dir/
$ lhotse cut --help
$ lhotse cut simple supervisions.yml features.yml simple_cuts.yml
$ lhotse cut stereo-mixed supervisions.yml features.yml mixed_cuts.yml
"""

# Note: we import all the CLI modes here so they get auto-registered
#       in Lhotse's main CLI entry-point. Then, setuptools is told to
#       invoke the "cli()" method from this script.
from lhotse.bin.modes import *
