#!/bin/bash

set -eou pipefail

LIBRIMIX_ROOT=$(pwd)
LIBRIMIX_CSV=${LIBRIMIX_ROOT}/MiniLibriMix/metadata/mixture_train_mix_both.csv

[[ `uname`=='Darwin' ]] && nj=`sysctl -n machdep.cpu.thread_count` || nj=`grep -c ^processor /proc/cpuinfo`

# Obtain MiniLibriMix
if [ ! -d MiniLibriMix ]; then
  wget https://zenodo.org/record/3871592/files/MiniLibriMix.zip
  unzip MiniLibriMix.zip
fi

# Prepare audio and supervision manifests
lhotse recipe librimix \
  --with-precomputed-mixtures \
  ${LIBRIMIX_CSV} \
  librimix

for type in sources mix noise; do
  # Extract features for each type of audio file
  lhotse make-feats -j $nj \
    -r ${LIBRIMIX_ROOT} \
    librimix/audio_${type}.yml \
    librimix/feats_${type}
  # Create cuts out of features - cuts_mix.yml will contain pre-mixed cuts for source separation
  lhotse cut simple \
    -s librimix/supervisions_${type}.yml \
    librimix/feats_${type}/feature_manifest.yml.gz \
    librimix/cuts_${type}.yml.gz
done

# Prepare cuts with feature-domain mixes performed on-the-fly - clean
lhotse cut mix-by-recording-id librimix/cuts_sources.yml.gz librimix/cuts_mix_dynamic_clean.yml.gz
# Prepare cuts with feature-domain mixes performed on-the-fly - noisy
lhotse cut mix-by-recording-id librimix/cuts_sources.yml.gz librimix/cuts_noise.yml.gz librimix/cuts_mix_dynamic_noisy.yml.gz

# Processing complete - the resulting YAML mixed cut manifests can be loaded in Python to create a PyTorch dataset.
