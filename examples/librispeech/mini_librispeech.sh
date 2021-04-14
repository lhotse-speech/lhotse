#!/usr/bin/env bash

set -eou pipefail

MINI_LIBRISPEECH_ROOT=$(pwd)/data
CORPUS_PATH=${MINI_LIBRISPEECH_ROOT}/LibriSpeech
OUTPUT_PATH=${MINI_LIBRISPEECH_ROOT}/mini_librispeech

data_parts='dev-clean-2 train-clean-5'

[[ $(uname) == 'Darwin' ]] && nj=$(sysctl -n machdep.cpu.thread_count) || nj=$(grep -c ^processor /proc/cpuinfo)

# Download and untar MiniLibriSpeech dataset
lhotse obtain librispeech --mini ${MINI_LIBRISPEECH_ROOT}

# Prepare audio and supervision manifests
lhotse prepare librispeech ${CORPUS_PATH} ${OUTPUT_PATH}

for part in ${data_parts}; do
  # Extract features
  lhotse feat extract -j ${nj} \
    -r ${MINI_LIBRISPEECH_ROOT} \
    ${OUTPUT_PATH}/recordings_${part}.json \
    ${OUTPUT_PATH}/feats_${part}
  # Create cuts out of features
  lhotse cut simple \
    -s ${OUTPUT_PATH}/supervisions_${part}.json \
    -f ${OUTPUT_PATH}/feats_${part}/feature_manifest.json.gz \
    ${OUTPUT_PATH}/cuts_${part}.json.gz
done

# Processing complete - the resulting YAML manifests can be loaded in Python to create a PyTorch dataset.
