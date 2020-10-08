#!/usr/bin/env bash

set -eou pipefail

root=$(pwd)/data
data_root=$root/LDC2006S37/data
speech_root=$data_root/speech
transcripts_dir=$data_root/transcripts
heroico_speech_answers_data=$speech_root/heroico/Answers_Spanish
heroico_speech_recordings_data=$speech_root/heroico/Recordings_Spanish
usma_speech_data=$speech_root/usma

heroico_transcripts='heroico-answers.txt heroico-recordings.txt'
usma_transcripts='usma-prompts.txt'

heroico_output_dir=$root/heroico
usma_output_dir=$root/usma

[[ $(uname) == 'Darwin' ]] && nj=$(sysctl -n machdep.cpu.thread_count) || nj=$(grep -c ^processor /proc/cpuinfo)

# Download and untar heroico dataset
#lhotse obtain heroico $root

echo "$0: Prepare audio and supervision manifests for heroico answers."
#lhotse prepare heroico-answers $heroico_speech_answers_data $transcripts_dir $heroico_output_dir

echo "$0: Prepare audio and supervision manifests for heroico recitations."
#lhotse prepare heroico-recitations $heroico_speech_recordings_data $transcripts_dir $heroico_output_dir

echo "$0: Prepare audio and supervision manifests for usma recitations."
#lhotse prepare usma $usma_speech_data $transcripts_dir $usma_output_dir

echo "$0 Extract features for heroico answers."
#lhotse feat extract -j ${nj} -r ${root} ${heroico_output_dir}/recordings-heroico-answers.json ${heroico_output_dir}/feats-heroico-answers

echo "$0 Extract features for heroico Recitations."
#lhotse feat extract -j ${nj} -r ${root} ${heroico_output_dir}/recordings-heroico-recitations.json ${heroico_output_dir}/feats-heroico-recitations

echo "$0 Extract features for usma recitations."
#lhotse feat extract -j ${nj} -r ${root} ${usma_output_dir}/recordings-usma-recitations.json ${usma_output_dir}/feats-usma-recitations

echo "$0:  Create cuts out of features for heroico answers."
#lhotse cut simple -f /Users/john/lhotse_jjm/data/heroico/feats-heroico-answers/feature_manifest.json.gz -s /Users/john/lhotse_jjm/data/heroico/supervisions-heroico-answers.json /Users/john/lhotse_jjm/data/heroico/cuts-heroico-answers.json.gz

echo "$0:  Create cuts out of features for heroico recitations."
#lhotse cut simple -f /Users/john/lhotse_jjm/data/heroico/feats-heroico-recitations/feature_manifest.json.gz -s /Users/john/lhotse_jjm/data/heroico/supervisions-heroico-recitations.json /Users/john/lhotse_jjm/data/heroico/cuts-heroico-recitations.json.gz

echo "$0:  Create cuts out of features for usma recitations."
lhotse cut simple -f /Users/john/lhotse_jjm/data/usma/feats-usma-recitations/feature_manifest.json.gz -s /Users/john/lhotse_jjm/data/usma/supervisions-usma-recitations.json /Users/john/lhotse_jjm/data/usma/cuts-usma-recitations.json.gz

echo "$0: Processing complete - the resulting YAML manifests can be loaded in Python to create a PyTorch dataset."
