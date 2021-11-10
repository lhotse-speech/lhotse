#!/usr/bin/env bash

# This script launches the lhotse data preparation steps for the Heroico corpus from the commandline

set -eou pipefail

root=$(pwd)/data
data_root=$root/LDC2006S37/data
speech_root=$data_root/speech
transcripts_dir=$data_root/transcripts
heroico_transcripts='heroico-answers.txt heroico-recordings.txt'
usma_transcripts='usma-prompts.txt'

output_dir=$root/heroico


[[ $(uname) == 'Darwin' ]] && nj=$(sysctl -n machdep.cpu.thread_count) || nj=$(grep -c ^processor /proc/cpuinfo)

# Download and untar heroico dataset
lhotse download heroico $root

echo "$0: Prepare audio and supervision manifests for heroico."
lhotse prepare heroico $speech_root $transcripts_dir $output_dir

for fld in train test devtest; do
  echo "$0 Extract features for heroico $fld corpus."
  lhotse feat extract -j ${nj} -r ${root} ${output_dir}/recordings_${fld}.json ${output_dir}/feats_${fld}

  echo "$0:  Create cuts out of features for heroico $fld corpus."
  lhotse cut simple -f $output_dir/feats_${fld}/feature_manifest.json.gz -s $output_dir/supervisions_${fld}.json $output_dir/cuts_${fld}.json.gz
done
echo "$0: Processing complete - the resulting manifests can be loaded in Python to create a PyTorch dataset."
