#!/usr/bin/env bash

dir=test/fixtures/libri
if [ ! -f $dir/audio.json ]; then
  echo "Expected to run this script in the main Lhotse repo directory."
  exit 1
fi

rm $dir/cuts*
rm $dir/feature_manifest.json.gz
rm -rf $dir/storage

lhotse feat extract $dir/audio.json $dir
# Create three variants of cut manifests.
# Seed 0 ensures the RNG always picks the same ID for the cuts.
lhotse --seed 0 cut simple -r $dir/audio.json -f $dir/feature_manifest.json.gz $dir/cuts.json
lhotse --seed 0 cut simple -r $dir/audio.json $dir/cuts_no_feats.json
lhotse --seed 0 cut simple -f $dir/feature_manifest.json.gz $dir/cuts_no_recording.json

for f in $dir/cuts*; do
  lhotse cut truncate -d 10.0 --preserve-id $f $f
done
