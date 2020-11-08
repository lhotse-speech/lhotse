#!/usr/bin/env bash

set -eou pipefail

mkdir -p deps
pushd deps

ext='so'
ld_var='LD_LIBRARY_PATH'
if [[ "$OSTYPE" == "darwin"* ]]; then
  ext='dylib'
  ld_var='DYLD_FALLBACK_LIBRARY_PATH'
fi

# Build libsox from source
if [ ! -f sox-code/src/.libs/libsox.${ext} ]; then
  git clone --depth 300 git://git.code.sf.net/p/sox/code sox-code
  pushd sox-code
  # Note(pzelasko): This seems to be the last version of libsox that builds on my MacOS
  # and CLSP grid without any issues...
  git checkout f0574854aff841d3be65f82bf74eb46272cd8588
  autoreconf -i
  ./configure
  make
  popd
fi
# WavAugment needs to see sox sources and libs
pushd sox-code
export CPPFLAGS="-I$(pwd)/src"
export LDFLAGS="-L$(pwd)/src/.libs"
popd

if [ ! -d WavAugment ]; then
  git clone https://github.com/facebookresearch/WavAugment
fi
pushd WavAugment
git checkout de4cbd5490cc32c1cd21aab4422978e57c95b9dc
pip install .
popd

echo "To be able to import WavAugment in Python, you will need to set the following env variable (e.g. in your ${HOME}/.zshrc - or ${HOME}/.bashrc - file):"
echo "  export ${ld_var}=$(pwd)/sox-code/src/.libs"
