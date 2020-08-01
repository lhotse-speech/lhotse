#!/usr/bin/env bash

set -eou pipefail

mkdir -p deps
pushd deps

if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Mac OS X detected - compiling libsox from source"
  if [ ! -f sox-code/src/.libs/libsox.dylib ]; then
    git clone --depth 1 git://git.code.sf.net/p/sox/code sox-code
    pushd sox-code
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
fi

if [ ! -d WavAugment ]; then
  git clone https://github.com/facebookresearch/WavAugment
fi
pushd WavAugment
git checkout de4cbd5490cc32c1cd21aab4422978e57c95b9dc
pip install .
popd

if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "To be able to import WavAugment in Python, you will need to set the following env variable (e.g. in your ${HOME}/.zshrc file):"
  echo "  export DYLD_FALLBACK_LIBRARY_PATH=$(pwd)/sox-code/src/.libs"
fi
