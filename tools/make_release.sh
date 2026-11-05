#!/usr/bin/env bash

set -eou pipefail  # "strict" mode

set -x  # show executed commands

# Resolve paths relative to this script, even when invoked from another directory.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

# Reject stale dependency metadata before removing existing artifacts.
uv lock --check

# Clean up old builds.
rm -rf -- dist/ build/ lhotse.egg-info/

# Build wheels and package current source code
LHOTSE_PREPARING_RELEASE=1 uv build --no-sources

set +x  # stop showing the executed commands

echo
echo "Lhotse is packaged SUCCESSFULLY!"
echo
echo "To upload a TEST RELEASE to testpypi (recommended):"
echo "  uv publish --publish-url https://test.pypi.org/legacy/ dist/*"
echo
echo "To upload a PUBLIC RELEASE to pypi:"
echo "  uv publish dist/*"
