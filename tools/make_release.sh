#!/usr/bin/env bash

set -eou pipefail  # "strict" mode

set -x  # show executed commands

# Clean up old builds.
rm -rf dist/ build/ lhotse.egg_info/

export LHOTSE_PREPARING_RELEASE=1

# Build wheels and package current source code
python setup.py sdist bdist_wheel

set +x  # stop showing the executed commands

echo
echo "Lhotse is packaged SUCCESSFULLY!"
echo
echo "To upload a TEST RELEASE to testpypi (recommended):"
echo "  twine upload -r testpypi dist/*"
echo
echo "To upload a PUBLIC RELEASE to pypi:"
echo "  twine upload dist/*"
