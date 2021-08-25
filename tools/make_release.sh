#!/usr/bin/env bash

set -eou pipefail  # "strict" mode

if grep '__version__' lhotse/__init__.py | grep '\.dev'; then
  echo 'It seems you are trying to release a development version of Lhotse.'
  echo 'To make a public release, first remove the .dev version specifier'
  echo 'in lhotse/__init__.py in variable called "__version__"'
  exit 1
fi

set -x  # show executed commands

# Clean up old builds.
rm -rf dist/ build/ lhotse.egg_info/

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
