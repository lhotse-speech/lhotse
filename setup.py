# coding=utf-8
import os
from pathlib import Path
from subprocess import PIPE, run

from setuptools import find_packages, setup

project_root = Path(__file__).parent

install_requires = (project_root / 'requirements.txt').read_text().splitlines()
docs_require = (project_root / 'docs' / 'requirements.txt').read_text().splitlines()
tests_require = ['pytest==5.4.3', 'flake8==3.8.3', 'coverage==5.1', 'hypothesis==5.41.2']
arrow_requires = ['pyarrow>=4.0.0', 'pandas>=1.0.0']
dev_requires = sorted(docs_require + tests_require + ['jupyterlab', 'matplotlib', 'isort'])
all_requires = sorted(dev_requires + arrow_requires)

if os.environ.get('READTHEDOCS', False):
    # When building documentation, omit torchaudio installation and mock it instead.
    # This works around the inability to install libsoundfile1 in read-the-docs env,
    # which caused the documentation builds to silently crash.
    install_requires = [
        req for req in install_requires
        if not any(
            req.startswith(dep) for dep in ['torchaudio', 'SoundFile']
        )]

try:
    git_commit = run(['git', 'rev-parse', '--short', 'HEAD'], check=True, stdout=PIPE).stdout.decode().rstrip(
        '\n').strip()
    dirty_commit = len(
        run(['git', 'diff', '--shortstat'], check=True, stdout=PIPE).stdout.decode().rstrip('\n').strip()) > 0
    git_commit = git_commit + '-dirty' if dirty_commit else git_commit + '-clean'
except Exception:
    git_commit = ''
# See the format https://packaging.python.org/guides/distributing-packages-using-setuptools/#local-version-identifiers
dev_version = '.dev-' + git_commit

if os.environ.get('RELEASE', False):
    dev_version = ''

setup(
    name='lhotse',
    version='0.8.0' + dev_version,
    python_requires='>=3.6.0',
    description='Data preparation for speech processing models training.',
    author='The Lhotse Development Team',
    author_email="pzelasko@jhu.edu",
    long_description=(project_root / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type="text/markdown",
    license='Apache-2.0 License',
    packages=find_packages(),
    # The line below makes every script in the list an executable that's inserted in PATH
    # as long as the virtualenv/conda env is active; they can be used like any other shell program
    scripts=['lhotse/bin/lhotse'],
    install_requires=install_requires,
    extras_require={
        'docs': docs_require,
        'tests': tests_require,
        'arrow': arrow_requires,
        'dev': docs_require + tests_require,
        'all': all_requires
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed"
    ],
)
