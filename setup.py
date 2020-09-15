# coding=utf-8
import os
from pathlib import Path

from setuptools import find_packages, setup

project_root = Path(__file__).parent

install_requires = (project_root / 'requirements.txt').read_text().splitlines()
docs_require = (project_root / 'docs' / 'requirements.txt').read_text().splitlines()
tests_require = ['pytest==5.4.3', 'flake8==3.8.3', 'coverage==5.1']
dev_requires = docs_require + tests_require + ['jupyterlab', 'matplotlib', 'isort']

if os.environ.get('READTHEDOCS', False):
    # When building documentation, omit torchaudio installation and mock it instead.
    # This works around the inability to install libsoundfile1 in read-the-docs env,
    # which caused the documentation builds to silently crash.
    install_requires = [req for req in install_requires if not req.startswith('torchaudio')]

setup(
    name='lhotse',
    version='0.1',
    python_requires='>=3.7.0',
    description='Data preparation for speech processing models training.',
    author='The Lhotse Development Team',
    author_email="pzelasko@jhu.edu",
    long_description=(project_root / 'README.md').read_text(),
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
        'dev': docs_require + tests_require
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
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
