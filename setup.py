# coding=utf-8
import os
from distutils.version import LooseVersion
from pathlib import Path
from subprocess import PIPE, run

from setuptools import find_packages, setup

project_root = Path(__file__).parent


def discover_lhotse_version():
    """
    Scans Lhotse source code to determine the current version.
    When development version is detected, it queries git for the commit hash
    to append it as a local version identifier.
    """

    # Read the version from "lhotse.__version__"
    try:
        version_line = [
            line
            for line in (project_root / "lhotse" / "__init__.py")
            .read_text()
            .splitlines()
            if line.startswith("__version__ = ")
        ][0]
        version = version_line.split('"')[1]
    except IndexError:
        print("Unable to discover Lhotse's version: specifying placeholder 0.0.0.dev0")
        version = "0.0.0.dev0"

    # .dev suffix is only going to be removed for public releases
    if ".dev" not in version:
        return version

    # This is not a PyPI release -- try to read the git commit
    try:
        git_commit = (
            run(["git", "rev-parse", "--short", "HEAD"], check=True, stdout=PIPE)
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
        dirty_commit = (
            len(
                run(["git", "diff", "--shortstat"], check=True, stdout=PIPE)
                .stdout.decode()
                .rstrip("\n")
                .strip()
            )
            > 0
        )
        git_commit = git_commit + ".dirty" if dirty_commit else git_commit + ".clean"
    except Exception:
        git_commit = ".unknowncommit"
    # See the format:
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#local-version-identifiers
    version = version + '+git.' + git_commit

    return version


LHOTSE_VERSION = discover_lhotse_version()


install_requires = [
    "audioread>=2.1.9",
    "SoundFile>=0.10",
    "click>=7.1.1",
    "cytoolz>=0.10.1",
    "dataclasses",
    "h5py>=2.10.0",
    "intervaltree>= 3.1.0",
    "lilcom>=1.1.0",
    "numpy>=1.18.1",
    "packaging",
    "pyyaml>=5.3.1",
    "tqdm",
]

try:
    # A matrix of compatible torch and torchaudio versions.
    # If the user already installed torch, we'll try to find the compatible
    # torchaudio version. If they haven't installed torch, we'll just install
    # the latest torch and torchaudio.
    # This code is partially borrowed from ESPnet's setup.py.
    import torch

    torch_ver = LooseVersion(torch.__version__)
    if torch_ver >= LooseVersion("1.9.1"):
        raise NotImplementedError("Not yet supported")
    elif torch_ver >= LooseVersion("1.9.0"):
        install_requires.append("torchaudio==0.9.0")
    elif torch_ver >= LooseVersion("1.8.1"):
        install_requires.append("torchaudio==0.8.1")
    elif torch_ver >= LooseVersion("1.8.0"):
        install_requires.append("torchaudio==0.8.0")
    elif torch_ver >= LooseVersion("1.7.1"):
        install_requires.append("torchaudio==0.7.2")
    else:
        raise ValueError(
            f"Lhotse requires torch>=1.7.1 and torchaudio 1.7.2 -- "
            f"please update your torch (detected version: {torch_ver})."
        )
except ImportError:
    install_requires.extend(["torch", "torchaudio"])

docs_require = (project_root / "docs" / "requirements.txt").read_text().splitlines()
tests_require = [
    "pytest==5.4.3",
    "flake8==3.8.3",
    "coverage==5.1",
    "hypothesis==5.41.2",
]
dev_requires = sorted(
    docs_require + tests_require + ["jupyterlab", "matplotlib", "isort"]
)
all_requires = sorted(dev_requires)

if os.environ.get("READTHEDOCS", False):
    # When building documentation, omit torchaudio installation and mock it instead.
    # This works around the inability to install libsoundfile1 in read-the-docs env,
    # which caused the documentation builds to silently crash.
    install_requires = [
        req
        for req in install_requires
        if not any(req.startswith(dep) for dep in ["torchaudio", "SoundFile"])
    ]

setup(
    name="lhotse",
    version=LHOTSE_VERSION,
    python_requires=">=3.6.0",
    description="Data preparation for speech processing models training.",
    author="The Lhotse Development Team",
    author_email="pzelasko@jhu.edu",
    long_description=(project_root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="Apache-2.0 License",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "lhotse=lhotse.bin.lhotse:cli",
        ]
    },
    install_requires=install_requires,
    extras_require={
        "docs": docs_require,
        "tests": tests_require,
        "dev": dev_requires,
        "all": all_requires,
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
        "Typing :: Typed",
    ],
)
