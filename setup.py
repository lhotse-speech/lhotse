# coding=utf-8
import os
from distutils.version import LooseVersion
from pathlib import Path
from subprocess import DEVNULL, PIPE, run

from setuptools import find_packages, setup

project_root = Path(__file__).parent

MAJOR_VERSION = 0
MINOR_VERSION = 8
PATCH_VERSION = 0
IS_DEV_VERSION = False  # False = public release, True = otherwise


def discover_lhotse_version() -> str:
    """
    Scans Lhotse source code to determine the current version.
    When development version is detected, it queries git for the commit hash
    to append it as a local version identifier.

    Ideally this function would have been imported from lhotse.version and
    re-used when lhotse is imported to set the version, but it introduces
    a circular dependency. To avoid this, we write the determined version
    into project_root / 'lhotse' / 'version.py' during setup and read it
    from there later. If it's not detected, the version will be 0.0.0.dev.
    """

    version = f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}"
    if not IS_DEV_VERSION:
        # This is a PyPI public release -- return a clean version string.
        return version

    version = version + ".dev"

    # This is not a PyPI release -- try to read the git commit
    try:
        git_commit = (
            run(
                ["git", "rev-parse", "--short", "HEAD"],
                check=True,
                stdout=PIPE,
                stderr=DEVNULL,
            )
                .stdout.decode()
                .rstrip("\n")
                .strip()
        )
        dirty_commit = (
                len(
                    run(
                        ["git", "diff", "--shortstat"],
                        check=True,
                        stdout=PIPE,
                        stderr=DEVNULL,
                    )
                        .stdout.decode()
                        .rstrip("\n")
                        .strip()
                )
                > 0
        )
        git_commit = git_commit + ".dirty" if dirty_commit else git_commit + ".clean"
        source_version = f"+git.{git_commit}"
    except Exception:
        source_version = ".unknownsource"
    # See the format:
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#local-version-identifiers
    version = version + source_version

    return version


def mark_lhotse_version(version: str) -> None:
    (project_root / 'lhotse' / 'version.py').write_text(f'__version__ = "{version}"')


LHOTSE_VERSION = discover_lhotse_version()
mark_lhotse_version(LHOTSE_VERSION)


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
            f"Lhotse requires torch>=1.7.1 and torchaudio>=0.7.2 -- "
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
    packages=find_packages(exclude=["*test*"]),
    include_package_data=True,
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
