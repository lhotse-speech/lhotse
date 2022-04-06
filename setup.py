# coding=utf-8
#                `-:/++oooooooooooooooooooooooooooooooooooooooooooooo++/:-.
#            `-+sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssso:`
#          .+ssosyyhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhyyosso-
#         /yosyhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhysos+`
#       `osoyhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhsoy.
#       osohhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhsoy`
#      /h+hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhoso
#      yoyhhhhhhhhhhyooyhhhhhysyhhhhyhhhyysyhhhhyhhhhhyyhhysosyhhhysyyyyhhhhhhhhhhh+h`
#     `h+hhhhhhhhhhhs  +hhhhh.`:hh+.-yy+.```./y/.----..:o:`   -+y+```..-hhhhhhhhhhh+h:
#     -h+hhhhhhhhhhh+  shhhhy  /yh/  yo..--.` .+//.  /os/  /oosso  -::::hhhhhhhhhhh+h/
#     -h+hhhhhhhhhhh: `hhhhho  .-:. `y++yhhho  -hh/  yhho` .:/shy  /oosshhhhhhhhhhh+h/
#     -h+hhhhhhhhhhh. .hhhhh/  .--` `+ `shhhy` .hh- `hhhho:.. `/o  ```./hhhhhhhhhhh+h+
#     -h+hhhhhhhhhhy` `+sssy:  yhh- `y` ./+/. `ohh. .hhhosyhy- .+  ossssyhhhhhhhhhh+h+
#     -h+hhhhhhhhhh+   ````o:``hhh:``hs-`` ``-ohhh. .hhhs---.`.o+   ``  ohhhhhhhhhh+h+
#     -h+hhhhhhhhhhhooosssshysyhhhysyhhhysosyhhhhhsoshhhhyo++oyhyo++++ooyhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhhhhyhhyhhhhhhhhyyhhhhyhhhyyyhhhhhhhhhhhyhhhhhyhyhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhsyhyyyyyyyyyyhhsyyyyyyyysyyyhhyyyyhhhyyyyyshhsyyyhhhhhhhhhhh+h+
#     -h+hhhhhhhhhyysyhoyyosyosyosyosyosyosyoyyosyoyysysohhoyyosssyyoyyoyshhhhhhhhh+h+
#     -h+hhhhhhyyysyosyoososy+ssosyyyyoss+ss+ssossoss/ssoyyoss+so+ss+sooyoyyyhhhhhh+h+
#     -h+hhhhhhhhhhhhhhsyhsyyyyyshhh:`-ohyyysyyyyyyhhsyyshhhyysyyyhhsyyshhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhyhhyhhyhhhy/`    /hhhhoshhhyhhhhhhhhhhhyhhyhhyhyhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhhhhhhhhhho` /y/ `:-oo-  +s+yhhhhhy++yhhhhhhhhhhhhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhhhhhhhhy/ `/hhh` .y/.     :::+o/`   `:ohhhhhhhhhhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhhhhys+-` `ohhhy`syos/:.   sho:-.` `--  .yhhhhhhhhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhhhh-   `-shhhhssh+.`s:y- ./yssy--oy/` .s:yhhhhhhhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhhhhs:  `-yyhhhhhhhy-` ohhyo:syyhosyh/  :yh+.ohhhhhhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhhhyo/`   oyhhhhhyso/`   yhh/oyo/`+/.so  :yhy-  .+yhhhhhhhhhhhhhh+h+
#     -h+hhhhhhhhhhs/`-    .yhhhhhh.`    `:hhhs`sho..-.-``-ss+hys:` .yyyhhhhhhhhhhh+h+
#     -h+hhhhhhhhy/.-:.  -oyhhhhhhh`     /yhhhy .hhy:. `+o+./yhhhhyo//-`/yhhhhhhhhh+h+
#     -h+hhhhhhho:/s+-`-oyhhhhhhhhh/     `-hhhy` yhh.  /-` -yhhhhhhhhy+  `+shhhhhhh+h+
#     -h+hhhhhs++ysososhhyosyhhhhhhy.    :shhhy  shh/  .  `yhhhhhhhhhhy:    -shhhhh+h/
#     .h+hhhysoyhsyhhhhhho.`./yhhhhhy- -syhhhhh-`hhhy.    -hhhhhhhhhhhhso :o:-/yhhh+h/
#     `h+hhhyyhhhhhhhhhhhhy+` :yhhhhhs-hhhhhhhh+-hhhh+  `+yhhhhhhhhhhhhhy/yhhysoyhh+h-
#      soshhhhhhhhhhhhhhhhhhs``-ohhhhhshhhhhhhhyohhhhh+.`-+shhhhhhhhhhhhhhhhhhhhhhh+h`
#      :h+hhhhhhhhhhhhhhhhhhhsss/ohhhhhhhhhhhhhhyhhhhhho.ohhhhhhhhhhhhhhhhhhhhhhhhoy+
#       oyohhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhs+yhhhhhhhhhhhhhhhhhhhhhhhhsss`
#       `+soyhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhyhhhhhhhhhhhhhhhhhhhhhhhhyoss`
#         :sooyhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhysos/`
#          `/ssssyyhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhyyssss+.
#             ./osssssssssssssssssssssssssssssssssssssssssssssssssssssssssss/-`
#                 .-:://++++++++++++++++++++++++++++++++++++++++++++///:-.`
import os
from pathlib import Path
from subprocess import DEVNULL, PIPE, run

from setuptools import find_packages, setup

project_root = Path(__file__).parent

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# NOTE: REMEMBER TO UPDATE THE FALLBACK VERSION IN lhotse/__init__.py WHEN RELEASING #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
MAJOR_VERSION = 1
MINOR_VERSION = 0
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
    (project_root / "lhotse" / "version.py").write_text(f'__version__ = "{version}"')


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
    # If the user already installed PyTorch, make sure he has torchaudio too.
    # Otherwise, we'll just install the latest versions from PyPI for the user.
    import torch

    try:
        import torchaudio
    except ImportError:
        raise ValueError(
            "We detected that you have already installed PyTorch, but haven't installed torchaudio. "
            "Unfortunately we can't detect the compatible torchaudio version for you; "
            "you will have to install it manually. "
            "For instructions, please refer either to https://pytorch.org/get-started/locally/ "
            "or https://github.com/pytorch/audio#dependencies"
        )
except ImportError:
    install_requires.extend(["torch", "torchaudio"])

docs_require = (project_root / "docs" / "requirements.txt").read_text().splitlines()
tests_require = [
    "pytest==5.4.3",
    "flake8==3.8.3",
    "coverage==5.1",
    "hypothesis==5.41.2",
    "black==22.3.0",
]
dev_requires = sorted(
    docs_require + tests_require + ["jupyterlab", "matplotlib", "isort"]
)
orjson_require = ["orjson>=3.6.6"]
all_requires = sorted(dev_requires + orjson_require)

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
    packages=find_packages(exclude=["test", "test.*"]),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "lhotse=lhotse.bin.lhotse:cli",
        ]
    },
    install_requires=install_requires,
    extras_require={
        "orjson": orjson_require,
        "docs": docs_require,
        "tests": tests_require,
        "dev": dev_requires,
        "all": all_requires,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
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
