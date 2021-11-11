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
from distutils.version import LooseVersion
from pathlib import Path
from subprocess import DEVNULL, PIPE, run

from setuptools import find_packages, setup

project_root = Path(__file__).parent

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# NOTE: REMEMBER TO UPDATE THE FALLBACK VERSION IN lhotse/__init__.py WHEN RELEASING #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
MAJOR_VERSION = 0
MINOR_VERSION = 12
PATCH_VERSION = 0
IS_DEV_VERSION = True  # False = public release, True = otherwise


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
    # A matrix of compatible torch and torchaudio versions.
    # If the user already installed torch, we'll try to find the compatible
    # torchaudio version. If they haven't installed torch, we'll just install
    # the latest torch and torchaudio.
    # This code is partially borrowed from ESPnet's setup.py.
    import torch

    installed_ver = LooseVersion(torch.__version__)
    major, minor, patch, *_ = installed_ver.version
    clean_vstr = f"{major}.{minor}.{patch}"
    debug_string = f"""(full installed PyTorch version string: '{installed_ver.vstring}', 
    PyTorch version string used for torchaudio version lookup: '{clean_vstr}')"""

    version_map = {
        "1.10.0": "0.10.0",
        "1.9.1": "0.9.1",
        "1.9.0": "0.9.0",
        "1.8.2": "0.8.2",
        "1.8.1": "0.8.1",
        "1.8.0": "0.8.0",
        "1.7.1": "0.7.2",
    }

    # Nice error message for PyTorch versions that are too new.
    latest_supported_ver = LooseVersion(next(iter(version_map)))
    # HACK: we recreate the LooseVersion of the current installed PyTorch version
    #       without any version suffixes. This solves the issue that:
    #
    #           >>> LooseVersion("1.10.0+cpu") > LooseVersion("1.10.0") == True
    #
    #       whereas:
    #
    #           >>> LooseVersion("1.10.0") > LooseVersion("1.10.0") == False
    #
    installed_ver_nosufix = LooseVersion(clean_vstr)
    if installed_ver_nosufix > latest_supported_ver:
        raise NotImplementedError(
            f"PyTorch version > {latest_supported_ver.vstring} "
            f"is not yet supported {debug_string}."
        )

    # Nice error message for PyTorch versions that are too old.
    earliest_supported_ver = LooseVersion(next(iter(reversed(version_map))))
    if installed_ver < earliest_supported_ver:
        raise NotImplementedError(
            f"PyTorch version < {earliest_supported_ver.vstring} is not supported: "
            f"please update your PyTorch version {debug_string}."
        )

    # Nice error message for PyTorch versions that should be supported but we might have missed them?
    if clean_vstr not in version_map:
        raise ValueError(
            f"Unknown PyTorch version {debug_string}."
            f"Please set up an issue at https://github.com/lhotse-speech/lhotse/issues"
        )

    # Everything worked and we can resolve the right PyTorch version.
    matching_torchaudio_ver = version_map[clean_vstr]
    install_requires.append(f"torchaudio=={matching_torchaudio_ver}")
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
