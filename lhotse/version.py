# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
# (this snippet is borrowed from scikit-learn's __init__.py)
#
# Generic release markers:
#   X.Y.0   # For first release after an increment in Y
#   X.Y.Z   # For bugfix releases
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
# >>>>>>>>>>>>>>>>>>>>>> CAUTION <<<<<<<<<<<<<<<<<<<<<<<
# > For public releases, set IS_DEV_VERSION = False    <
# > For non-public releases, set IS_DEV_VERSION = True <
# >>>>>>>>>>>>>>>>>>>>>> CAUTION <<<<<<<<<<<<<<<<<<<<<<<
#
# Important: do not add non-standard-lib imports to this file,
# otherwise it might break the installation / release process.

from pathlib import Path
from subprocess import DEVNULL, PIPE, run

MAJOR_VERSION = 0
MINOR_VERSION = 8
PATCH_VERSION = 0
IS_DEV_VERSION = False # False = public release, True = otherwise


project_root = Path(__file__).parent.parent


def discover_lhotse_version():
    """
    Scans Lhotse source code to determine the current version.
    When development version is detected, it queries git for the commit hash
    to append it as a local version identifier.
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


LHOTSE_VERSION = discover_lhotse_version()
