import shutil

import pytest

import lhotse  # noqa: F401  # Adds the lhotse tools directory to PATH.

requires_sph2pipe = pytest.mark.skipif(
    shutil.which("sph2pipe") is None,
    reason="Requires sph2pipe; run 'lhotse install-sph2pipe' to install it.",
)
