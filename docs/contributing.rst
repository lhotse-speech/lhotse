Contributing
============

Development installation
------------------------

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ 0.13.7,
and CMake (native extensions may need source builds on the newest Python), then
clone the repository::

    git clone https://github.com/lhotse-speech/lhotse
    cd lhotse
    uv sync --locked
    uv run --locked pre-commit install

This creates an editable installation in ``.venv`` using Python 3.15.2 from
``.python-version``. The default development group selects the existing ``tests``
extra and contributor tools. To use another supported interpreter, pass
``--python 3.10`` to ``uv sync``. Set ``UV_PROJECT_ENVIRONMENT=.venv-py310``
to keep a separate environment instead of replacing ``.venv``.

Pip-based editable installation remains supported: ``python -m pip install -e '.[dev]'``.
The published ``dev`` and ``all`` extras include documentation and notebooks,
while the default uv development group selects tests and tooling. The repository's
pre-commit hooks require uv and use its locked tool versions.

Run the complete suite with the optional CI integrations::

    uv sync --locked --extra tests --group ci
    uv run --no-sync pytest test

Run formatting and lint checks in the isolated tooling environment::

    uv run --locked --only-group lint pre-commit run --all-files

Build documentation using the published ``docs`` extra::

    uv run --locked --no-default-groups --extra docs sphinx-build -b html docs docs/_build/html

Install FFmpeg, SoX, and libsndfile through your platform's package manager
(on macOS: ``brew install cmake ffmpeg sox libsndfile hdf5``), then install SPHERE support
with ``uv run --no-sync lhotse install-sph2pipe``. HDF5 headers are needed when
h5py has no wheel for the selected interpreter; set ``HDF5_DIR`` to its installation
prefix when necessary.

Dependency updates
------------------

Declare dependencies and CI constraints in ``pyproject.toml`` and commit changes
to the generated ``uv.lock``. Use ``uv lock --upgrade-package PACKAGE`` for a
targeted update and ``uv lock --check`` to verify freshness. CI installs with
``--locked``. The lock contains the Python-specific PyTorch combinations and
optional dependencies previously installed directly by the workflows. Linux uv
environments use CPU PyTorch wheels; downstream training applications choose
their own accelerator packages. Python interpreters and system libraries are
managed separately from the package lock.

The ``uv_lock`` GitHub workflow reports a check on every pull request to
``master``, so it can be required by branch protection. It validates the lock
when dependency/build metadata or the workflow itself changes; otherwise the
job succeeds without installing uv or running validation. Failure to inspect the
changed files fails the job. Manual runs and matching pushes to ``master``
always validate the lock.

A stale lock keeps the check failing and produces a ``uv-lock-update`` artifact
containing a refreshed lock and patch. Review and commit the update. The workflow
only needs read access and works for pull requests from forks.

Accelerator environments
------------------------

CUDA-enabled PyTorch works with the published Lhotse package. To let uv detect
the installed driver in a separate accelerator environment::

    uv venv --python 3.15.2 .venv-cuda
    uv pip install --python .venv-cuda --no-sources --torch-backend auto -e .

Use ``uv run --no-project --python .venv-cuda ...`` in that environment.
``uv sync`` against this repository applies its CPU wheel selection on Linux.
See uv's `PyTorch backend documentation
<https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection>`_
for explicit CUDA backend choices.

Releases
--------

To prepare a release, update ``VERSION``, run ``uv lock``, then invoke
``bash tools/make_release.sh``. It checks the lock and builds the wheel and source
archive with the release version, then prints ``uv publish`` commands for the
separate upload step.
