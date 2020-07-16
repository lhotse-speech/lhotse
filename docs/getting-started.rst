Getting started
===============

Lhotse is a library for audio data preparation. It's a part of K2, an effort to integrate the Kaldi ASR library
with PyTorch.

Main goals of Lhotse are:

- Attract a wider community to speech processing tasks with a **Python-centric design**.
- Accommodate experienced Kaldi users with an **expressive command-line interface**.
- Provide **standard data preparation recipes** for commonly used corpora.
- Provide **PyTorch Dataset classes** for speech and audio related tasks.
- Flexible data preparation for model training with the notion of **audio cuts**.
- **Efficiency**, especially in terms of I/O bandwidth and storage capacity.

Installation
------------

Pip
***

Once it's more stable, we will upload Lhotse to pip.

Development installation
************************

For development installation, you can fork/clone the GitHub repo and install with pip::

    git clone https://github.com/pzelasko/lhotse
    cd lhotse
    pip install -e '.[dev]'

    # Running unit tests
    pytest test

This is an editable installation (``-e`` option), meaning that your changes to the source code are automatically
reflected when importing lhotse (no re-install needed). The ``[dev]`` part means you're installing extra dependencies
that are used to run tests, build documentation or launch jupyter notebooks.


Examples
--------

We have example recipes showing how to prepare data and load it in Python as a PyTorch `Dataset`.
They are located in the `examples` directory. We will eventually demonstrate them here as well.

