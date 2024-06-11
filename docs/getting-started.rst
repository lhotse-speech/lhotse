Getting started
===============

.. image:: logo.png

Lhotse is a Python library aiming to make speech and audio data preparation flexible and accessible to a wider community. Alongside `k2`_, it is a part of the next generation `Kaldi`_ speech processing library.


About
-----

Main goals
**********

* Attract a wider community to speech processing tasks with a **Python-centric design**.

* Accommodate experienced Kaldi users with an **expressive command-line interface**.

* Provide **standard data preparation recipes** for commonly used corpora.

* Provide **PyTorch Dataset classes** for speech and audio related tasks.

* Flexible data preparation for model training with the notion of **audio cuts**.

* **Efficiency**, especially in terms of I/O bandwidth and storage capacity.

Tutorials
*********

We currently have the following tutorials available in `examples` directory:

* Basic complete Lhotse workflow |tutorial00|

* Transforming data with Cuts |tutorial01|

* WebDataset integration |tutorial02|

* How to combine multiple datasets |tutorial03|

* Lhotse Shar: storage format optimized for sequential I/O and modularity |tutorial04|

.. |tutorial00| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/00-basic-workflow.ipynb
.. |tutorial01| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/01-cut-python-api.ipynb
.. |tutorial02| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/02-webdataset-integration.ipynb
.. |tutorial03| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/03-combining-datasets.ipynb
.. |tutorial04| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/04-lhotse-shar.ipynb


Examples of use
***************

Check out the following links to see how Lhotse is being put to use:

* `Icefall recipes`_: where k2 and Lhotse meet.

* Minimal ESPnet+Lhotse example: |mini librispeech colab notebook|

 .. |mini librispeech colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1HKSYPsWx_HoCdrnLpaPdYj5zwlPsM3NH

Main ideas
**********

Like Kaldi, Lhotse provides standard data preparation recipes, but extends that with a seamless PyTorch integration through task-specific Dataset classes. The data and meta-data are represented in human-readable text manifests and exposed to the user through convenient Python classes.

.. image:: lhotse-concept-graph.png

Lhotse introduces the notion of audio cuts, designed to ease the training data construction with operations such as mixing, truncation and padding that are performed on-the-fly to minimize the amount of storage required. Data augmentation and feature extraction are supported both in pre-computed mode, with highly-compressed feature matrices stored on disk, and on-the-fly mode that computes the transformations upon request. Additionally, Lhotse introduces feature-space cut mixing to make the best of both worlds.

.. image:: lhotse-cut-illustration.png

Installation
------------

Lhotse supports Python version 3.7 and later.

Pip
***

Lhotse is available on PyPI::

    pip install lhotse

To install the latest, unreleased version, do::

    pip install git+https://github.com/lhotse-speech/lhotse

*Hint: for up to 50% faster reading of JSONL manifests, use:* ``pip install lhotse[orjson]`` *to leverage the* `orjson`_ *library.*

Development installation
************************

For development installation, you can fork/clone the GitHub repo and install with pip::

    git clone https://github.com/lhotse-speech/lhotse
    cd lhotse
    pip install -e '.[dev]'
    pre-commit install  # installs pre-commit hooks with style checks

    # Running unit tests
    pytest test

    # Running linter checks
    pre-commit run

This is an editable installation (``-e`` option), meaning that your changes to the source code are automatically
reflected when importing lhotse (no re-install needed). The ``[dev]`` part means you're installing extra dependencies
that are used to run tests, build documentation or launch jupyter notebooks.

Environment variables
*********************

Lhotse uses several environment variables to customize it's behavior. They are as follows:

* ``LHOTSE_REQUIRE_TORCHAUDIO`` - when it's set and not any of ``1|True|true|yes``, we'll not check for torchaudio being installed and remove it from the requirements. It will disable many functionalities of Lhotse but the basic capabilities will remain (including reading audio with `soundfile`).

* ``LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE`` - used when we load audio from a file and receive a different number of samples than declared in ``Recording.num_samples``. This is sometimes necessary because different codecs (or even different versions of the same codec) may use different padding when decoding compressed audio. Typically values up to 0.1, or even 0.3 (second) are still reasonable, and anything beyond that indicates a serious issue.

* ``LHOTSE_AUDIO_BACKEND`` - may be set to any of the values returned from CLI ``lhotse list-audio-backends`` to override the default behavior of trial-and-error and always use a specific audio backend.

* ``LHOTSE_AUDIO_LOADING_EXCEPTION_VERBOSE`` - when set to 1 we'll emit full exception stack traces when every available audio backend fails to load a given file (they might be very large).

* ``LHOTSE_DILL_ENABLED`` - when it's set to ``1|True|true|yes``, we will enable ``dill``-based serialization of ``CutSet`` and ``Sampler`` across processes (it's disabled by default even when ``dill`` is installed).

* ``LHOTSE_LEGACY_OPUS_LOADING`` - (``=1``) reverts to a legacy OPUS loading mechanism that triggered a new ffmpeg subprocess for each OPUS file.

* ``LHOTSE_PREPARING_RELEASE`` - used internally by developers when releasing a new version of Lhotse.

* ``TORCHAUDIO_USE_BACKEND_DISPATCHER`` - when set to 1 and torchaudio version is below 2.1, we'll enable the experimental ffmpeg backend of torchaudio.

* ``AIS_ENDPOINT`` is read by AIStore client to determine AIStore endpoint URL. Required for AIStore dataloading.

* ``RANK``, ``WORLD_SIZE``, ``WORKER``, and ``NUM_WORKERS`` are internally used to inform Lhotse Shar dataloading subprocesses.

* ``READTHEDOCS`` is internally used for documentation builds.


Optional dependencies
*********************

**Other pip packages.** You can leverage optional features of Lhotse by installing the relevant supporting package like this: ``pip install lhotse[package_name]``. The supported optional packages include:

* ``pip install lhotse[kaldi]`` for a maximal feature set related to Kaldi compatibility. It includes libraries such as ``kaldi_native_io`` (a more efficient variant of ``kaldi_io``) and ``kaldifeat`` that port some of Kaldi functionality into Python.

* ``pip install lhotse[orjson]`` for up to 50% faster reading of JSONL manifests.

* ``pip install lhotse[webdataset]``. We support "compiling" your data into WebDataset tarball format for more effective IO. You can still interact with the data as if it was a regular lazy CutSet. To learn more, check out the following tutorial: |tutorial02|

* ``pip install h5py`` if you want to extract speech features and store them as HDF5 arrays.

* ``pip install dill``. When ``dill`` is installed, we'll use it to pickle CutSet that uses a lambda function in calls such as ``.map`` or ``.filter``. This is helpful in PyTorch DataLoader with ``num_jobs>0``. Without ``dill``, depending on your environment, you'll see an exception or a hanging script.

* ``pip install aistore`` to read manifests, tar fles, and other data from AIStore using AIStore-supported URLs (set ``AIS_ENDPOINT`` environment variable to activate it). See |AIStore| for more details.

* ``pip install smart_open`` to read and write manifests and data in any location supported by ``smart_open`` (e.g. cloud, http).

* ``pip install opensmile`` for feature extraction using the OpenSmile toolkit's Python wrapper.

**sph2pipe.** For reading older LDC SPHERE (.sph) audio files that are compressed with codecs unsupported by ffmpeg and sox, please run::

    # CLI
    lhotse install-sph2pipe

    # Python
    from lhotse.tools import install_sph2pipe
    install_sph2pipe()

It will download it to ``~/.lhotse/tools``, compile it, and auto-register in ``PATH``. The program should be automatically detected and used by Lhotse.


Examples
--------

We have example recipes showing how to prepare data and load it in Python as a PyTorch ``Dataset``.
They are located in the ``examples`` directory.

A short snippet to show how Lhotse can make audio data preparation quick and easy:

.. code-block::

    from torch.utils.data import DataLoader
    from lhotse import CutSet, Fbank
    from lhotse.dataset import VadDataset, SimpleCutSampler
    from lhotse.recipes import prepare_switchboard

    # Prepare data manifests from a raw corpus distribution.
    # The RecordingSet describes the metadata about audio recordings;
    # the sampling rate, number of channels, duration, etc.
    # The SupervisionSet describes metadata about supervision segments:
    # the transcript, speaker, language, and so on.
    swbd = prepare_switchboard('/export/corpora3/LDC/LDC97S62')

    # CutSet is the workhorse of Lhotse, allowing for flexible data manipulation.
    # We create 5-second cuts by traversing SWBD recordings in windows.
    # No audio data is actually loaded into memory or stored to disk at this point.
    cuts = CutSet.from_manifests(
        recordings=swbd['recordings'],
        supervisions=swbd['supervisions']
    ).cut_into_windows(duration=5)

    # We compute the log-Mel filter energies and store them on disk;
    # Then, we pad the cuts to 5 seconds to ensure all cuts are of equal length,
    # as the last window in each recording might have a shorter duration.
    # The padding will be performed once the features are loaded into memory.
    cuts = cuts.compute_and_store_features(
        extractor=Fbank(),
        storage_path='feats',
        num_jobs=8
    ).pad(duration=5.0)

    # Construct a Pytorch Dataset class for Voice Activity Detection task:
    dataset = VadDataset()
    sampler = SimpleCutSampler(cuts, max_duration=300)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=None)
    batch = next(iter(dataloader))

The ``VadDataset`` will yield a batch with pairs of feature and supervision tensors such as the following -
the speech starts roughly at the first second (100 frames):

.. image:: vad_sample.png


.. _k2: https://github.com/k2-fsa/k2
.. _Kaldi: https://github.com/kaldi-asr/kaldi
.. _Icefall recipes: https://github.com/k2-fsa/icefall
.. _orjson: https://pypi.org/project/orjson/
.. _AIStore: https://aiatscale.org
