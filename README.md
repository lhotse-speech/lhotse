<div align="center">
<img src="https://raw.githubusercontent.com/lhotse-speech/lhotse/master/docs/logo.png" width=376>

[![PyPI Status](https://badge.fury.io/py/lhotse.svg)](https://badge.fury.io/py/lhotse)
[![Python Versions](https://img.shields.io/pypi/pyversions/lhotse.svg)](https://pypi.org/project/lhotse/)
[![PyPI Status](https://pepy.tech/badge/lhotse)](https://pepy.tech/project/lhotse)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fpzelasko%2Flhotse%2Fbadge%3Fref%3Dmaster&style=flat)](https://actions-badge.atrox.dev/pzelasko/lhotse/goto?ref=master)
[![Documentation Status](https://readthedocs.org/projects/lhotse/badge/?version=latest)](https://lhotse.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/lhotse-speech/lhotse/branch/master/graph/badge.svg)](https://codecov.io/gh/lhotse-speech/lhotse)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhotse-speech/lhotse-speech.github.io/blob/master/notebooks/lhotse-introduction.ipynb)

</div>

# Lhotse

Lhotse is a Python library aiming to make speech and audio data preparation flexible and accessible to a wider community. Alongside [k2](https://github.com/k2-fsa/k2), it is a part of the next generation [Kaldi](https://github.com/kaldi-asr/kaldi) speech processing library.

⚠️ **Lhotse is not fully stable yet - while many features are already implemented, the APIs are still subject to change!** ⚠️

## About

### Main goals

- Attract a wider community to speech processing tasks with a **Python-centric design**.
- Accommodate experienced Kaldi users with an **expressive command-line interface**.
- Provide **standard data preparation recipes** for commonly used corpora.
- Provide **PyTorch Dataset classes** for speech and audio related tasks.
- Flexible data preparation for model training with the notion of **audio cuts**.
- **Efficiency**, especially in terms of I/O bandwidth and storage capacity.

### Tutorials

We currently have the following tutorials available in `examples` directory:
- Basic complete Lhotse workflow [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/00-basic-workflow.ipynb)
- Transforming data with Cuts [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/01-cut-python-api.ipynb)
- *(experimental)* WebDataset integration [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/02-webdataset-integration.ipynb)
- How to combine multiple datasets [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/03-combining-datasets.ipynb)

### Examples of use

Check out the following links to see how Lhotse is being put to use:
- [Icefall recipes](https://github.com/k2-fsa/icefall): where k2 and Lhotse meet.
- Minimal ESPnet+Lhotse example: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HKSYPsWx_HoCdrnLpaPdYj5zwlPsM3NH)

### Main ideas

Like Kaldi, Lhotse provides standard data preparation recipes, but extends that with a seamless PyTorch integration
through task-specific Dataset classes. The data and meta-data are represented in human-readable text manifests and
exposed to the user through convenient Python classes.

![image](https://raw.githubusercontent.com/lhotse-speech/lhotse/master/docs/lhotse-concept-graph.png)

Lhotse introduces the notion of audio cuts, designed to ease the training data construction with operations such as
mixing, truncation and padding that are performed on-the-fly to minimize the amount of storage required. Data
augmentation and feature extraction are supported both in pre-computed mode, with highly-compressed feature matrices
stored on disk, and on-the-fly mode that computes the transformations upon request. Additionally, Lhotse introduces
feature-space cut mixing to make the best of both worlds.

![image](https://raw.githubusercontent.com/lhotse-speech/lhotse/master/docs/lhotse-cut-illustration.png)

## Installation

Lhotse supports Python version 3.6 and later.

### Pip

Lhotse is available on PyPI:

    pip install lhotse

To install the latest, unreleased version, do:

    pip install git+https://github.com/lhotse-speech/lhotse

_Hint: for up to 50% faster reading of JSONL manifests, use: `pip install lhotse[orjson]` to leverage the [orjson](https://pypi.org/project/orjson/) library._

### Development installation

For development installation, you can fork/clone the GitHub repo and install with pip:

    git clone https://github.com/lhotse-speech/lhotse
    cd lhotse
    pip install -e '.[dev]'
    pre-commit install  # installs pre-commit hooks with style checks

    # Running unit tests
    pytest test

    # Running linter checks
    pre-commit run

This is an editable installation (`-e` option), meaning that your changes to the source code are automatically
reflected when importing lhotse (no re-install needed). The `[dev]` part means you're installing extra dependencies
that are used to run tests, build documentation or launch jupyter notebooks.

### Extra dependencies

For reading older LDC SPHERE (.sph) audio files that are compressed with codecs unsupported by ffmpeg and sox, please run:

    # CLI
    lhotse install-sph2pipe

    # Python
    from lhotse.tools import install_sph2pipe
    install_sph2pipe()

It will download it to `~/.lhotse/tools`, compile it, and auto-register in `PATH`. The program should be automatically detected and used by Lhotse.

## Examples

We have example recipes showing how to prepare data and load it in Python as a PyTorch `Dataset`.
They are located in the `examples` directory.

A short snippet to show how Lhotse can make audio data preparation quick and easy:

```python
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
```

The `VadDataset` will yield a batch with pairs of feature and supervision tensors such as the following - the speech
starts roughly at the first second (100 frames):

![image](https://raw.githubusercontent.com/lhotse-speech/lhotse/master/docs/vad_sample.png)
