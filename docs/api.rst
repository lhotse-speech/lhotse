API Reference
=============

This page contains a comprehensive list of all classes and functions within `lhotse`.

Audio loading, saving, and manifests
-------------------

Data structures and utilities used for describing and manipulating audio recordings.

.. automodule:: lhotse.audio
  :members:
  :inherited-members:

Supervision manifests
---------------------

Data structures used for describing supervisions in a dataset.

.. automodule:: lhotse.supervision
  :members:
  :inherited-members:

Lhotse Shar -- sequential storage
---------------------------------

Documentation for Lhotse Shar multi-tarfile sequential I/O format.

Lhotse Shar readers
*******************

.. automodule:: lhotse.shar.readers
  :members:
  :inherited-members:

Lhotse Shar writers
*******************

.. automodule:: lhotse.shar.writers
  :members:
  :inherited-members:

Feature extraction and manifests
--------------------------------

Data structures and tools used for feature extraction and description.

Features API - extractor and manifests
**************************************

.. automodule:: lhotse.features.base
  :members:
  :inherited-members:

Lhotse's feature extractors
***************************

.. autoclass:: lhotse.features.kaldi.extractors.Fbank

.. autoclass:: lhotse.features.kaldi.extractors.Mfcc

Kaldi feature extractors as network layers
******************************************

.. automodule:: lhotse.features.kaldi.layers
    :members:
    :inherited-members:

Torchaudio feature extractors
*****************************

.. automodule:: lhotse.features.fbank
  :members:
  :inherited-members:

.. automodule:: lhotse.features.mfcc
  :members:
  :inherited-members:

.. automodule:: lhotse.features.spectrogram
  :members:
  :inherited-members:

Librosa filter-bank
*******************

.. automodule:: lhotse.features.librosa_fbank
    :members:
    :inherited-members:

Feature storage
***************

.. automodule:: lhotse.features.io
  :members:
  :inherited-members:

Feature-domain mixing
*********************

.. automodule:: lhotse.features.mixer
  :members:
  :inherited-members:

Augmentation
------------

.. automodule:: lhotse.augmentation
  :members:
  :inherited-members:

Cuts
----

Data structures and tools used to create training/testing examples.

.. automodule:: lhotse.cut
  :members:
  :inherited-members:

Recipes
-------

Convenience methods used to prepare recording and supervision manifests for standard corpora.

.. automodule:: lhotse.recipes
  :members:
  :inherited-members:

Kaldi conversion
----------------

Convenience methods used to interact with Kaldi data directories.

.. automodule:: lhotse.kaldi
  :members:
  :inherited-members:

Others
------

Helper methods used throughout the codebase.

.. automodule:: lhotse.manipulation
  :members:
  :inherited-members:
