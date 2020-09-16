API Reference
=============

This page contains a comprehensive list of all classes and functions within `lhotse`.

Datasets
--------

PyTorch Dataset wrappers for common tasks.

.. automodule:: lhotse.dataset.speech_recognition
  :members:

.. automodule:: lhotse.dataset.source_separation
  :members:

.. automodule:: lhotse.dataset.unsupervised
  :members:

.. automodule:: lhotse.dataset.vad
  :members:

Recording manifests
-------------------

Data structures used for describing audio recordings in a dataset.

.. automodule:: lhotse.audio
  :members:

Supervision manifests
---------------------

Data structures used for describing supervisions in a dataset.

.. automodule:: lhotse.supervision
  :members:

Feature extraction and manifests
--------------------------------

Data structures and tools used for feature extraction and description.

Torchaudio feature extractors
*****************************

.. automodule:: lhotse.features.fbank
  :members:

.. automodule:: lhotse.features.mfcc
  :members:

.. automodule:: lhotse.features.spectrogram
  :members:

Other feature extraction related classes
****************************************

.. automodule:: lhotse.features.base
  :members:

.. automodule:: lhotse.features.mixer
  :members:

Augmentation
------------

.. automodule:: lhotse.augmentation
  :members:

Cuts
----

Data structures and tools used to create training/testing examples.

.. automodule:: lhotse.cut
  :members:

Recipes
-------

Convenience methods used to prepare recording and supervision manifests for standard corpora.

.. automodule:: lhotse.recipes
  :members:

Kaldi conversion
----------------

Convenience methods used to interact with Kaldi data directories.

.. automodule:: lhotse.kaldi
  :members:

Others
------

Helper methods used throughout the codebase.

.. automodule:: lhotse.manipulation
  :members:

