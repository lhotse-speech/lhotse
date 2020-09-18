PyTorch Datasets
================

.. caution::
    Lhotse datasets are still very much in the works and are subject to breaking changes.

We supply subclasses of the :class:`torch.data.Dataset` for various audio/speech tasks.
These datasets are created from :class:`CutSet` objects and load the features from disk into memory on-the-fly.
Each dataset accepts an optional ``root_dir`` argument which is used as a prefix for the paths to features and audio.

Currently, we provide the following:

.. automodule:: lhotse.dataset.diarization
  :members:
  :noindex:

.. automodule:: lhotse.dataset.unsupervised
  :members:
  :noindex:

.. automodule:: lhotse.dataset.speech_recognition
  :members:
  :noindex:

.. autoclass:: lhotse.dataset.source_separation.DynamicallyMixedSourceSeparationDataset
  :members:
  :noindex:

.. autoclass:: lhotse.dataset.source_separation.PreMixedSourceSeparationDataset
  :members:
  :noindex:

.. automodule:: lhotse.dataset.vad
  :members:
  :noindex:
