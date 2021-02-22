PyTorch Datasets
================

Lhotse supports PyTorch’s dataset API, providing implementations for the ``Dataset`` and ``Sampler`` concepts.
They can be used together with the standard ``DataLoader`` class for efficient mini-batch collection with multiple parallel readers and pre-fetching.

A quick re-cap of PyTorch’s data API
------------------------------------

PyTorch defines the Dataset class that is responsible for reading the data from disk/memory/Internet/database/etc., and converting it to tensors that can be used for network training or inference.
These ``Dataset``'s are typically „map-style” datasets which are given an index (or a list of indices) and return the corresponding data samples.

The selection of indices is performed by the ``Sampler`` class.
``Sampler``, knowing the length (number of items) in a ``Dataset``, can use various strategies to determine the order of elements to read (e.g. sequential reads, or random reads).

More details about the data pipeline API in PyTorch can be found `here <https://pytorch.org/docs/stable/data.html>`_.

About Lhotse’s Datasets and Samplers
------------------------------------

Lhotse provides a number of utilities that make it simpler to define ``Dataset``'s for speech processing tasks.
:class:`~lhotse.cut.CutSet` is the base data structure that is used to initialize the ``Dataset`` class.
This makes it possible to manipulate the speech data in convenient ways - pad, mix, concatenate, augment, compute features, look up the supervision information, etc.

Lhotse’s ``Dataset``'s will perform batching by themselves, because auto-collation in ``DataLoader`` is too limiting for speech data handling.
These ``Dataset``'s expect to be handed lists of element indices, so that they can collate the data *before* it is passed to the ``DataLoader`` (which must use ``batch_size=None``).
It allows for interesting collation methods - e.g. **padding the speech with noise recordings, or actual acoustic context**, rather than artificial zeroes; or **dynamic batch sizes**.

The items for mini-batch creation are selected by the ``Sampler``.
Lhotse defines ``Sampler`` classes that are initialized with :class:`~lhotse.cut.CutSet`'s, so that they can look up specific properties of an utterance to stratify the sampling.
For example, :class:`~lhotse.dataset.sampling.SingleCutSampler` has a defined ``max_frames`` attribute, and it will keep sampling cuts for a batch until they do not exceed the specified number of frames.
Another strategy — used in :class:`~lhotse.dataset.sampling.BucketingSampler` — will first group the cuts of similar durations into buckets, and then randomly select a bucket to draw the whole batch from.

For tasks where both input and output of the model are speech utterances, we can use the :class:`~lhotse.dataset.sampling.CutPairsSampler`, which accepts two :class:`~lhotse.cut.CutSet`'s and will match the cuts in them by their IDs.

Dataset's list
--------------

.. automodule:: lhotse.dataset.diarization
  :members:

.. automodule:: lhotse.dataset.unsupervised
  :members:

.. automodule:: lhotse.dataset.speech_recognition
  :members:

.. autoclass:: lhotse.dataset.speech_synthesis
  :members:

.. autoclass:: lhotse.dataset.source_separation.DynamicallyMixedSourceSeparationDataset
  :members:

.. autoclass:: lhotse.dataset.source_separation.PreMixedSourceSeparationDataset
  :members:

.. automodule:: lhotse.dataset.vad
  :members:

Sampler's list
--------------

.. automodule:: lhotse.dataset.sampling
  :members:
