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

A typical Lhotse's dataset API usage might look like this:

.. code-block::

    from torch.utils.data import DataLoader
    from lhotse.dataset import SpeechRecognitionDataset, SingleCutSampler

    cuts = CutSet(...)
    dset = SpeechRecognitionDataset(cuts)
    sampler = SingleCutSampler(cuts, max_frames=50000)
    # Dataset performs batching by itself, so we have to indicate that
    # to the DataLoader with batch_size=None
    dloader = DataLoader(dset, sampler=sampler, batch_size=None, num_workers=1)
    for batch in dloader:
        ...  # process data

Batch I/O: pre-computed vs. on-the-fly features
-----------------------------------------------

Depending on the experimental setup and infrastructure, it might be more convenient to either pre-compute and store features like filter-bank energies for later use (as traditionally done in Kaldi/ESPnet/Espresso toolkits), or compute them dynamically during training ("on-the-fly").
Lhotse supports both modes of computation by introducing a class called :class:`~lhotse.dataset.input_strategies.BatchIO`.
It is accepted as an argument in most dataset classes, and defaults to :class:`~lhotse.dataset.input_strategies.PrecomputedFeatures`.
Other available choices are :class:`~lhotse.dataset.input_strategies.AudioSamples` for working with waveforms directly,
and :class:`~lhotse.dataset.input_strategies.OnTheFlyFeatures`, which wraps a :class:`~lhotse.features.base.FeatureExtractor` and applies it to a batch of recordings. These strategies automatically pad and collate the inputs, and provide information about the original signal lengths: as a number of frames/samples, binary mask, or start-end frame/sample pairs.

Which strategy to choose?
*************************

In general, pre-computed features can be greatly compressed (we achieve 70% size reduction with regard to un-compressed features), and so the I/O load on your computing infrastructure will be much smaller than if you read the recordings directly. This is especially valuable when working with network file systems (NFS) that are typically used in computational grids for storage. When your experiment is I/O bound, then it is best to use pre-computed features.

When I/O is not the issue, it might be preferable to use on-the-fly computation as it shouldn't require any prior steps to perform the network training. It is also simpler to apply a vast range of data augmentation methods in a fully randomized way (e.g. reverberation), although Lhotse provides support for approximate feature-domain signal mixing (e.g. for additive noise augmentation) to alleviate that to some extent.

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

Input strategies' list
----------------------

.. automodule:: lhotse.dataset.input_strategies
  :members:

Augmentation - transforms on cuts
---------------------------------

Some transforms, in order for us to have accurate information about the start and end times of the signal and its supervisions, have to be performed on cuts (or CutSets).

.. automodule:: lhotse.dataset.cut_transforms
  :members:

Augmentation - transforms on signals
------------------------------------

These transforms work directly on batches of collated feature matrices (or possibly raw waveforms, if applicable).

.. automodule:: lhotse.dataset.signal_transforms
  :members:


Collation utilities for building custom Datasets
------------------------------------------------

.. automodule:: lhotse.dataset.collation

Experimental: LhotseDataLoader
------------------------------

.. autoclass:: lhotse.dataset.dataloading.LhotseDataLoader