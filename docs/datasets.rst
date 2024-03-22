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
For example, :class:`~lhotse.dataset.sampling.SimpleCutSampler` has a defined ``max_frames`` attribute, and it will keep sampling cuts for a batch until they do not exceed the specified number of frames.
Another strategy — used in :class:`~lhotse.dataset.sampling.BucketingSampler` — will first group the cuts of similar durations into buckets, and then randomly select a bucket to draw the whole batch from.

For tasks where both input and output of the model are speech utterances, we can use the :class:`~lhotse.dataset.sampling.CutPairsSampler`, which accepts two :class:`~lhotse.cut.CutSet`'s and will match the cuts in them by their IDs.

A typical Lhotse's dataset API usage might look like this:

.. code-block::

    from torch.utils.data import DataLoader
    from lhotse.dataset import SpeechRecognitionDataset, SimpleCutSampler

    cuts = CutSet(...)
    dset = SpeechRecognitionDataset(cuts)
    sampler = SimpleCutSampler(cuts, max_frames=50000)
    # Dataset performs batching by itself, so we have to indicate that
    # to the DataLoader with batch_size=None
    dloader = DataLoader(dset, sampler=sampler, batch_size=None, num_workers=1)
    for batch in dloader:
        ...  # process data

Restoring sampler's state: continuing the training
--------------------------------------------------

All :class:`~lhotse.dataset.sampling.CutSampler` types can save their progress and pick up from that checkpoint.
For consistency with PyTorch tensors, the relevant methods are called ``.state_dict()`` and ``.load_state_dict()``.
The following example illustrates how to save the sampler's state (pay attention to the last bit):

.. code-block::

    dataset = ...  # Some task-specific dataset initialization
    sampler = BucketingSampler(cuts, max_duration=200, shuffle=True, num_buckets=30)
    dloader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=4)
    global_step = 0
    for epoch in range(30):
        dloader.sampler.set_epoch(epoch)
        for batch in dloader:
            # ... processing forward, backward, etc.
            global_step += 1

            if global_step % 5000 == 0:
                state = dloader.sampler.state_dict()
                torch.save(state, f'sampler-ckpt-ep{epoch}-step{global_step}.pt')

In case that the training is ended abruptly and the epochs are very long
(10k+ steps, not uncommon with large datasets these days),
we can resume the training from where it left off like the following:

.. code-block::

    # Creating a vanilla sampler, we will read the previous progress into it.
    sampler = BucketingSampler(cuts, max_duration=200, shuffle=True, num_buckets=30)

    # Restore the sampler's state.
    state = torch.load('sampler-ckpt-ep5-step75000.pt')
    sampler.load_state_dict(state)

    dloader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=4)

    global_step = sampler.diagnostics.total_cuts  # <-- Restore the global step idx.
    for epoch in range(sampler.epoch, 30):  # <-- Skip previous epochs that are already processed.

        dloader.sampler.set_epoch(epoch)
        for batch in dloader:
            # Note: the first batch is going to be from step 75009.
            # With DataLoader num_workers==0, it would have been 75001, but we get
            # +8 because of num_workers==4 * prefetching_factor==2

            # ... processing forward, backward, etc.
            global_step += 1

.. note::

    In general, the sampler arguments may be different -- loading a ``state_dict`` will
    overwrite the arguments, and emit a warning for the user to be aware what happened.
    :class:`~lhotse.dataset.sampling.BucketingSampler` is an exception -- the ``num_buckets``
    and ``bucket_method`` must be consistent, otherwise we couldn't guarantee identical
    outcomes after training resumption.

.. note::

    The ``DataLoader``'s ``num_workers`` can be different after resuming.


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

Handling random seeds
---------------------

Lhotse provides several mechanisms for controlling randomness. At a basic level, there is a function :func:`lhotse.utils.fix_random_seed` which seeds Python's, numpy's and torch's RNGs with the provided number.

However, many functions and classes in Lhotse accept either a random seed or an RNG instance to provide a finer control over randomness. Whenever random seed is accepted, it can be either an integer, or one of two strings: ``"randomized"`` or ``"trng"``.

* ``"randomized``" seed is resolved lazily at the moment it's needed and is intended as a mechanism to provide a different seed to each dataloading worker. In order for ``"randomized"`` to work, you have to first invoke :func:`lhotse.dataset.dataloading.worker_init_fn` in a given subprocess which sets the right environment variables. With a PyTorch ``DataLoader`` you can pass the keyword argument ``worker_init_fn==make_worker_init_fn(seed=int_seed, rank=..., world_size=...)`` using :func:`lhotse.dataset.dataloading.make_worker_init_fn` which will set the right seeds for you in multiprocessing and multi-node training. Note that if you resume training, you should change the ``seed`` passed to ``make_worker_init_fn`` on each resumed run to make the model train on different data.
* ``"trng"`` seed is also resolved lazily at runtime, but it uses a true RNG (if available on your OS; consult Python's ``secrets`` module documentation). It's an easy way to ensure that every time you iterate data it's done in different order, but may cause debugging data issues to be more difficult.

.. note:: The lazy seed resolution is done by calling :func:`lhotse.dataset.dataloading.resolve_seed`.


Customizing sampling constraints
--------------------------------

Since version 1.22.0, Lhotse provides a mechanism to customize how samplers measure the "length" of each example
for the purpose of determining dynamic batch size. To leverage this option, use the keyword argument ``constraint``
in :class:`~lhotse.dataset.sampling.DynamicCutSampler` or :class:`~lhotse.dataset.sampling.DynamicBucketingSampler`.
The sampling criteria are defined by implementing a subclass of :class:`~lhotse.dataset.sampling.base.SamplingConstraint`:

.. autoclass:: lhotse.dataset.sampling.base.SamplingConstraint
    :members:

The default constraint is :class:`~lhotse.dataset.sampling.base.TimeConstraint` which is created from
``max_duration``, ``max_cuts``, and ``quadratic_duration`` args passed to samplers constructor.

Sampling non-audio data
***********************

Because :class:`~lhotse.dataset.sampling.base.SamplingConstraint` defines the method ``measure_length``,
it's possible to use a different attribute than duration (or a different formula) for computing the effective batch size.
This enables re-using Lhotse's sampling algorithms for other data than speech, and passing around other objects than :class:`~lhotse.cut.Cut`.

To showcase this, we added an experimental support for text-only dataloading. We introduced a few classes specifically for this purpose:

.. autoclass:: lhotse.cut.text.TextExample
    :members:

.. autoclass:: lhotse.cut.text.TextPairExample
    :members:

.. autoclass:: lhotse.lazy.LazyTxtIterator
    :members:

.. autoclass:: lhotse.dataset.sampling.base.TokenConstraint
    :members:

A minimal example of how to perform text-only dataloading is available below (note that any of these classes may be replaced by your own implementation if that is more suitable to your work)::

    import torch
    import numpy as np
    from lhotse import CutSet
    from lhotse.lazy import LazyTxtIterator
    from lhotse.cut.text import TextPairExample
    from lhotse.dataset import DynamicBucketingSampler, TokenConstraint
    from lhotse.dataset.collation import collate_vectors

    examples = CutSet(LazyTxtIterator("data.txt"))

    def tokenize(example):
        # tokenize as individual bytes; BPE or another technique may be used here instead
        example.tokens = np.frombuffer(example.text.encode("utf-8"), np.int8)
        return example

    examples = examples.map(tokenize, apply_fn=None)

    sampler = DynamicBucketingSampler(examples, constraint=TokenConstraint(max_tokens=1024, quadratic_length=128),      num_buckets=2)

    class ExampleTextDataset(torch.utils.data.Dataset):
        def __getitem__(self, examples: CutSet):
            tokens = [ex.tokens for ex in examples]
            token_lens = torch.tensor([len(t) for t in tokens])
            tokens = collate_vectors(tokens, padding_value=-1)
            return tokens, token_lens

    dloader = torch.utils.data.DataLoader(ExampleTextDataset(), sampler=sampler, batch_size=None)

    for batch in dloader:
        print(batch)

.. note:: Support for this kind of dataloading is experimental in Lhotse. If you run into any rough edges, please let us know.

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

Dataloading seeding utilities
-----------------------------

.. automodule:: lhotse.dataset.dataloading
