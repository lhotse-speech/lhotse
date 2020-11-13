Augmentation
============

We support time-domain data augmentation via `WavAugment`_ and `torchaudio`_ libraries.
They both leverage libsox to provide about 50 different audio effects like reverb, speed perturbation, pitch, etc.

Since ``WavAugment`` depends on libsox, it is an optional depedency for Lhotse, which can be installed using ``tools/install_wavaugment.sh`` (for convenience, the script will also compile libsox from source - note that the ``WavAugment`` authors warn their library is untested on Mac).

Torchaudio also depends on libsox, but seems to provide it when installed via anaconda.
This functionality is only available with PyTorch 1.7+ and torchaudio 0.7+.

Using Lhotse's Python API, you can compose an arbitrary effect chain. On the other hand, for the CLI we provide a small number of predefined effect chains, such as ``pitch`` (pitch shifting), ``reverb`` (reverberation), and ``pitch_reverb_tdrop`` (pitch shift + reverberation + time dropout of a 50ms chunk).

Python usage
************

.. warning::
    When using WavAugment or torchaudio data augmentation together with a multiprocessing executor (i.e. ``ProcessPoolExecutor``), it is necessary to start it using the "spawn" context. Otherwise the process may hang (or terminate) on some systems due to libsox internals not handling forking well. Use: ``ProcessPoolExecutor(..., mp_context=multiprocessing.get_context("spawn"))``.

Lhotse's ``FeatureExtractor`` and ``Cut`` offer convenience functions for feature extraction with data augmentation
performed before that. These functions expose an optional parameter called ``augment_fn`` that has a signature like:

.. code-block::

    def augment_fn(audio: Union[np.ndarray, torch.Tensor], sampling_rate: int) -> np.ndarray: ...

For ``torchaudio`` we define a ``SoxEffectTransform`` class:

.. autoclass:: lhotse.augmentation.SoxEffectTransform
  :members:
  :noindex:

We define a ``WavAugmenter`` class that is a thin wrapper over ``WavAugment``. It can either be created with a predefined, or a user-supplied effect chain.

.. autoclass:: lhotse.augmentation.WavAugmenter
  :members:
  :noindex:

CLI usage
*********

To extract features from augmented audio, you can pass an extra ``--augmentation`` argument to ``lhotse feat extract``.

.. code-block:: bash

    lhotse feat extract -a pitch ...

You can create a dataset with both clean and augmented features by combining different variants of extracted features, e.g.:

.. code-block:: bash

    lhotse feat extract audio.yml clean_feats/
    lhotse feat extract -a pitch audio.yml pitch_feats/
    lhotse feat extract -a reverb audio.yml reverb_feats/
    lhotse yaml combine {clean,pitch,reverb}_feats/feature_manifest.yml.gz combined_feats.yml

.. _WavAugment: https://github.com/facebookresearch/WavAugment
.. _torchaudio: https://pytorch.org/audio/stable/index.html
