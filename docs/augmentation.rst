Augmentation
============

We support time-domain data augmentation via `WavAugment`_ library. ``WavAugment`` combines libsox and its own implementations to provide a range of augmentations. Since ``WavAugment`` depends on libsox, it is an optional depedency for Lhotse, which can be installed using ``tools/install_wavaugment.sh`` (for convenience, on Mac OS X the script will also compile libsox from source - though note that the ``WavAugment`` authors warn their library is untested on Mac).

Using Lhotse's Python API, you can compose an arbitrary effect chain. On the other hand, for the CLI we provide a small number of predefined effect chains, such as ``pitch`` (pitch shifting), ``reverb`` (reverberation), and ``pitch_reverb_tdrop`` (pitch shift + reverberation + time dropout of a 50ms chunk).

Python usage
************

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

.. _WavAugment: link: https://github.com/facebookresearch/WavAugment
