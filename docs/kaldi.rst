Kaldi Interoperability
======================

Data import/export
******************

We support importing Kaldi data directories that contain at least the ``wav.scp`` file,
required to create the :class:`~lhotse.audio.RecordingSet`.
Other files, such as ``segments``, ``utt2spk``, etc. are used to create the :class:`~lhotse.supervision.SupervisionSet`.
We also support converting ``feats.scp`` to :class:`~lhotse.features.base.FeatureSet`, and reading features
directly from Kaldi's scp/ark files via `kaldi_native_io`_ library (which is an optional Lhotse's dependency).

We also allow to export a pair of :class:`~lhotse.audio.RecordingSet` and :class:`~lhotse.supervision.SupervisionSet`
to a Kaldi data directory.

We currently do not support the following (but may start doing so in the future):

* Exporting Lhotse extracted features to Kaldi's ``feats.scp``
* Export Lhotse's multi-channel recording sets to Kaldi

Kaldi feature extractors
************************

We support Kaldi-compatible log-mel filter energies ("fbank") and MFCCs.
We provide a PyTorch implementation that is GPU-compatible, allows batching, and backpropagation.
To learn more about feature extraction in Lhotse, see :doc:`features`.

Python
******

Python methods related to Kaldi support:

.. automodule:: lhotse.kaldi
  :members:
  :noindex:

CLI
***

Converting Kaldi data directory called ``data/train``, with 16kHz sampling rate recordings,
to a directory with Lhotse manifests called ``train_manifests``:

.. code-block:: bash

    # Convert data/train to train_manifests/{recordings,supervisions}.json
    lhotse kaldi import \
        data/train \
        16000 \
        train_manifests

    # Convert train_manifests/{recordings,supervisions}.json to data/train
    lhotse kaldi export \
        train_manifests/recordings.json \
        train_manifests/supervisions.json \
        data/train


.. _kaldi_native_io: https://pypi.org/project/kaldi_native_io/
