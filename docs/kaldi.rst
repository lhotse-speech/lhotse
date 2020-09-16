Kaldi Interoperability
======================

We support importing Kaldi data directories that contain at least the ``wav.scp`` file, required to create the ``RecordingSet``. Other files, such as ``segments``, ``utt2spk``, etc. are used to create the ``SupervisionSet``.

We currently do not support the following (but may start doing so in the future):

* Importing Kaldi's extracted features (``feats.scp`` is ignored)
* Exporting Lhotse manifests as Kaldi data directories.

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

    lhotse convert-kaldi data/train 16000 train_manifests