Feature extraction
==================

Lhotse provides the following feature extractor implementations:

- Log-Mel filter-bank :class:`~lhotse.features.kaldi.extractors.Fbank` and MFCC :class:`~lhotse.features.kaldi.extractors.Mfcc` PyTorch implementations. They are very close to Kaldi's, and their underlying components are PyTorch modules that can be used as layers in neural networks (i.e. support batching, GPUs, autograd, and TorchScript). These classes are found in ``lhotse.features.kaldi.layers`` (in particular: :class:`~lhotse.features.kaldi.layers.Wav2LogFilterBank` and :class:`~lhotse.features.kaldi.layers.Wav2MFCC`). We also provide online inference methods to support deployment in audio streaming applications.
- `Torchaudio`_ Kaldi-compatible extractors :class:`~lhotse.features.fbank.TorchaudioFbank`, :class:`~lhotse.features.mfcc.TorchaudioMfcc`, and :class:`~lhotse.features.spectrogram.Spectrogram`. They only support processing one utterance at a time (batching is not possible).
- `Librosa`_ compatible filter-bank feature extractor :class:`~lhotse.features.librosa_fbank.LibrosaFbank` (compatible with the one used in `ESPnet`_ and `ParallelWaveGAN`_ projects for TTS and vocoders);
- `kaldifeat`_ -- another Kaldi-compatible feature extraction implementation that can process batches of uneven lengths efficiently, implemented in C++ with Python wrappers.
- `opensmile`_ -- a wrapper over popular set of feature extractors, often used in modeling non-verbal aspects of speech (e.g., emotion recognition).

We also support custom defined feature extractors via a Python API.

We are striving for a simple relation between the audio duration, the number of frames,
and the frame shift (with a known sampling rate)::

    num_samples = round(duration * sampling_rate)
    window_hop = round(frame_shift * sampling_rate)
    num_frames = int((num_samples + window_hop // 2) // window_hop)

This is equivalent of having Kaldi's ``snip_edges`` parameter set to False, and Lhotse expects **every** feature extractor to conform to that requirement.


Storing features
****************

Features in Lhotse are stored as numpy matrices with shape ``(num_frames, num_features)``.
By default, we use `lilcom`_ for lossy compression and reduce the size on the disk by about 3x.
The lilcom compression method uses a fixed precision that doesn't depend on the magnitude of the thing being compressed, so it's better suited to log-energy features than energy features.
By default, we store these matrices in archives with our own custom format that allows efficient reads of chunks compressed with lilcom. Other options such as HDF5 are also available.

There are two types of manifests:

- one describing the feature extractor;
- one describing the extracted feature matrices.

The feature extractor manifest is mapped to a Python configuration dataclass. An example for *spectrogram*:

.. code-block:: yaml

    dither: 0.0
    energy_floor: 1e-10
    frame_length: 0.025
    frame_shift: 0.01
    min_duration: 0.0
    preemphasis_coefficient: 0.97
    raw_energy: true
    remove_dc_offset: true
    round_to_power_of_two: true
    window_type: povey
    type: spectrogram

And the corresponding configuration class:

.. autoclass:: lhotse.features.SpectrogramConfig
  :members:
  :noindex:

The feature matrices manifest is a list of documents.
These documents contain the information necessary to tie the features to a particular recording: ``start``, ``duration``,
``channel`` and ``recording_id``.
They also provide some useful information, such as the type of features, number of frames and feature dimension.
Finally, they specify how the feature matrix is stored with ``storage_type`` (currently ``numpy`` or ``lilcom``),
and where to find it with the ``storage_path``. In the future there might be more storage types.

.. code-block:: yaml

    - channels: 0
      duration: 16.04
      num_features: 23
      num_frames: 1604
      recording_id: recording-1
      start: 0.0
      storage_path: test/fixtures/libri/storage/dc2e0952-f2f8-423c-9b8c-f5481652ee1d.llc
      storage_type: lilcom
      type: fbank


Feature normalization
*********************

We will briefly discuss how to perform mean and variance normalization (a.k.a. CMVN) in Lhotse effectively. We compute and store unnormalized features, and it is up to the user to normalize them if they want to do so. There are three common ways to perform feature normalization:

- **Global normalization**: we compute the means and variances using the whole data (``FeatureSet`` or ``CutSet``), and apply the same transform on every sample. The global statistics can be computed efficiently with ``FeatureSet.compute_global_stats()`` or ``CutSet.compute_global_feature_stats()``. They use an iterative algorithm that does not require loading the whole dataset into memory.
- **Per-instance normalization**: we compute the means and variances separately for each data sample (i.e. a single feature matrix). Each feature matrix undergoes a different transform. This approach seems to be common in computer vision modeling.
- **Sliding window ("online") normalization**: we compute the means and variances using a slice of the feature matrix with a specified duration, e.g. 3 seconds (a standard value in Kaldi). This is useful when we expect the model to work on incomplete inputs, e.g. streaming speech recognition. We currently recommend using `Torchaudio CMVN`_ for that.

Python usage
************

Typically you'll want to extract features from cuts. In case of long recordings, it is fine to extract the features for long-recording cuts, and cut those into shorter segments later. Our default feature storage mechanism is fairly efficient when reading chunks.

.. code-block:: python

    from lhotse import CutSet

    cuts = CutSet.from_file("data/cuts.jsonl.gz")
    # Create a log Mel energy filter bank feature extractor with default settings
    fbank = Fbank()
    # Compute features for cuts with 8 parallel jobs and return a new CutSet which
    # references those features.
    cuts = cuts.compute_and_store_features(
        extractor=fbank,
        storage_path="data/fbank",
        num_jobs=8,
    )
    cuts.to_file("data/cuts_fbank.jsonl.gz")

CLI usage
*********

An equivalent example using the terminal:

.. code-block:: console

    lhotse feat write-default-config feat-config.yml
    lhotse feat extract-cuts -j 8 -f feat-config.yml \
        data/cuts.jsonl.gz data/cuts_fbank.jsonl.gz data/fbank


Kaldi compatibility caveats
***************************

Most of the spectrogram/fbank/mfcc parameters are the same as in Kaldi.
However, we are not fully compatible - Kaldi computes energies from a signal scaled between -32,768 to 32,767, while we scale signal between -1.0 and 1.0.
It results in Kaldi energies being significantly greater than in Lhotse.
Also, by default, we turn off dithering for deterministic feature extraction.

.. _Torchaudio: https://pytorch.org/audio/
.. _Librosa: https://librosa.org
.. _ESPnet: https://github.com/espnet/espnet
.. _ParallelWaveGAN: https://github.com/kan-bayashi/ParallelWaveGAN
.. _lilcom: https://github.com/danpovey/lilcom
.. _power quantity: https://en.wikipedia.org/wiki/Power,_root-power,_and_field_quantities
.. _Torchaudio CMVN: https://pytorch.org/audio/stable/transforms.html#slidingwindowcmn
.. _kaldifeat: https://github.com/csukuangfj/kaldifeat