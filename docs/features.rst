Feature extraction
==================

.. caution::
    Lhotse feature extraction is still very much in the works. We will extend it to support custom feature extractors.

Feature extraction in Lhotse is currently based exclusively on the `Torchaudio`_ library.
We support spectrograms, log-Mel energies (*fbank*) and MFCCs.
*Fbank* are the default features, and as of now the only ones supported in feature-space cut mixing.

We are striving for a simple relation between the audio duration, the number of frames,
and the frame shift.
You only need to know two of those values to compute the third one, regardless of the frame length.
This is equivalent of having Kaldi's ``snip_edges`` parameter set to False.


Storing features
****************

Features in Lhotse are stored as numpy matrices with shape ``(num_frames, num_features)``.
By default, we use `lilcom`_ for lossy compression and reduce the size on the disk approximately by half.
For now, we extract the features for the whole recordings, and store them in separate files.
We retrieve them by loading the whole feature matrix and selecting the relevant region (e.g. specified by a cut).
Eventually we will look into optimizing the storage to further reduce the I/O.

The feature matrices are accompanied by a YAML feature manifest. An example:

.. code-block:: yaml

    feature_extractor:
      fbank_config:
        use_log_fbank: true
      mfcc_config:
        cepstral_lifter: 22.0
        num_ceps: 13
      mfcc_fbank_common_config:
        high_freq: 0.0
        low_freq: 20.0
        num_mel_bins: 23
        use_energy: false
        vtln_high: -500.0
        vtln_low: 100.0
        vtln_warp: 1.0
      spectrogram_config:
        dither: 0.0
        energy_floor: 0.0
        frame_length: 0.025
        frame_shift: 0.01
        min_duration: 0.0
        preemphasis_coefficient: 0.97
        raw_energy: true
        remove_dc_offset: true
        round_to_power_of_two: true
        window_type: povey
      type: fbank
    features:
      - channel_id: 0
        duration: 16.04
        num_features: 23
        num_frames: 1604
        recording_id: recording-1
        start: 0.0
        storage_path: test/fixtures/libri/storage/dc2e0952-f2f8-423c-9b8c-f5481652ee1d.llc
        storage_type: lilcom
        type: fbank

The manifest consists of two parts:

- a feature extractor configuration;
- a list of documents describing each feature matrix.

The feature extractor config is a detailed list of arguments used to extract the features in a given manifest.
Currently, these arguments are the sum of all possible arguments for all feature types - we will likely refactor that
in the future, and we might either move this section into a separate manifest or drop it entirely and store
only the non-default values in each feature document.

The feature documents contain the information necessary to tie them to a particular recording - ``start``, ``duration``,
``channel`` and ``recording_id``. They currently do not have their own IDs.
They also provide some useful information, such as the type of features, number of frames and feature dimension.
Finally, they specify how the feature matrix is stored with ``storage_type`` (currently ``numpy`` or ``lilcom``),
and where to find it with the ``storage_path``. In the future there might be more storage types.


Python
******

The feature manifest is represented by a :class:`FeatureSet` object.
Feature extractors have a class that represents both the extract and its configuration, named :class:`FeatureExtractor`.
We provide a utility called :class:`FeatureSetBuilder` that can process a :class:`RecordingSet` in parallel,
store the feature matrices on disk and generate a feature manifest.

For example:

.. code-block:: python

    # Read a RecordingSet from disk
    recording_set = RecordingSet.from_yaml('audio.yml')
    # Create a feature extractor with default settings
    feature_extractor = FeatureExtractor()
    # Create a feature set builder that uses this extractor and stores the results in a directory called 'features'
    builder = FeatureSetBuilder(feature_extractor=FeatureExtractor(), output_dir='features/')
    # Extract the features using 8 parallel processes, compress, and store them on in 'features/storage/' directory.
    # Then, return the feature manifest object, which is also compressed and
    # stored in 'features/feature_manifest.yml.gz'
    feature_set = builder.process_and_store_recordings(
        recordings=recording_set,
        compressed=True,
        num_jobs=8
    )

CLI
***

An equivalent example using the terminal:

.. code-block:: bash

    lhotse write-default-feature-config feat-config.yml
    lhotse make-feats -j 8 --compressed -f feat-config.yml audio.yml features/


Kaldi compatibility caveats
***************************

We are relying on `Torchaudio`_ Kaldi compatibility module, so most of the spectrogram/fbank/mfcc parameters are
the same as in Kaldi.
However, we are not fully compatible - Kaldi computes energies from a signal scaled between -32,768 to 32,767, while
`Torchaudio`_ scales the signal between -1.0 and 1.0.
It results in Kaldi energies being significantly greater than in Lhotse.
By default, we turn off dithering for deterministic feature extraction.


.. _Torchaudio: link: https://pytorch.org/audio/
.. _lilcom: link: https://github.com/danpovey/lilcom

