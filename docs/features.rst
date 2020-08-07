Feature extraction
==================

Feature extraction in Lhotse is currently based exclusively on the `Torchaudio`_ library.
We support spectrograms, log-Mel energies (*fbank*) and MFCCs.
*Fbank* are the default features.
We also support custom defined feature extractors via a Python API
(which won't be available in the CLI, unless there is a popular demand for that).

We are striving for a simple relation between the audio duration, the number of frames,
and the frame shift.
You only need to know two of those values to compute the third one, regardless of the frame length.
This is equivalent of having Kaldi's ``snip_edges`` parameter set to False.


Storing features
****************

Features in Lhotse are stored as numpy matrices with shape ``(num_frames, num_features)``.
By default, we use `lilcom`_ for lossy compression and reduce the size on the disk approximately by half.
The lilcom compression method uses a fixed precision that doesn't depend on the magnitude of the thing being compressed, so it's better suited to log-energy features than energy features.
For now, we extract the features for the whole recordings, and store them in separate files.
We retrieve them by loading the whole feature matrix and selecting the relevant region (e.g. specified by a cut).
Eventually we will look into optimizing the storage to further reduce the I/O.

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

The feature matrices manifest is a list of documents.
These documents contain the information necessary to tie the features to a particular recording: ``start``, ``duration``,
``channel`` and ``recording_id``. They currently do not have their own IDs.
They also provide some useful information, such as the type of features, number of frames and feature dimension.
Finally, they specify how the feature matrix is stored with ``storage_type`` (currently ``numpy`` or ``lilcom``),
and where to find it with the ``storage_path``. In the future there might be more storage types.

.. code-block:: yaml

    - channel_id: 0
      duration: 16.04
      num_features: 23
      num_frames: 1604
      recording_id: recording-1
      start: 0.0
      storage_path: test/fixtures/libri/storage/dc2e0952-f2f8-423c-9b8c-f5481652ee1d.llc
      storage_type: lilcom
      type: fbank


Creating custom feature extractor
*********************************

There are two components needed to implement a custom feature extractor: a configuration and the extractor itself.
We expect the configuration class to be a dataclass, so that it can be automatically mapped to dict and serialized.
The feature extractor should inherit from :ref:`lhotse.features.FeatureExtractor`,
and implement a small number of methods/properties.
The base class takes care of initialization (you need to pass a config object), serialization to YAML, etc.
A minimal, complete example of adding a new feature extractor:

.. code-block::

    @dataclass
    class ExampleFeatureExtractorConfig:
        frame_shift: Seconds = 0.01

    class ExampleFeatureExtractor(FeatureExtractor):
        name = 'example-feature-extractor'
        config_type = ExampleFeatureExtractorConfig

        def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
            f, t, Zxx = stft(samples, sampling_rate, noverlap=round(self.frame_shift * sampling_rate))
            return np.abs(Zxx)

        @property
        def frame_shift(self) -> Seconds:
            return self.config.frame_shift

The overridden members include:

- ``name`` for easy debuggability/automatic re-creation of an extractor;
- ``config_type`` which specifies the complementary configuration class type;
- ``extract()`` where the actual computation takes place;
- ``frame_shift`` property, which is key to know the relationship between the duration and the number of frames.

Additionally, there are two extra methods than when overridden, allow to perform dynamic feature-space mixing (see Cuts):

.. code-block::

    @staticmethod
    def mix(features_a: np.ndarray, features_b: np.ndarray, gain_b: float) -> np.ndarray:
        raise ValueError(f'The feature extractor\'s "mix" operation is undefined.')

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        raise ValueError(f'The feature extractor\'s "compute_energy" is undefined.')

They are:

- ``mix()`` which specifies how to mix two feature matrices to obtain a new feature matrix representing the sum of signals;
- ``compute_energy()`` which specifies how to obtain a total energy of the feature matrix, which is needed to mix two signals with a specified SNR.

Python usage
************

The feature manifest is represented by a :class:`FeatureSet` object.
Feature extractors have a class that represents both the extract and its configuration, named :class:`FeatureExtractor`.
We provide a utility called :class:`FeatureSetBuilder` that can process a :class:`RecordingSet` in parallel,
store the feature matrices on disk and generate a feature manifest.

For example:

.. code-block:: python

    # Read a RecordingSet from disk
    recording_set = RecordingSet.from_yaml('audio.yml')
    # Create a log Mel energy filter bank feature extractor with default settings
    feature_extractor = Fbank()
    # Create a feature set builder that uses this extractor and stores the results in a directory called 'features'
    builder = FeatureSetBuilder(feature_extractor=feature_extractor, output_dir='features/')
    # Extract the features using 8 parallel processes, compress, and store them on in 'features/storage/' directory.
    # Then, return the feature manifest object, which is also compressed and
    # stored in 'features/feature_manifest.yml.gz'
    feature_set = builder.process_and_store_recordings(
        recordings=recording_set,
        compressed=True,
        num_jobs=8
    )

CLI usage
*********

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

