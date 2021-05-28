Feature extraction
==================

Lhotse provides the following feature extractor implementations:

- `Torchaudio`_ based extractors, which involve :class:`~lhotse.features.fbank.Fbank`, :class:`~lhotse.features.mfcc.Mfcc`, and :class:`~lhotse.features.spectrogram.Spectrogram`;
- `Librosa`_ compatible filter-bank feature extractor :class:`~lhotse.features.librosa_fbank.LibrosaFbank` (compatible with the one used in `ESPnet`_ and `ParallelWaveGAN`_ projects for TTS and vocoders);
- Log-Mel filter-bank :class:`~lhotse.features.kaldi.extractors.KaldiFbank` and MFCC :class:`~lhotse.features.kaldi.extractors.KaldiMfcc` PyTorch implementations. They are very close to Kaldi's, and their underlying components are PyTorch modules that can be used as layers in neural networks (i.e. support batching, GPUs, and autograd). These classes are found in ``lhotse.features.kaldi.layers`` (in particular: :class:`~lhotse.features.kaldi.layers.Wav2LogFilterBank` and :class:`~lhotse.features.kaldi.layers.Wav2MFCC`).

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
We currently support two kinds of storage:

- HDF5 files with multiple feature matrices
- directory with feature matrix per file

We retrieve the arrays by loading the whole feature matrix from disk and selecting the relevant region (e.g. specified by a cut). Therefore it makes sense to cut the recordings first, and then extract the features for them to avoid loading unnecessary data from disk (especially for very long recordings).

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
``channel`` and ``recording_id``. They currently do not have their own IDs.
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


Creating custom feature extractor
*********************************

There are two components needed to implement a custom feature extractor: a configuration and the extractor itself.
We expect the configuration class to be a dataclass, so that it can be automatically mapped to dict and serialized.
The feature extractor should inherit from ``FeatureExtractor``,
and implement a small number of methods/properties.
The base class takes care of initialization (you need to pass a config object), serialization to YAML, etc.
A minimal, complete example of adding a new feature extractor:

.. code-block::

    from scipy.signal import stft

    @dataclass
    class ExampleFeatureExtractorConfig:
        frame_len: Seconds = 0.025
        frame_shift: Seconds = 0.01


    class ExampleFeatureExtractor(FeatureExtractor):
        """
        A minimal class example, showing how to implement a custom feature extractor in Lhotse.
        """
        name = 'example-feature-extractor'
        config_type = ExampleFeatureExtractorConfig

        def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
            f, t, Zxx = stft(
                samples,
                sampling_rate,
                nperseg=round(self.config.frame_len * sampling_rate),
                noverlap=round(self.frame_shift * sampling_rate)
            )
            # Note: returning a magnitude of the STFT might interact badly with lilcom compression,
            # as it performs quantization of the float values and works best with log-scale quantities.
            # It's advised to turn lilcom compression off, or use log-scale, in such cases.
            return np.abs(Zxx)

        @property
        def frame_shift(self) -> Seconds:
            return self.config.frame_shift

        def feature_dim(self, sampling_rate: int) -> int:
            return (sampling_rate * self.config.frame_len) / 2 + 1

The overridden members include:

- ``name`` for easy debuggability/automatic re-creation of an extractor;
- ``config_type`` which specifies the complementary configuration class type;
- ``extract()`` where the actual computation takes place;
- ``frame_shift`` property, which is key to know the relationship between the duration and the number of frames.
- ``feature_dim()`` method, which accepts the ``sampling_rate`` as its argument, as some types of features (e.g. spectrogram) will depend on that.

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
- ``compute_energy()`` which specifies how to obtain a total energy of the feature matrix, which is needed to mix two signals with a specified SNR. E.g. for a power spectrogram, this could be the sum of every time-frequency bin. It is expected to never return a zero.

During the feature-domain mix with a specified signal-to-noise ratio (SNR), we assume that one of the signals is a reference signal - it is used to initialize the ``FeatureMixer`` class. We compute the energy of both signals and scale the non-reference signal, so that its energy satisfies the requested SNR.

Note that we interpret the energy and the SNR in a `power quantity`_ context (as opposed to root-power/field quantities).

Feature normalization
*********************

We will briefly discuss how to perform mean and variance normalization (a.k.a. CMVN) in Lhotse effectively. We compute and store unnormalized features, and it is up to the user to normalize them if they want to do so. There are three common ways to perform feature normalization:

- **Global normalization**: we compute the means and variances using the whole data (``FeatureSet`` or ``CutSet``), and apply the same transform on every sample. The global statistics can be computed efficiently with ``FeatureSet.compute_global_stats()`` or ``CutSet.compute_global_feature_stats()``. They use an iterative algorithm that does not require loading the whole dataset into memory.
- **Per-instance normalization**: we compute the means and variances separately for each data sample (i.e. a single feature matrix). Each feature matrix undergoes a different transform. This approach seems to be common in computer vision modelling.
- **Sliding window ("online") normalization**: we compute the means and variances using a slice of the feature matrix with a specified duration, e.g. 3 seconds (a standard value in Kaldi). This is useful when we expect the model to work on incomplete inputs, e.g. streaming speech recognition. We currently recommend using `Torchaudio CMVN`_ for that.

Storage backend details
***********************

Lhotse can be extended with additional storage backends via two abstractions: ``FeaturesWriter`` and ``FeaturesReader``. We currently implement the following writers (and their corresponding readers):

- ``lhotse.features.io.LilcomFilesWriter``
- ``lhotse.features.io.NumpyFilesWriter``
- ``lhotse.features.io.LilcomHdf5Writer``
- ``lhotse.features.io.NumpyHdf5Writer``

The ``FeaturesWriter`` and ``FeaturesReader`` API is as follows:

.. autoclass:: lhotse.features.io.FeaturesWriter
  :members:
  :noindex:

.. autoclass:: lhotse.features.io.FeaturesReader
  :members:
  :noindex:

Python usage
************

The feature manifest is represented by a :class:`FeatureSet` object.
Feature extractors have a class that represents both the extract and its configuration, named :class:`FeatureExtractor`.
We provide a utility called :class:`FeatureSetBuilder` that can process a :class:`RecordingSet` in parallel,
store the feature matrices on disk and generate a feature manifest.

For example:

.. code-block:: python

    from lhotse import RecordingSet, Fbank, LilcomHdf5Writer

    # Read a RecordingSet from disk
    recording_set = RecordingSet.from_yaml('audio.yml')
    # Create a log Mel energy filter bank feature extractor with default settings
    feature_extractor = Fbank()
    # Create a feature set builder that uses this extractor and stores the results in a directory called 'features'
    with LilcomHdf5Writer('feats.h5') as storage:
        builder = FeatureSetBuilder(feature_extractor=feature_extractor, storage=storage)
        # Extract the features using 8 parallel processes, compress, and store them on in 'features/storage/' directory.
        # Then, return the feature manifest object, which is also compressed and
        # stored in 'features/feature_manifest.json.gz'
        feature_set = builder.process_and_store_recordings(
            recordings=recording_set,
            num_jobs=8
        )

.. py:currentmodule:: lhotse.cut

It is also possible to extract the features directly from :class:`CutSet` - see below:

.. autofunction:: lhotse.cut.CutSet.compute_and_store_features
  :noindex:

CLI usage
*********

An equivalent example using the terminal:

.. code-block:: console

    lhotse feat write-default-config feat-config.yml
    lhotse feat extract -j 8 --storage-type lilcom_files -f feat-config.yml audio.yml features/


Kaldi compatibility caveats
***************************

We are relying on `Torchaudio`_ Kaldi compatibility module, so most of the spectrogram/fbank/mfcc parameters are
the same as in Kaldi.
However, we are not fully compatible - Kaldi computes energies from a signal scaled between -32,768 to 32,767, while
`Torchaudio`_ scales the signal between -1.0 and 1.0.
It results in Kaldi energies being significantly greater than in Lhotse.
By default, we turn off dithering for deterministic feature extraction.
The same is true for features extracted with ``lhotse.features.kaldi`` module.


.. _Torchaudio: https://pytorch.org/audio/
.. _Librosa: https://librosa.org
.. _ESPnet: https://github.com/espnet/espnet
.. _ParallelWaveGAN: https://github.com/kan-bayashi/ParallelWaveGAN
.. _lilcom: https://github.com/danpovey/lilcom
.. _power quantity: https://en.wikipedia.org/wiki/Power,_root-power,_and_field_quantities
.. _Torchaudio CMVN: https://pytorch.org/audio/stable/transforms.html#slidingwindowcmn
