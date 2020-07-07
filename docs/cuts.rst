Cuts
====

Overview
********

Audio cuts are one of the main Lhotse features.
Cut is a part of a recording, but it can be longer than a supervision segment, or even span multiple segments.
The regions without a supervision are just audio that we don't assume we know anything about - there may be silence,
noise, non-transcribed speech, etc.
Task-specific datasets can leverage this information to generate masks for such regions.

Currently, cuts are created after the feature extraction step (we might still change that).
It means that every cut also represents the extracted features for the part of recording it represents.

Cuts can be modified using three basic operations: truncation, overlaying (mixing) and appending.
These operations are not immediately performed on the audio or features.
Instead, we create new :class:`Cut` objects, possibly of different types, that represent a cut after modification.
We only modify the actual audio and feature matrices once the user calls :meth:`load_features` or :meth:`load_audio`.

This design allows for quick on-the-fly data augmentation.
In each training epoch, we may mix the cuts with different noises, SNRs, etc.
We also do not need to re-compute and store the features for different mixes, as the mixing process consists of
element-wise addition of the spectral energies (possibly with additional exp and log operations for log-energies).
As of now, we only support this dynamic mix on log Mel energy (_fbank_) features.
We anticipate to add support for other types of features as well.

The common attributes for all cut objects are the following:

- id
- duration
- supervisions
- num_frames
- num_features
- load_features()
- truncate()
- overlay()
- append()
- from_dict()

:meth:`load_audio` is not yet supported for overlayed cuts, but will be eventually.

Types of cuts
*************

There are three cut classes:

- :class:`Cut`, also referred to as "simple cut", can be traced back to a single particular recording (and channel).
- :class:`PaddingCut` is an "artificial" recording used for padding other Cuts through mixing to achieve uniform duration.
- :class:`MixedCut` is a collection of :class:`Cut` and :class:`PaddingCut` objects, together with mix parameters: offset and desired sound-to-noise ratio (SNR) for each track. Both the offset and the SNR are relative to the first cut in the mix.

Each of these types has additional attributes that are not common - e.g., it makes sense to specify *start* for
:class:`Cut` to locate it in the source recording, but it is undefined for :class:`MixedCut` and :class:`PaddingCut`.

Cut manifests
*************

All cut types can be stored in the YAML manifests. An example manifest with simple cuts might look like:

.. code-block:: yaml

    - duration: 10.0
      features:
        channel_id: 0
        duration: 16.04
        num_features: 23
        num_frames: 1604
        recording_id: recording-1
        start: 0.0
        storage_path: test/fixtures/libri/storage/dc2e0952-f2f8-423c-9b8c-f5481652ee1d.llc
        storage_type: lilcom
        type: fbank
      id: 849e13d8-61a2-4d09-a542-dac1aee1b544
      start: 0.0
      supervisions: []
      type: Cut

Notice that the cut type is specified in YAML. The supervisions list might be empty - some tasks do not need them,
e.g. unsupervised training, source separation, or speech enhancement.

Mixed cuts look differently in the manifest:

.. code-block:: yaml

    - id: mixed-cut-id
      tracks:
        - cut:
            duration: 7.78
            features:
              channel_id: 0
              duration: 7.78
              type: fbank
              num_frames: 778
              num_features: 23
              recording_id: 7850-286674-0014
              start: 0.0
              storage_path: test/fixtures/mix_cut_test/feats/storage/9dc645db-cbe4-4529-85e4-b6ed4f59c340.llc
              storage_type: lilcom
            id: 0c5fdf79-efe7-4d45-b612-3d90d9af8c4e
            start: 0.0
            supervisions:
              - channel_id: 0
                duration: 7.78
                gender: f
                id: 7850-286674-0014
                language: null
                recording_id: 7850-286674-0014
                speaker: 7850-286674
                start: 0.0
                text: SURE ENOUGH THERE HE CAME THROUGH THE SHALLOW WATER HIS WET BACK SHELL PARTLY
                  OUT OF IT AND SHINING IN THE SUNLIGHT
          offset: 0.0
        - cut:
            duration: 9.705
            features:
              channel_id: 0
              duration: 9.705
              type: fbank
              num_frames: 970
              num_features: 23
              recording_id: 2412-153948-0014
              start: 0.0
              storage_path: test/fixtures/mix_cut_test/feats/storage/5078e7eb-57a6-4000-b0f2-fa4bf9c52090.llc
              storage_type: lilcom
            id: 78bef88d-e62e-4cfa-9946-a1311442c6f7
            start: 0.0
            supervisions:
              - channel_id: 0
                duration: 9.705
                gender: f
                id: 2412-153948-0014
                language: null
                recording_id: 2412-153948-0014
                speaker: 2412-153948
                start: 0.0
                text: THERE WAS NO ONE IN THE WHOLE WORLD WHO HAD THE SMALLEST IDEA SAVE THOSE
                  WHO WERE THEMSELVES ON THE OTHER SIDE OF IT IF INDEED THERE WAS ANY ONE AT ALL
                  COULD I HOPE TO CROSS IT
          offset: 3.89
          snr: 20.0
      type: MixedCut

Mixed cuts literally consist of simple cuts, their feature descriptions, and their supervisions.
These are combined together when a user queries :class:`MixedCut` for supervisions, features, or duration.
Note that the first simple cut is missing an SNR field - it is optional (i.e. *None*).
That is because the semantics of 0 SNR are: re-scale one of the signals, so that the SNR between two signals is zero.
We denote no re-scaling by not specifying the SNR at all.

The amount of text in these manifests can be considerable in larger datasets, but they are highly compressible.
We support their automated (de-)compression with gzip - it's sufficient to add ".gz" at the end of filename
when writing or reading, both in Python classes and the CLI tools.

Python
******

Some examples of how cuts can be manipulated to create a desired dataset for model training.

.. code-block:: python

    cuts = CutSet.from_yaml('cuts.yml')
    # Reject short segments
    cuts = cuts.filter(lambda cut: cut.duration >= 3.0)
    # Pad short segments to 5 seconds.
    cuts = cuts.pad(desired_duration=5.0)
    # Truncate longer segments to 5 seconds.
    cuts = cuts.truncate(max_duration=5.0, offset_type='random')
    # Save cuts
    cuts.to_yaml('cuts-5s.yml')

CLI
***

Analogous examples of how to perform the same operations in the terminal:

.. code-block:: bash

    # Reject short segments
    lhotse yaml filter duration>=3.0 cuts.yml cuts-3s.yml
    # Pad short segments to 5 seconds.
    lhotse cut pad --duration 5.0 cuts-3s.yml cuts-5s-pad.yml
    # Truncate longer segments to 5 seconds.
    lhotse cut truncate --max-duration 5.0 --offset-type random cuts-5s-pad.yml cuts-5s.yml
