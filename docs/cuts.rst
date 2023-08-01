Cuts
====

Overview
********

Audio cuts are one of the main Lhotse features.
Cut is a part of a recording, but it can be longer than a supervision segment, or even span multiple segments.
The regions without a supervision are just audio that we don't assume we know anything about - there may be silence,
noise, non-transcribed speech, etc.
Task-specific datasets can leverage this information to generate masks for such regions.

.. autoclass:: lhotse.cut.Cut
    :no-members:
    :no-special-members:
    :noindex:

.. autoclass:: lhotse.cut.CutSet
    :no-members:
    :no-special-members:
    :noindex:

Types of cuts
*************

There are three cut classes: :class:`~lhotse.cut.MonoCut`, :class:`~lhotse.cut.MixedCut`, and :class:`~lhotse.cut.PaddingCut` that are described below in more detail:

.. autoclass:: lhotse.cut.MonoCut
    :no-members:
    :no-special-members:
    :noindex:

.. autoclass:: lhotse.cut.MixedCut
    :no-members:
    :no-special-members:
    :noindex:


.. autoclass:: lhotse.cut.PaddingCut
    :no-members:
    :no-special-members:
    :noindex:

CLI
***

We provide a limited CLI to manipulate Lhotse manifests.
Some examples of how to perform manipulations in the terminal:

.. code-block:: bash

    # Reject short segments
    lhotse filter 'duration>=3.0' cuts.jsonl cuts-3s.jsonl
    # Pad short segments to 5 seconds.
    lhotse cut pad --duration 5.0 cuts-3s.jsonl cuts-5s-pad.jsonl
    # Truncate longer segments to 5 seconds.
    lhotse cut truncate --max-duration 5.0 --offset-type random cuts-5s-pad.jsonl cuts-5s.jsonl
