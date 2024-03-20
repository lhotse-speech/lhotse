from pathlib import Path
from typing import List, Optional

import click

from lhotse.bin.modes.cli_base import cli
from lhotse.cut import CutSet, append_cuts, mix_cuts
from lhotse.serialization import load_manifest_lazy_or_eager
from lhotse.utils import Pathlike


@cli.group()
def cut():
    """Group of commands used to create CutSets."""
    pass


@cut.command()
@click.argument("output_cut_manifest", type=click.Path(allow_dash=True))
@click.option(
    "-r",
    "--recording-manifest",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional recording manifest - will be used to attach the recordings to the cuts.",
)
@click.option(
    "-f",
    "--feature-manifest",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional feature manifest - will be used to attach the features to the cuts.",
)
@click.option(
    "-s",
    "--supervision-manifest",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional supervision manifest - will be used to attach the supervisions to the cuts.",
)
@click.option(
    "--force-eager",
    is_flag=True,
    help="Force reading full manifests into memory before creating the manifests "
    "(useful when you are not sure about the input manifest sorting).",
)
def simple(
    output_cut_manifest: Pathlike,
    recording_manifest: Optional[Pathlike],
    feature_manifest: Optional[Pathlike],
    supervision_manifest: Optional[Pathlike],
    force_eager: bool,
):
    """
    Create a CutSet stored in OUTPUT_CUT_MANIFEST. Depending on the provided options, it may contain any combination
    of recording, feature and supervision manifests.
    Either RECORDING_MANIFEST or FEATURE_MANIFEST has to be provided.
    When SUPERVISION_MANIFEST is provided, the cuts time span will correspond to that of the supervision segments.
    Otherwise, that time span corresponds to the one found in features, if available, otherwise recordings.

    .. hint::
        ``--force-eager`` must be used when the RECORDING_MANIFEST is not sorted by recording ID.
    """
    supervision_set, feature_set, recording_set = [
        load_manifest_lazy_or_eager(p) if p is not None else None
        for p in (supervision_manifest, feature_manifest, recording_manifest)
    ]

    if (
        all(
            m is None or m.is_lazy
            for m in (supervision_set, feature_set, recording_set)
        )
        and not force_eager
    ):
        # Create the CutSet lazily; requires sorting by recording_id
        CutSet.from_manifests(
            recordings=recording_set,
            supervisions=supervision_set,
            features=feature_set,
            output_path=output_cut_manifest,
            lazy=True,
        )
    else:
        cut_set = CutSet.from_manifests(
            recordings=recording_set, supervisions=supervision_set, features=feature_set
        )
        cut_set.to_file(output_cut_manifest)


@cut.command()
@click.argument("cuts", type=click.Path(exists=True, dir_okay=False, allow_dash=True))
@click.argument("output_cuts", type=click.Path(allow_dash=True))
@click.option(
    "--keep-overlapping/--discard-overlapping",
    type=bool,
    default=True,
    help="""when `False`, it will discard parts of other supervisions that overlap with the
            main supervision. In the illustration, it would discard `Sup2` in `Cut1` and `Sup1` in `Cut2`.""",
)
@click.option(
    "-d",
    "--min-duration",
    type=float,
    default=None,
    help="""An optional duration in seconds; specifying this argument will extend the cuts
            that would have been shorter than `min_duration` with actual acoustic context in the recording/features.
            If there are supervisions present in the context, they are kept when `keep_overlapping` is true.
            If there is not enough context, the returned cut will be shorter than `min_duration`.
            If the supervision segment is longer than `min_duration`, the return cut will be longer.""",
)
@click.option(
    "-c",
    "--context-direction",
    type=click.Choice(["center", "left", "right", "random"]),
    default="center",
    help="""Which direction should the cut be expanded towards to include context.
            The value of "center" implies equal expansion to left and right;
            random uniformly samples a value between "left" and "right".""",
)
@click.option(
    "--keep-all-channels/--discard-extra-channels",
    type=bool,
    default=False,
    help="""If ``True``, the output cut will have the same channels as the input cut. By default,
            the trimmed cut will have the same channels as the supervision.""",
)
def trim_to_supervisions(
    cuts: Pathlike,
    output_cuts: Pathlike,
    keep_overlapping: bool,
    min_duration: Optional[float],
    context_direction: str,
    keep_all_channels: bool,
):
    """
    Splits each input cut into as many cuts as there are supervisions.
    These cuts have identical start times and durations as the supervisions.
    When there are overlapping supervisions, they can be kept or discarded with options.
    """
    cuts = CutSet.from_file(cuts)

    with CutSet.open_writer(output_cuts) as writer:
        for cut in cuts.trim_to_supervisions(
            keep_overlapping=keep_overlapping,
            min_duration=min_duration,
            context_direction=context_direction,
            keep_all_channels=keep_all_channels,
        ):
            writer.write(cut)


@cut.command()
@click.argument("cuts", type=click.Path(exists=True, dir_okay=False, allow_dash=True))
@click.argument("output_cuts", type=click.Path(allow_dash=True))
@click.option(
    "--type", type=str, default="word", help="Alignment type to use for trimming"
)
@click.option(
    "--max-pause",
    type=float,
    default=0.0,
    help="Merge alignments separated by a pause shorter than this value",
)
@click.option(
    "--delimiter",
    "-d",
    type=str,
    default=" ",
    help="Delimiter to use for concatenating alignment symbols for merging",
)
@click.option(
    "--keep-all-channels/--discard-extra-channels",
    type=bool,
    default=False,
    help="""If ``True``, the output cut will have the same channels as the input cut. By default,
            the trimmed cut will have the same channels as the supervision.""",
)
def trim_to_alignments(
    cuts: Pathlike,
    output_cuts: Pathlike,
    type: str,
    max_pause: float,
    delimiter: str,
    keep_all_channels: bool,
):
    """
    Return a new CutSet with Cuts that have identical spans as the alignments of
    type `type`. An additional `max_pause` is allowed between the alignments to
    merge contiguous alignment items.

    For the case of a multi-channel cut with multiple alignments, we can either trim
    while respecting the supervision channels (in which case output cut has the same channels
    as the supervision) or ignore the channels (in which case output cut has the same channels
    as the input cut).
    """
    cuts = CutSet.from_file(cuts)

    with CutSet.open_writer(output_cuts) as writer:
        for cut in cuts.trim_to_alignments(
            type=type,
            max_pause=max_pause,
            delimiter=delimiter,
            keep_all_channels=keep_all_channels,
        ):
            writer.write(cut)


@cut.command()
@click.argument("cuts", type=click.Path(exists=True, dir_okay=False, allow_dash=True))
@click.argument("output_cuts", type=click.Path(allow_dash=True))
@click.option(
    "--max-pause",
    type=float,
    default=0.0,
    help="Merge supervision groups separated by a pause shorter than this value",
)
def trim_to_supervision_groups(
    cuts: Pathlike,
    output_cuts: Pathlike,
    max_pause: float,
):
    """
    Return a new CutSet with Cuts that have identical spans as the supervision groups.
    An additional `max_pause` is allowed to merge contiguous supervision groups.

    A supervision group is defined as a set of supervisions that are overlapping or
    separated by a pause shorter than `max_pause`.
    """
    cuts = CutSet.from_file(cuts)

    with CutSet.open_writer(output_cuts) as writer:
        for cut in cuts.trim_to_supervision_groups(max_pause=max_pause):
            writer.write(cut)


@cut.command()
@click.argument("cut_manifests", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.argument("output_cut_manifest", type=click.Path())
def mix_sequential(cut_manifests: List[Pathlike], output_cut_manifest: Pathlike):
    """
    Create a CutSet stored in OUTPUT_CUT_MANIFEST by iterating jointly over CUT_MANIFESTS and mixing the Cuts
    on the same positions. E.g. the first output cut is created from the first cuts in each input manifest.
    The mix is performed by summing the features from all Cuts.
    If the CUT_MANIFESTS have different number of Cuts, the mixing ends when the shorter manifest is depleted.
    """
    cut_manifests = [CutSet.from_file(path) for path in cut_manifests]
    with CutSet.open_writer(output_cut_manifest) as w:
        for cuts in zip(*cut_manifests):
            w.write(mix_cuts(cuts))


@cut.command()
@click.argument("cut_manifests", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.argument("output_cut_manifest", type=click.Path())
def mix_by_recording_id(cut_manifests: List[Pathlike], output_cut_manifest: Pathlike):
    """
    Create a CutSet stored in OUTPUT_CUT_MANIFEST by matching the Cuts from CUT_MANIFESTS by their recording IDs
    and mixing them together.
    """
    from cytoolz.itertoolz import groupby

    from lhotse.manipulation import combine

    all_cuts = combine(*[CutSet.from_file(path) for path in cut_manifests])
    recording_id_to_cuts = groupby(lambda cut: cut.recording_id, all_cuts)
    mixed_cut_set = CutSet.from_cuts(
        mix_cuts(cuts) for recording_id, cuts in recording_id_to_cuts.items()
    )
    mixed_cut_set.to_file(output_cut_manifest)


@cut.command(context_settings=dict(show_default=True))
@click.argument(
    "cut_manifest", type=click.Path(exists=True, dir_okay=False, allow_dash=True)
)
@click.argument("output_cut_manifest", type=click.Path(allow_dash=True))
@click.option(
    "--preserve-id",
    is_flag=True,
    help="Should the cuts preserve IDs (by default, they will get new, random IDs)",
)
@click.option(
    "-d",
    "--max-duration",
    type=float,
    required=True,
    help="The maximum duration in seconds of a cut in the resulting manifest.",
)
@click.option(
    "-o",
    "--offset-type",
    type=click.Choice(["start", "end", "random"]),
    default="start",
    help='Where should the truncated cut start: "start" - at the start of the original cut, '
    '"end" - MAX_DURATION before the end of the original cut, '
    '"random" - randomly choose somewhere between "start" and "end" options.',
)
@click.option(
    "--keep-overflowing-supervisions/--discard-overflowing-supervisions",
    type=bool,
    default=False,
    help="When a cut is truncated in the middle of a supervision segment, should the supervision be kept.",
)
def truncate(
    cut_manifest: Pathlike,
    output_cut_manifest: Pathlike,
    preserve_id: bool,
    max_duration: float,
    offset_type: str,
    keep_overflowing_supervisions: bool,
):
    """
    Truncate the cuts in the CUT_MANIFEST and write them to OUTPUT_CUT_MANIFEST.
    Cuts shorter than MAX_DURATION will not be modified.
    """
    cut_set = CutSet.from_file(cut_manifest)
    truncated_cut_set = cut_set.truncate(
        max_duration=max_duration,
        offset_type=offset_type,
        keep_excessive_supervisions=keep_overflowing_supervisions,
        preserve_id=preserve_id,
    )
    truncated_cut_set.to_file(output_cut_manifest)


@cut.command()
@click.argument("cut_manifests", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.argument("output_cut_manifest", type=click.Path())
def append(
    cut_manifests: List[Pathlike],
    output_cut_manifest: Pathlike,
):
    """
    Create a new CutSet by appending the cuts in CUT_MANIFESTS. CUT_MANIFESTS are iterated position-wise (the
    cuts on i'th position in each manfiest are appended to each other).
    The cuts are appended in the order in which they appear in the
    input argument list.
    If CUT_MANIFESTS have different lengths, the script stops once the shortest CutSet is depleted.
    """
    cut_sets = [CutSet.from_file(path) for path in cut_manifests]
    with CutSet.open_writer(output_cut_manifest) as w:
        for cuts in zip(*cut_sets):
            w.write(append_cuts(cuts))


@cut.command()
@click.argument(
    "cut_manifest", type=click.Path(exists=True, dir_okay=False, allow_dash=True)
)
@click.argument("output_cut_manifest", type=click.Path(allow_dash=True))
@click.option(
    "-d",
    "--duration",
    default=None,
    type=float,
    help="Desired duration of cuts after padding. "
    "Cuts longer than this won't be affected. "
    "By default, pad to the longest cut duration found in CUT_MANIFEST.",
)
def pad(
    cut_manifest: Pathlike, output_cut_manifest: Pathlike, duration: Optional[float]
):
    """
    Create a new CutSet by padding the cuts in CUT_MANIFEST. The cuts will be right-padded, i.e. the padding
    is placed after the signal ends.
    """
    cut_set = CutSet.from_file(cut_manifest)
    padded_cut_set = cut_set.pad(duration=duration)
    padded_cut_set.to_file(output_cut_manifest)


@cut.command()
@click.argument("cutset", type=click.Path(exists=True, dir_okay=False, allow_dash=True))
@click.argument("output", type=click.Path())
def decompose(cutset: Pathlike, output: Pathlike):
    """
    \b
    Decompose CUTSET into:
        * recording set (recordings.jsonl.gz)
        * feature set (features.jsonl.gz)
        * supervision set (supervisions.jsonl.gz)

    If any of these are not preset in any of the cuts,
    the corresponding file for them will be empty.
    """
    CutSet.from_file(cutset).decompose(output_dir=Path(output), verbose=True)


@cut.command()
@click.argument("cutset", type=click.Path(exists=True, dir_okay=False, allow_dash=True))
def describe(cutset: Pathlike):
    """
    Describe some statistics of CUTSET, such as the total speech and audio duration.
    """
    CutSet.from_file(cutset).describe()


@cut.command(context_settings=dict(show_default=True))
@click.argument("cutset", type=click.Path(exists=True, dir_okay=False, allow_dash=True))
@click.argument("wspecifier", type=str)
@click.option(
    "-s",
    "--shard-size",
    type=int,
    help="Number of cuts per shard (sharding disabled if not defined).",
)
@click.option(
    "-f",
    "--audio-format",
    type=str,
    default="flac",
    help="Format in which the audio is encoded (uses torchaudio available formats).",
)
@click.option(
    "--audio/--no-audio",
    default=True,
    help="Should we load and add audio data.",
)
@click.option(
    "--features/--no-features",
    default=True,
    help="Should we load and add feature data.",
)
@click.option(
    "--custom/--no-custom",
    default=True,
    help="Should we load and add custom data.",
)
@click.option(
    "--fault-tolerant/--stop-on-fail",
    default=True,
    help="Should we omit the cuts for which loading data failed, or stop the execution.",
)
def export_to_webdataset(
    cutset: Pathlike,
    wspecifier: str,
    shard_size: Optional[int],
    audio_format: str,
    audio: bool,
    features: bool,
    custom: bool,
    fault_tolerant: bool,
):
    """
    Export CUTS into a WebDataset tarfile, or a collection of tarfile shards, as specified by
    WSPECIFIER.

    \b
    WSPECIFIER can be:
    - a regular path (e.g., "data/cuts.tar"),
    - a path template for sharding (e.g., "data/shard-06%d.tar"), or
    - a "pipe:" expression (e.g., "pipe:gzip -c > data/shard-06%d.tar.gz").

    The resulting CutSet contains audio/feature data in addition to metadata, and can be read in
    Python using 'CutSet.from_webdataset' API.

    This function is useful for I/O intensive applications where random reads are too slow, and
    a one-time lengthy export step that enables fast sequential reading is preferable.

    See the WebDataset project for more information: https://github.com/webdataset/webdataset
    """
    from lhotse.dataset.webdataset import export_to_webdataset as export_

    cuts = CutSet.from_file(cutset)
    assert isinstance(
        cuts, CutSet
    ), f"Only CutSet can be exported to WebDataset format (got: {type(cuts)} from '{cutset}')"

    export_(
        cuts=cuts,
        output_path=wspecifier,
        shard_size=shard_size,
        audio_format=audio_format,
        load_audio=audio,
        load_features=features,
        load_custom=custom,
        fault_tolerant=fault_tolerant,
    )


@cut.command()
@click.argument("cutset", type=click.Path(exists=True, dir_okay=False, allow_dash=True))
@click.option(
    "-b", "--num-buckets", default=30, type=int, help="Desired number of buckets."
)
@click.option(
    "-s",
    "--sample",
    default=None,
    type=int,
    help="How many samples to use for estimation (first N, by default use full cutset).",
)
def estimate_bucket_bins(
    cutset: Pathlike, num_buckets: int, sample: Optional[int]
) -> None:
    """
    Estimate duration bins for dynamic bucketing.
    Prints a Python list of num_buckets-1 floats (seconds) which constitute the boundaries between buckets.
    The bins are estimated in such a way so that each bucket has a roughly equal total duration of data.
    """
    from lhotse.dataset.sampling.dynamic_bucketing import estimate_duration_buckets

    cuts = load_manifest_lazy_or_eager(cutset, manifest_cls=CutSet)
    if sample is not None:
        cuts = cuts.subset(first=sample)
    click.echo(estimate_duration_buckets(cuts, num_buckets=num_buckets))
