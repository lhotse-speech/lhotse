import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.ami import download_ami, prepare_ami
from lhotse.utils import Pathlike

__all__ = ["ami"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--annotations",
    type=click.Path(),
    default=None,
    help="Provide if annotations are download in a different directory than corpus.",
)
@click.option(
    "--mic",
    type=click.Choice(["ihm", "ihm-mix", "sdm", "mdm"], case_sensitive=False),
    default="ihm",
    help="AMI microphone setting.",
)
@click.option(
    "--partition",
    type=click.Choice(
        ["scenario-only", "full-corpus", "full-corpus-asr"], case_sensitive=False
    ),
    default="full-corpus-asr",
    help="Data partition to use (see http://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml).",
)
@click.option(
    "--normalize-text",
    type=click.Choice(["none", "upper", "kaldi"], case_sensitive=False),
    default="kaldi",
    help="Type of text normalization to apply (kaldi style, by default)",
)
def ami(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    annotations: Pathlike,
    mic: str,
    partition: str,
    normalize_text: bool,
):
    """AMI data preparation."""
    prepare_ami(
        corpus_dir,
        annotations_dir=annotations,
        output_dir=output_dir,
        mic=mic,
        partition=partition,
        normalize_text=normalize_text,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--annotations",
    type=click.Path(),
    default=None,
    help="To download annotations in a different directory than corpus.",
)
@click.option(
    "--mic",
    type=click.Choice(["ihm", "ihm-mix", "sdm", "mdm"], case_sensitive=False),
    default="ihm",
    help="AMI microphone setting.",
)
@click.option(
    "--url",
    type=str,
    default="http://groups.inf.ed.ac.uk/ami",
    help="AMI data downloading URL.",
)
@click.option(
    "--force-download",
    type=bool,
    default=False,
    help="If True, download even if file is present.",
)
def ami(
    target_dir: Pathlike,
    annotations: Pathlike,
    mic: str,
    url: str,
    force_download: bool,
):
    """AMI download."""
    download_ami(
        target_dir,
        annotations_dir=annotations,
        mic=mic,
        url=url,
        force_download=force_download,
    )
