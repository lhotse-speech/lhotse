import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_grid, prepare_grid
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--with-supervisions/--no-supervisions",
    default=True,
    help="Note: using supervisions might discard some recordings that do not have them.",
)
@click.option("-j", "--jobs", default=1, type=int, help="The number of parallel jobs.")
def grid(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    with_supervisions: bool,
    jobs: int,
):
    """Grid audio-visual speech corpus preparation."""
    prepare_grid(
        corpus_dir,
        output_dir=output_dir,
        with_supervisions=with_supervisions,
        num_jobs=jobs,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def grid(
    target_dir: Pathlike,
):
    """Grid audio-visual speech corpus download."""
    download_grid(target_dir)
