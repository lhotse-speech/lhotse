import click

from lhotse.bin.modes import prepare
from lhotse.recipes.emilia import prepare_emilia
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-l",
    "--lang",
    type=str,
    help="The language to process. Valid values: zh, en, ja, ko, de, fr",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def emilia(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    lang: str,
    num_jobs: int = 1,
):
    """Prepare the Emilia corpus manifests."""
    prepare_emilia(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        lang=lang,
        num_jobs=num_jobs,
    )
