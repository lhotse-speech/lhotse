import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.notsofar import download_notsofar, prepare_notsofar
from lhotse.utils import Pathlike

__all__ = ["notsofar"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def notsofar(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """NOTSOFAR data preparation."""
    prepare_notsofar(
        corpus_dir,
        output_dir=output_dir,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--force-download",
    type=bool,
    is_flag=True,
    default=False,
    help="Force download.",
)
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    is_flag=False,
    multiple=True,
    default=("train", "dev", "test"),
    help="Dataset parts to download.",
)
@click.option(
    "--train-version",
    type=str,
    is_flag=False,
    default="240825.1_train",
    help="Train dataset version.",
)
@click.option(
    "--dev-version",
    type=str,
    is_flag=False,
    default="240825.1_dev1",
    help="Dev dataset version.",
)
@click.option(
    "--test-version",
    type=str,
    is_flag=False,
    default="240629.1_eval_small_with_GT",
    help="Test dataset version.",
)
def notsofar(
    target_dir: Pathlike,
    force_download: bool,
    dataset_parts: str,
    train_version: str,
    dev_version: str,
    test_version: str,
):
    """NOTSOFAR download."""
    download_notsofar(
        target_dir,
        parts=dataset_parts,
        train_version=train_version,
        dev_version=dev_version,
        test_version=test_version,
        force_download=force_download,
    )
