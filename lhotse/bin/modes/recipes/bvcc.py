import click
from lhotse.bin.modes import download, prepare
from lhotse.recipes import download_bvcc, prepare_bvcc
from lhotse.utils import Pathlike

__all__ = ["bvcc"]


@prepare.command()
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("-nj", "--num_jobs", type=int, default=1)
def bvcc(corpus_dir: Pathlike, output_dir: Pathlike, num_jobs):
    """BVCC data preparation.

    CORPUS_DIR should contain the following dir structure

        ./phase1-main/README
        ./phase1-main/DATA/sets/*
        ./phase1-main/DATA/wav/*
        ...

        ./phase1-ood/README
        ./phase1-ood/DATA/sets/
        ./phase1-ood/DATA/wav/
        ...

    Check the READMEs for details.

    See 'lhotse download bvcc' for links to instructions how to obtain the corpus.
    """
    prepare_bvcc(corpus_dir, output_dir=output_dir, num_jobs=num_jobs)


@download.command()
def bvcc():
    """BVCC/VoiceMOS challange data cannot be downloaded.

    See info and instructions how to obtain BVCC dataset used for VoiceMOS challange:
    - https://arxiv.org/abs/2105.02373
    - https://nii-yamagishilab.github.io/ecooper-demo/VoiceMOS2022/index.html
    - https://codalab.lisn.upsaclay.fr/competitions/695
    """
    download_bvcc(
        target_dir="Not needed - just prints the docstring. Hopefully the license will be lifted."
    )
