from typing import Optional, Sequence, Union
from lhotse.utils import Pathlike, is_module_available


def download_gigaspeech(
        target_dir: Pathlike = '.',
        dataset_parts: Optional[Union[str, Sequence[str]]] = "XS",
        force_download: Optional[bool] = False
) -> None:
    if is_module_available('speechcolab'):
        from speechcolab.datasets import gigaspeech

    gigaspeech_data = gigaspeech.GigaSpeech(target_dir)
    gigaspeech_data.download(dataset_parts, force_download)
