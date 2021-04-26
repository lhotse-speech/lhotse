from speechcolab.datasets import gigaspeech


def download_gigaspeech(
        target_dir: Pathlike = '.',
        dataset_parts: Optional[Union[str, Sequence[str]]] = "XS",
        force_download: Optional[bool] = False
) -> None:
    gigaspeech_data = gigaspeech.GigaSpeech(target_dir)
    gigaspeech_data.download(dataset_parts, force_download)
