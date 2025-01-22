import logging
import os
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Generator, List, Optional, Union

import numpy as np
from tqdm import tqdm

from lhotse import CutSet, MonoCut, RecordingSet, SupervisionSegment
from lhotse.utils import fastcopy, is_module_available, resumable_download


class ComputeScore:
    def __init__(self, primary_model_path) -> None:
        import onnxruntime as ort

        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.SAMPLING_RATE = 16000
        self.INPUT_LENGTH = 9.01

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        import librosa

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_mos):
        if is_personalized_mos:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, manifest, is_personalized_mos):
        fs = self.SAMPLING_RATE
        audio = manifest.resample(fs).load_audio()
        len_samples = int(self.INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - self.INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int(
                    (idx + self.INPUT_LENGTH) * hop_len_samples
                )
            ]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            oi = {"input_1": input_features}
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_mos
            )
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)

        return manifest, {
            "OVRL": np.mean(predicted_mos_ovr_seg),
            "SIG": np.mean(predicted_mos_sig_seg),
            "BAK": np.mean(predicted_mos_bak_seg),
        }


def download_model(
    is_personalized_mos: bool = False,
    download_root: Optional[str] = None,
) -> str:
    download_root = download_root if download_root is not None else "/tmp"
    url = (
        "https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/DNSMOS/pDNSMOS/sig_bak_ovr.onnx"
        if is_personalized_mos
        else "https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
    )
    filename = os.path.join(download_root, "sig_bak_ovr.onnx")
    resumable_download(url, filename=filename)
    return filename


def annotate_dnsmos(
    manifest: Union[RecordingSet, CutSet],
    is_personalized_mos: bool = False,
    download_root: Optional[str] = None,
) -> Generator[MonoCut, None, None]:
    """
    Use Microsoft DNSMOS P.835 prediction model to annotate either RECORDINGS_MANIFEST, RECORDINGS_DIR, or CUTS_MANIFEST.
    It will predict DNSMOS P.835 score including SIG, NAK, and OVRL.

    See the original repo for more details: https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS

    :param manifest: a ``RecordingSet`` or ``CutSet`` object.
    :param is_personalized_mos: flag to indicate if personalized MOS score is needed or regular.
    :param download_root: if specified, the model will be downloaded to this directory. Otherwise,
        it will be downloaded to /tmp.
    :return: a generator of cuts (use ``CutSet.open_writer()`` to write them).
    """
    assert is_module_available("librosa"), (
        "This function expects librosa to be installed. "
        "You can install it via 'pip install librosa'"
    )

    assert is_module_available("onnxruntime"), (
        "This function expects onnxruntime to be installed. "
        "You can install it via 'pip install onnxruntime'"
    )

    if isinstance(manifest, RecordingSet):
        yield from _annotate_recordings(
            manifest,
            is_personalized_mos,
            download_root,
        )
    elif isinstance(manifest, CutSet):
        yield from _annotate_cuts(
            manifest,
            is_personalized_mos,
            download_root,
        )
    else:
        raise ValueError("The ``manifest`` must be either a RecordingSet or a CutSet.")


def _annotate_recordings(
    recordings: RecordingSet,
    is_personalized_mos: bool = False,
    download_root: Optional[str] = None,
):
    """
    Helper function that annotates a RecordingSet with DNSMOS P.835 prediction model.
    """
    primary_model_path = download_model(is_personalized_mos, download_root)
    compute_score = ComputeScore(primary_model_path)

    with ThreadPoolExecutor() as ex:
        futures = []
        for recording in tqdm(recordings, desc="Distributing tasks"):
            if recording.num_channels > 1:
                logging.warning(
                    f"Skipping recording '{recording.id}'. It has {recording.num_channels} channels, "
                    f"but we currently only support mono input."
                )
                continue
            futures.append(ex.submit(compute_score, recording, is_personalized_mos))

        for future in tqdm(futures, desc="Processing"):
            recording, result = future.result()
            supervision = SupervisionSegment(
                id=recording.id,
                recording_id=recording.id,
                start=0,
                duration=recording.duration,
            )
            cut = MonoCut(
                id=recording.id,
                start=0,
                duration=recording.duration,
                channel=0,
                recording=recording,
                supervisions=[supervision],
                custom=result,
            )
            yield cut


def _annotate_cuts(
    cuts: CutSet,
    is_personalized_mos: bool = False,
    download_root: Optional[str] = None,
):
    """
    Helper function that annotates a CutSet with DNSMOS P.835 prediction model.
    """
    primary_model_path = download_model(is_personalized_mos, download_root)
    compute_score = ComputeScore(primary_model_path)

    with ThreadPoolExecutor() as ex:
        futures = []
        for cut in tqdm(cuts, desc="Distributing tasks"):
            if cut.num_channels > 1:
                logging.warning(
                    f"Skipping cut '{cut.id}'. It has {cut.num_channels} channels, "
                    f"but we currently only support mono input."
                )
                continue
            futures.append(ex.submit(compute_score, cut, is_personalized_mos))

        for future in tqdm(futures, desc="Processing"):
            cut, result = future.result()
            if cut.custom is not None:
                cut.custom.update(result)
            else:
                cut.custom = result
            yield cut
