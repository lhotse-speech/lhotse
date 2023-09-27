import logging
import shutil
import tempfile

import numpy as np
import torch
from attr import frozen
from cytoolz.itertoolz import groupby
from tqdm import tqdm

from lhotse import CutSet, Recording
from lhotse.utils import fastcopy, is_module_available

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def diarize_segments_with_speechbrain(
    cuts: CutSet,
    device: str = "cpu",
    num_speakers: int = None,
    threshold: float = 0.5,
) -> CutSet:
    """
    This workflow uses SpeechBrain's pretrained speaker embedding model to compute speaker embeddings
    for each cut in the CutSet. The cuts for the same recording are then clustered using
    agglomerative hierarchical clustering, and the resulting cluster indices are used to create new cuts
    with the speaker labels.

    Please refer to https://huggingface.co/speechbrain/spkrec-xvect-voxceleb for more details
    about the speaker embedding extractor.

    :param manifest: a ``CutSet`` object.
    :param device: Where to run the inference (cpu, cuda, etc.).
    :param num_speakers: Number of speakers to cluster the cuts into. If not specified, we will use
        the threshold parameter to determine the number of speakers.
    :param threshold: The threshold for agglomerative clustering.
    :return: a new ``CutSet`` with speaker labels.
    """
    assert is_module_available("speechbrain"), (
        "This function expects SpeechBrain to be installed. "
        "You can install it via 'pip install speechbrain' "
    )

    assert is_module_available("sklearn"), (
        "This function expects scikit-learn to be installed. "
        "You can install it via 'pip install scikit-learn' "
    )

    from sklearn.cluster import AgglomerativeClustering
    from speechbrain.pretrained import EncoderClassifier

    threshold = None if num_speakers is not None else threshold
    dirpath = tempfile.mkdtemp()

    recordings, _, _ = cuts.decompose(dirpath, verbose=True)
    recordings = recordings.to_eager()
    recording_ids = frozenset(recordings.ids)

    logging.info("Saving cut recordings temporarily to disk...")
    cuts_ = []
    for cut in tqdm(cuts):
        save_path = f"{dirpath}/{cut.recording_id}.wav"
        _ = cut.save_audio(save_path)
        cuts_.append(fastcopy(cut, recording=Recording.from_file(save_path)))

    cuts_ = CutSet.from_cuts(cuts_).trim_to_supervisions(keep_overlapping=False)

    # Load the pretrained model
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
        run_opts={"device": device},
    )

    out_cuts = []

    for recording_id in tqdm(recording_ids, total=len(recording_ids)):
        logging.info(f"Processing recording {recording_id}...")
        embeddings = []
        reco_cuts = cuts_.filter(lambda c: c.recording_id == recording_id)
        num_cuts = len(frozenset(reco_cuts.ids))
        if num_cuts == 0:
            continue
        for cut in tqdm(reco_cuts, total=num_cuts):
            audio = torch.from_numpy(cut.load_audio())
            embedding = model.encode_batch(audio).cpu().numpy()
            embeddings.append(embedding.squeeze())

        embeddings = np.vstack(embeddings)
        clusterer = AgglomerativeClustering(
            n_clusters=num_speakers,
            affinity="euclidean",
            linkage="ward",
            distance_threshold=threshold,
        )
        clusterer.fit(embeddings)

        # Assign the cluster indices to the cuts
        for cut, cluster_idx in zip(reco_cuts, clusterer.labels_):
            sup = fastcopy(cut.supervisions[0], speaker=f"spk{cluster_idx}")
            out_cuts.append(
                fastcopy(
                    cut,
                    recording=recordings[cut.recording_id],
                    supervisions=[sup],
                )
            )

    # Remove the temporary directory
    shutil.rmtree(dirpath)

    return CutSet.from_cuts(out_cuts)
