import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from lhotse.cut import CutSet
from lhotse.utils import Pathlike

EPS = 1e-8


class SpeechRecognitionDataset(Dataset):
    """
    The PyTorch Dataset for the speech recognition task.
    Returns a dict of:

    .. code-block::

        {
            'features': (T x F) tensor
            'text': string
        }

    In the future, will be extended by graph supervisions.
    """

    def __init__(
            self,
            cuts: CutSet,
            root_dir: Optional[Pathlike] = None
    ):
        super().__init__()
        self.cuts = cuts.cuts
        self.root_dir = Path(root_dir) if root_dir else None
        self.cut_ids = list(self.cuts.keys())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        cut = self.cuts[cut_id]

        features = torch.from_numpy(cut.load_features(root_dir=self.root_dir))
        # We assume there is only one supervision for each cut in ASR tasks (for now)
        if len(cut.supervisions) > 1:
            logging.warning(
                "More than one supervision in ASR task! Selected the first one and ignored others.")
        supervision = cut.supervisions[0]
        if supervision.pychain_graph is not None:
            pychain_graph = supervision.pychain_graph.load()

        return {
            'features': features,
            'text': supervision.text
            'pychain_graph': pychain_graph if supervision.pychain_graph is not None
        }

    def __len__(self) -> int:
        return len(self.cut_ids)


""" These functions are copied from https://github.com/YiwenShaoStephen/pychain_example/blob/master/dataset.py
    They do not perfectly fit with SpeechRecognitionDataset but would probably still be useful and necessary here
    especially when people tend to load this SpeechRecognitionDataset with a pytorch Dataloader.
    As the default pytorch dataloader can only batchify tensor-like 
    (e.g. numpy array or pytorch tensor) items from the __getitem__ function but we have a dictionary output in
    SpeechRecognitionDataset so that we need to give customized dataloader here (another option is to leave it to
    the users themselves to decide how to load data). All these usages below are quite basic and flexible and can
    be replaced/re-writtern by other functions. The only special thing here is to use ChainGraphBatch class to 
    batchify a list of ChainGraph.
"""


def _collate_fn_train(batch):
    # sort the batch by its feature length in a descending order
    batch = sorted(
        batch, key=lambda sample: sample[1], reverse=True)
    max_seqlength = batch[0][1]
    feat_dim = batch[0][0].size(1)
    minibatch_size = len(batch)
    feats = torch.zeros(minibatch_size, max_seqlength, feat_dim)
    feat_lengths = torch.zeros(minibatch_size, dtype=torch.int)
    graph_list = []
    utt_ids = []
    max_num_transitions = 0
    max_num_states = 0
    for i in range(minibatch_size):
        feat, length, utt_id, graph = batch[i]
        feats[i, :length, :].copy_(feat)
        feat_lengths[i] = length
        graph_list.append(graph)
        if graph.num_transitions > max_num_transitions:
            max_num_transitions = graph.num_transitions
        if graph.num_states > max_num_states:
            max_num_states = graph.num_states
            utt_ids.append(utt_id)

    graphs = ChainGraphBatch(
        graph_list, max_num_transitions=max_num_transitions, max_num_states=max_num_states)
    return feats, feat_lengths, utt_ids, graphs


def _collate_fn_test(batch):
    # sort the batch by its feature length in a descending order
    batch = sorted(
        batch, key=lambda sample: sample[1], reverse=True)
    max_seqlength = batch[0][1]
    feat_dim = batch[0][0].size(1)
    minibatch_size = len(batch)
    feats = torch.zeros(minibatch_size, max_seqlength, feat_dim)
    feat_lengths = torch.zeros(minibatch_size, dtype=torch.int)
    utt_ids = []
    for i in range(minibatch_size):
        feat, length, utt_id = batch[i]
        feats[i, :length, :].copy_(feat)
        feat_lengths[i] = length
        utt_ids.append(utt_id)
    return feats, feat_lengths, utt_ids


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for ChainDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        if self.dataset.train:
            self.collate_fn = _collate_fn_train
        else:
            self.collate_fn = _collate_fn_test


class BucketSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        codes from deepspeech.pytorch 
        https://github.com/SeanNaren/deepspeech.pytorch/blob/master/data/data_loader.py
        """
        super(BucketSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size]
                     for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)
