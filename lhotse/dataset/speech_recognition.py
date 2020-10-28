from itertools import chain
from typing import Dict

import torch
from torch.utils.data import Dataset

from lhotse.cut import CutSet

EPS = 1e-8


class SpeechRecognitionDataset(Dataset):
    """
    The PyTorch Dataset for the speech recognition task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'features': (T x F) tensor,
            'text': string,
            'supervisions_mask': (T) tensor
        }

    The ``supervisions_mask`` field is a mask that specifies which frames are covered by a supervision
    by assigning a value of 1 (in this case: segments with transcribed speech contents),
    and which are not by asigning a value of 0 (in this case: padding, contextual noise,
    or in general the acoustic context without transcription).

    In the future, will be extended by graph supervisions.
    """

    def __init__(self, cuts: CutSet):
        super().__init__()
        self.cuts = cuts
        self.cut_ids = list(self.cuts.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        cut = self.cuts[cut_id]

        features = torch.from_numpy(cut.load_features())
        mask = torch.from_numpy(cut.supervisions_feature_mask())

        # There should be only one supervision because we expect that trim_to_supervisions() was called,
        # or the dataset was created from pre-segment recordings
        assert len(cut.supervisions) == 1, "SpeechRecognitionDataset does not support multiple supervisions yet. " \
                                           "Use CutSet.trim_to_supervisions() to cut long recordings into short " \
                                           "supervisions segment, and follow up with either .pad(), " \
                                           ".truncate(), and possibly .filter() to make sure that all cuts " \
                                           "have a uniform duration."

        return {
            'features': features,
            'text': cut.supervisions[0].text,
            'supervisions_mask': mask
        }

    def __len__(self) -> int:
        return len(self.cut_ids)


class K2SpeechRecognitionDataset(Dataset):
    """
    .. warning::
        This is just a draft of how Lhotse can interact with K2 - it's under active development.

    The PyTorch Dataset for the speech recognition task using K2 library.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'features': (T x F) tensor,
            'supervisions': List[Dict] -> [
                {
                    'text': string,
                    'fsa': k2.Fsa,
                    'start_frame': int,
                    'end_frame': int
                } (multiplied N times, for each of the N supervisions present in the Cut)
            ],
            'supervisions_mask': (T) tensor
        }

    The ``supervisions_mask`` field is a mask that specifies which frames are covered by a supervision
    by assigning a value of 1 (in this case: segments with transcribed speech contents),
    and which are not by asigning a value of 0 (in this case: padding, contextual noise,
    or in general the acoustic context without transcription).
    """

    def __init__(self, cuts: CutSet):
        import k2
        super().__init__()
        self.cuts = cuts
        self.cut_ids = list(self.cuts.ids)

        # We're creating a vocabulary of all characters present in the training CutSet.
        # To do that we iterate through all the supervisions, and use chain.from_iterable
        # to convert a sequence of their 'text' fields into a flat iterable of characters.

        # Add special symbols first
        self.vocab = ['<eps>', '<sil>', '<unk>'] + sorted(
            # Add symbols present in the corpus
            set(
                chain.from_iterable(
                    list(supervision.text)
                    for cut in self.cuts
                    for supervision in cut.supervisions
                )
            )
        )
        self.symbol_table = k2.SymbolTable(
            _id2sym={i: tok for i, tok in enumerate(self.vocab)},
            _sym2id={tok: i for i, tok in enumerate(self.vocab)}
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import k2
        cut_id = self.cut_ids[idx]
        cut = self.cuts[cut_id]

        features = torch.from_numpy(cut.load_features())
        mask = torch.from_numpy(cut.supervisions_feature_mask())

        sil_id = self.symbol_table.get('<sil>')
        eps_id = self.symbol_table.get('<eps>')

        supervision_fsas = []
        for supervision in cut.supervisions:
            # TODO(pzelasko): I am not sure how to attach the symbol table to the FSA object
            #                 (or whether that's need at all?)

            # Creating the core supervision FSA corresponding to the supervision text.
            fsa = k2.linear_fsa([self.symbol_table.get(c) for c in supervision.text])

            # TODO: Adding optional silence at beginning and end.

            # TODO(pzelasko): Should be possible to create it with sth like:
            #   silfsa = k2.union(
            #     k2.linear_fsa([sil_id]), k2.linear_fsa([eps_id])
            #   ).closure()
            silfsa = k2.Fsa.from_str(f'''0 1 {sil_id} 0
                                         0 1 {eps_id} 0
                                         1 0 {sil_id} 0
                                         1''')
            # TODO(pzelasko): is it possible to do:
            #   training_fsa = silfsa + fsa + silfsa
            #  or:
            #   training_fsa = k2.concat(silfsa, fsa, silfsa)
            #  ?
            supervision_fsas.append(fsa)

        # TODO(pzelasko): I think 'supervision_fsas' could be "joined" (connected) with
        #                 some sort of self-looped union of <eps>, <sil> and <unk>
        #                 (or maybe a sigma star, i.e. the full vocabulary) in between...

        return {
            'features': features,
            'supervisions': [
                {
                    'text': sup.text,
                    'fsa': fsa,
                    'start_frame': round(sup.start / cut.frame_shift),
                    'end_frame': round(sup.end / cut.frame_shift),
                } for sup, fsa in zip(
                    # CutSet's supervisions can exceed the cut, when the cut starts/ends in the middle
                    # of a supervision (they would have relative times e.g. -2 seconds start, meaning
                    # it started 2 seconds before the Cut starts). We use s.trim() to get rid of that
                    # property, ensuring the supervision time span does not exceed that of the cut.
                    (s.trim(cut.duration) for s in cut.supervisions),
                    supervision_fsas
                )
            ],
            'supervisions_mask': mask
        }

    def __len__(self) -> int:
        return len(self.cut_ids)
