import copy
import json
import pickle
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

class InputAugExample(object):
    def __init__(self, guid, text_a, text_s=None, text_k=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_s = text_s
        self.text_k = text_k
        self.label = label



    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List["InputAugExample"]:
        """Load a set of input examples from a file"""
        with open(path, "rb") as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List["InputAugExample"], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, "wb") as fh:
            pickle.dump(examples, fh)

class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()

        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)


class InputFeatures(object):

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        label,
        mlm_labels=None,
       
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.mlm_labels = mlm_labels
      

    def __repr__(self):
        return str(self.to_json_string())

    def pretty_print(self, tokenizer):
        return (
            f"input_ids         = {tokenizer.convert_ids_to_tokens(self.input_ids)}\n"
            + f"attention_mask    = {self.attention_mask}\n"
            + f"token_type_ids    = {self.token_type_ids}\n"
            + f"mlm_labels        = {self.mlm_labels}\n"
            + f"label             = {self.label}"
        )

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True


def save_logits(path: str, logits: np.ndarray):
    """Save an array of logits to a file"""
    with open(path, "w") as fh:
        for example_logits in logits:
            fh.write(" ".join(str(logit) for logit in example_logits) + "\n")
    pass


def save_predictions(path: str, wrapper, results: Dict):
    """Save a sequence of predictions to a file"""
    predictions_with_idx = []

   
    inv_label_map = {idx: label for label, idx in wrapper.preprocessor.label_map.items()}
    if 'indices' in results:
        for idx, prediction_idx, actual_idx in zip(results["indices"], results["predictions"], results["labels"]):
            prediction = inv_label_map[prediction_idx]
            actual = inv_label_map[actual_idx]
            idx = idx.tolist() if isinstance(idx, np.ndarray) else int(idx)
            predictions_with_idx.append({"idx": idx, "predicted": prediction, "actual": actual})
    else:
        for prediction_idx, actual_idx, uid_idx in zip(results["predictions"], results["labels"], results['uid']):
            prediction = inv_label_map[prediction_idx]
            actual = inv_label_map[actual_idx]
            predictions_with_idx.append({"predicted": prediction, "actual": actual, 'uid':uid_idx})

    with open(path, "w", encoding="utf8") as fh:
        for line in predictions_with_idx:
            fh.write(json.dumps(line) + "\n")


