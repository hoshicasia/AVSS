import numpy as np
import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    result_batch["mix"] = torch.stack([item["mix"] for item in dataset_items])

    optional_keys = ["label_1", "label_2", "mouths_1", "mouths_2"]
    for key in optional_keys:
        first = dataset_items[0][key]
        if first is None:
            if all(item[key] is None for item in dataset_items):
                result_batch[key] = None
        else:
            result_batch[key] = torch.stack([item[key] for item in dataset_items])
    return result_batch
