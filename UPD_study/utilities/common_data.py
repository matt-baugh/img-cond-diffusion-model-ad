from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset, ABC):
    """
    Base class for all our datasets.
    """

    def __init__(self, config: Namespace) -> None:
        super().__init__()
        self.center = config.center
        self.aug_fn = config.aug_fn

    @abstractmethod
    def get_sample(self, idx) -> np.ndarray:
        """
        Return a sample from the dataset.
        """

    def __getitem__(self, idx) -> Union[Tensor, tuple[Tensor, Tensor]]:

        img = self.get_sample(idx)

        if self.aug_fn is None:
            # Center input
            if self.center:
                img = (img - 0.5) * 2

            return torch.FloatTensor(img)
        else:
            img, mask = self.aug_fn(img, self)
            if self.center:
                img = (img - 0.5) * 2

            return torch.FloatTensor(img), torch.FloatTensor(mask)


class GenericDataloader(DataLoader):
    """
    Generic Dataloader class to reduce boilerplate.
    Requires only Dataset object and configuration object for instantiation.

    Args:
        dataset (Dataset): dataset from which to load the data.
        config (Namespace): configuration object.
    """

    def __init__(self, dataset: Dataset, config: Namespace, shuffle: bool = True, drop_last: bool = False):
        super().__init__(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            pin_memory=False,
            num_workers=config.num_workers,
            drop_last=False)
