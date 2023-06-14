from abc import ABC, abstractmethod
import copy
import os

import torch
from torch.utils.data import Dataset


class InMemoryDataset(Dataset, ABC):
    """Base class for all datasets that are loaded into memory. 

    .. note::
        This class is an abstract class and should not be instantiated directly. All subclasses should implement the 
        ``process`` method, which should return a dictionary of attributes to be added to the dataset.

    Args:
        root (string): Root directory of dataset where the raw data is stored, and where the processed data will be 
            saved.
        name (string): Name of the dataset.
        transform (callable, optional): A function/transform that takes in a data sample and returns a transformed 
            version. This function will be applied each time the __getitem__ method is called. (default: None)
        force_process (bool, optional): If True, the dataset will be processed even if a processed version already
            exists. (default: False)
    """

    processed_dir = 'processed/'  # relative to root, can be overriden by subclasses

    def __init__(self, root, name, transform=None, force_process=False):
        super().__init__()
        self.root = root
        self.name = name
        self.transform = transform

        # check if already processed
        already_processed, filename = self._look_for_processed_file()

        # if not processed or force_process
        if not already_processed or force_process:
            # process and save data
            data_dict = self.process()
            self.save(data_dict)
        else:
            data_dict = self.load()

        self.__dict__.update(data_dict)

    def __getitem__(self, item):
        data = self.data_list[item]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data_list)

    def to(self, device):
        r"""Moves all data to the specified device. This is useful for moving all data to the GPU at once, as long as 
            the GPU has enough memory. This should avoid the CPU to GPU copy operation."""
        for data in self.data_list:
            data.to(device)

    @property
    def processed_filename(self):
        return os.path.join(self.root, self.processed_dir, '{}.pt'.format(self.name))

    def _look_for_processed_file(self):
        filename = self.processed_filename
        return os.path.exists(filename), filename

    def load(self):
        r"""Loads the processed data from the file."""
        filename = self.processed_filename
        print('filename', filename)
        processed = torch.load(filename)
        return processed

    def map(self, transform):
        r"""Returns a new dataset with the specified transform."""
        # shallow copy of the dataset
        new_dataset = copy.copy(self)
        assert new_dataset.transform is None
        new_dataset.transform = transform
        return new_dataset

    def save(self, data_dict):
        # make directory if needed
        os.path.makedirs(os.path.dirname(self.processed_filename), exist_ok=True)
        # save data to file
        torch.save(data_dict, self.processed_filename)

    @abstractmethod
    def process(self):
        pass
