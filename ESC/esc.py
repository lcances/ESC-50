import os
import time
import torchaudio
import torch.utils.data import datasetManager
from torch import Tensor
from torchaudio.datasets.utils import download_url, extract_archive, walk_files
import pandas
import numpy

# Default parameters
URL = {
    "esc-10": None,
    "esc-50": None,
    "esc-us": None,
}

CHECKSUMS = {
    "esc-10": None,
    "esc-50": None,
    "esc-us": None,
}

VERSION = "esc-10"
FOLDS = (1, 2, 3, 4, 5)

class ESC(Dataset):
    """
    ESC datasets

    Args:
        root (string): Root directory of datasets where directory
            ``ESC`` exists or will be saved to if download is set to True.
        version (string, optional): The version of the dataset to use.
            The available version are esc-10 and esc-50
        download (bool, optional): If true, download the dataset from the internet
            and puts it in root directory. If datasets is already downloaded, it is
            not downloaded again.
    """
    def __init__(self,
                root: str,
                version: str = VERSION,
                folds: tuple = FOLDS,
                download: bool = False) -> None:

        super().__init__()

        self.root = root
        self.version = version
        self.folds = folds
        self.download = download

    def __getitem__(self, index: int) -> Tuple[tensor, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (raw_audio, label).
        """
        pass

    def __len__(self) -> int:
        pass

    def download(self) -> None:
        """Download the dataset and extract the archive"""
        pass

    def check_integrity(self):
        """Check if the dataset already exist and if yes, if it is not corrupted.

        Returns:
            bool: False if the dataset doesn't exist or if it is corrupted.
        """
        pass