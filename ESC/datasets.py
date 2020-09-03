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
        pass

    def __len__(self) -> int:
        pass

    def download(self) -> None:
        pass

    def check_integrity(self):
        pass