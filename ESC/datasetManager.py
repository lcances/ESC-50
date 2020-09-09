import os
from typing import Union

from numpy.core._multiarray_umath import ndarray

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np

import librosa
import tqdm
import time
import h5py
import pandas as pd
import logging
from multiprocessing import Process, Manager


def conditional_cache_v2(func):
    def decorator(*args, **kwargs):
        key = kwargs.get("key", None)
        cached = kwargs.get("cached", None)
        
        if cached is not None and key is not None:
            if key not in decorator.cache.keys():
                decorator.cache[key] = func(*args, **kwargs)

            return decorator.cache[key]
        
        return func(*args, **kwargs)

    decorator.cache = dict()

    return decorator


class DatasetManager:
    def __init__(self, dataset_root,
                 folds: tuple = (1, 2, 3, 4, 5),
                 augments: tuple = (),
                 verbose=1):
        """
        

        """
        self.sr = sr
        self.dataset_root = dataset_root
        self.metadata_path = os.path.join(dataset_root, "meta/Esc50.csv")
        self.audio_path = os.path.join(dataset_root, "esc.hdf")
        
        self.folds = folds
        
        self.augments = augments

        # verbose mode
        self.verbose = verbose
        if self.verbose == 1: self.tqdm_func = tqdm.tqdm
        elif self.verbose == 2: self.tqdm_func = tqdm.tqdm_notebook

        # Store the dataset metadata information
        self.meta = {}
        self._load_metadata()

        # Store the raw audio
        self.audio:= self._hdf_to_dict(self.audio_path, folds)

    def _load_metadata(self):
        metadata_path = os.path.join(self.metadata_root, "UrbanSound8K.csv")

        self.meta = pd.read_csv(metadata_path, sep=",")
        self.meta = self.meta.set_index("filename")
        self.meta["idx"] = list(range(len(self.meta)))

    def _hdf_to_dict(self, hdf_path, folds: list, key: str = "data") -> dict:
        output = dict()

        # open hdf file
        with h5py.File(hdf_path, "r") as hdf:
            for fold in self.tqdm_func(folds):
                hdf_fold = hdf["fold%d" % fold]

                filenames = np.asarray(hdf_fold["filenames"])
                audios = np.asarray(hdf_fold[key])

                fold_dict = dict(zip(filenames[:], audios[:]))
                output["fold%d" % fold] = fold_dict

        logging.info("nb file loaded: %d" % len(output))
        return output

    @conditional_cache_v2
    def extract_feature(self, raw_data, **kwargs):
        """
        extract the feature for the model. Cache behaviour is implemented with the two parameters filename and cached
        :param raw_data: to audio to transform
        :key key: Unique key link to the raw_data, used internally by the cache system
        :key cached: use or not the cache system
        :return: the feature extracted from the raw audio
        """
        feat = librosa.feature.melspectrogram(
            raw_data, self.sr, n_fft=2048, hop_length=512, n_mels=64, fmin=0, fmax=self.sr // 2)
        feat = librosa.power_to_db(feat, ref=np.max)
        
        return feat

if __name__ == '__main__':
    # audio_root = "../dataset/audio"
    # metadata_root = "../dataset/metadata"
    audio_root = os.path.join("E:/", "Corpus", "UrbanSound8K", "audio")
    metadata_root = os.path.join("E:/", "Corpus", "UrbanSound8K", "metadata")
    static_augment_file = os.path.join("E:/", "Corpus", "UrbanSound8K", "audio", "urbansound8k_22050_augmentations.hdf5")
    augment_list = ["I_PSC1"]




