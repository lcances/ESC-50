import os
import time

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd
import torch
import tqdm
import weakref
from collections.abc import Callable, Iterable

from ubs8k.datasetManager import DatasetManager, StaticManager
from augmentation_utils.augmentations import SignalAugmentation, SpecAugmentation

import logging

choose_one = lambda x: np.random.choice(x, size=1)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, manager, folds: tuple = (), esc10: bool = False,
                 augments: tuple = (), augment_choser: Callable = choose_one,
                 cached: bool = False):
        super().__init__()

        self.manager = manager
        self.folds = folds
        self.esc10 = esc10
        self.nb_class = 10 if esc10 else 50
        
        # cache management
        self.cached = cached
        self.check_cache()

        # Get only necessary audio
        self.x = weakref.WeakValueDictionary()
        for fold_number in self.folds:
            self.x.update(self.manager.audio["fold%d" % fold_number])

        # Get only necessary metadata
        meta = self.manager.meta
        self.y = meta.loc[meta.fold.isin(self.folds)]

        # varialbe
        self.filenames = list(self.x.keys())
        self.s_idx = []
        self.u_idx = []
        
        # alias for verbose mode
        self.tqdm_func = self.manager.tqdm_func

    def __len__(self):
        nb_file = len(self.filenames)
        return nb_file

    def __getitem__(self, index: int):
        return self._generate_data(index)
    
    def _get_valid_files():
        self.x = weakref.WeakValueDictionary()
        
        for fold_number in self.folds:
            self.x.update(self.manager.audio["fold{}".format(fold_number)])
            
        # If esc10 mode is activate, remove all reference to non esc10 files
        if self.esc10:
            for name in self.x.keys():
                if not self.y.at[name, "esc10"]:
                    del self.x[name]

    def split_s_u(self, s_ratio: float):
        if s_ratio == 1.0:
            return list(range(len(self.y))), []
        
        self.y["idx"] = list(range(len(self.y)))

        for i in range(self.nb_class):
            class_meta = self.y.loc[self.y.target == i]
            nb_sample_s = int(np.ceil(len(class_meta) * s_ratio))
            
            total_idx = class_meta.idx.values
            self.s_idx_ = class_meta.sample(n=nb_sample_s).idx.values
            self.u_idx_ = set(total_idx) - set(self.s_idx_)

            self.s_idx += list(self.s_idx_)
            self.u_idx += list(self.u_idx_)

        return self.s_idx, self.u_idx

    def _generate_data(self, index: int):
        # load the raw_audio
        filename = self.filenames[index]
        raw_audio = self.x[filename]

        # recover ground truth
        y = self.y.at[filename, "target"]
        
        # check if augmentation should be applied
        apply_augmentation = self.augment_S if index in self.s_idx else self.augment_U

        # chose augmentation, if no return an empty list
        augment_fn = self.augment_choser(self.augments) if self.augments else []

        # Apply augmentation, only one that applies on the signal will be executed
        raw_audio, cache_id = self._apply_augmentation(raw_audio, augment_fn, filename, apply_augmentation)

        # extract feature and apply spec augmentation
        feat = self.manager.extract_feature(raw_audio, key=cache_id, cached=self.cached)
        feat, _ = self._apply_augmentation(feat, augment_fn, filename, apply_augmentation)
        y = np.asarray(y)
        
        # call end of generation callbacks
        self.end_of_generation_callback()

        return feat, y
    
    def end_of_generation_callback(self):
        self.reset_augmentation_flags()

    def _pad_and_crop(self, raw_audio):
        LENGTH = DatasetManager.LENGTH
        SR = self.manager.sr

        if len(raw_audio) < LENGTH * SR:
            missing = (LENGTH * SR) - len(raw_audio)
            raw_audio = np.concatenate((raw_audio, [0] * missing))

        if len(raw_audio) > LENGTH * SR:
            raw_audio = raw_audio[:LENGTH * SR]

        return raw_audio
    
    # =================================================================================================================
    #   Augmentation
    # =================================================================================================================
    def _apply_augmentation(self, data, augment_list, filename: str = None, apply: bool = True):
        """
        Choose the proper augmentation function depending on the type. If augmentation_style is static and augType is
        SignalAugmentation, then call the static augmentation function, otherwise call the dynamic augmentation
        function.

        :param data: the data to augment.
        :param augment_fn: The taugmentation function to apply.
        :param filename: The filename of the current file to processe (usefull for static augmentation)
        :return: the augmented data.
        """
        # format augmentation
        if not isinstance(augment_list, Iterable):
            augment_list = [augment_list]
            
        # Apply all the augmentation inside the list
        augmented = data
        cache_id = filename
        
        # If no augmentation should be applied (see self.augment_S and self.augment_U)
        if not apply:
            return augmented, cache_id
        
        for augment_fn in augment_list:
            # If augmentaiton already applied, don't do it again
            if self.applied_augmentation.get(augment_fn, False):
                continue
                
            # dynamic signal augmentation
            if isinstance(augment_fn, SignalAugmentation):
                augmented = self._apply_dynamic_augmentation_helper(augment_fn, data)
                
                # Some function like TimeStretch modifiy the size of the signal.
                augmented = self._pad_and_crop(augmented)
                cache_id = None      # Dynamic augmentation can't be cached.

            # dynamic spec augmentation
            elif isinstance(augment_fn, SpecAugmentation):
                # Can apply spec augmentation only on spectrogram
                if len(data.shape) == 2:
                    augmented = self._apply_dynamic_augmentation_helper(augment_fn, data)
                    cache_id = None     #  SpecAugmentation happen after the cache system
                
            # Static augmentation
            elif isinstance(augment_fn, str):
                augmented, augment_str, flavor = self._apply_static_augmentation_helper(augment_fn, data, filename)

                if augment_str is None or flavor is None:
                    cache_id = filename
                else:
                    cache_id = "{}.{}.{}".format(filename, augment_str, flavor)

            # Unknow type, must be callable and can't be cached
            elif callable(augment_fn):
                augmented = augment_fn(data)
                
                # Just in case
                augmented = self._pad_and_crop(augmented)

                cache_id = None

            # unknown type and not callable ERROR
            else:
                raise TypeError("Augmentation must be callable. %s is not" % augment_fn)
                
        return augmented, cache_id

    def _apply_dynamic_augmentation_helper(self, augment_func, data):
        # Mark the augmentation as processed to avoid double application
        self.applied_augmentation[augment_func] = True
        
        return augment_func(data)

    def _apply_static_augmentation_helper(self, augment_str, data, filename):
        # Mark the augmentation as processed to avoid double application
        self.applied_augmentation[augment_func] = True
        
        apply = np.random.random()

        if apply <= self.static_augmentation.get(augment_str, 0.5):
            number_of_flavor = self.manager.static_augmentation[augment_str][filename].shape[0]
            flavor_to_use = np.random.randint(0, number_of_flavor)

            return self.manager.static_augmentation[augment_str][filename][flavor_to_use], augment_str, flavor_to_use
        return data, None, None
    
    def reset_augmentation_flags(self):
        self.applied_augmentation = dict(zip(self.augments, (False, ) * len(self.augments)))
        return self.applied_augmentation

    def set_static_augment_ratio(self, ratios: dict):
        self.static_augmentation_ratios = ratios


