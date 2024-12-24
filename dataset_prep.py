# -*- coding: UTF-8 -*-
from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import random
from torch import tensor, float32, save, load
import pickle


class Dataset_Train(Dataset):
    def __init__(self, tr_data_subjid, tr_label):
        # tr_data_subjid: numpy array with shape (subjnum, 5), here 5 is a tuple
        # containing 1) subjid, 2) fMRI (230, 116), 3) sMRI trad (4858), 4) sMRI deep brainspace (256, 256, 256), 5) sMRI deep mnispace (181, 217, 181)
        # tr_label: numpy array with shape (subjnum, )

        super().__init__()
        subj_num = tr_data_subjid.shape[0]
        numbers = [int(x) for x in range(subj_num)]

        # (1) subject id
        d1 = zip(numbers, tr_data_subjid[:, 0])
        self.subjid_dict = dict(d1)

        # (2) fMRI data
        d2 = zip(numbers, tr_data_subjid[:, 1])
        self.fMRIseries_dict = dict(d2)
        self.full_subject_list = list(self.fMRIseries_dict.keys())

        # (3) sMRI traditional data
        norm_sMRI_trad_init = tr_data_subjid[:, 2]
        norm_sMRI_trad_lst = []
        for i in range(norm_sMRI_trad_init.shape[0]):
            norm_sMRI_trad_lst.append(norm_sMRI_trad_init[i])
        norm_sMRI_trad_arr = np.array(norm_sMRI_trad_lst)
        norm_sMRI_trad = (norm_sMRI_trad_arr - np.min(norm_sMRI_trad_arr, axis=0)) / (np.max(norm_sMRI_trad_arr, axis=0) - np.min(norm_sMRI_trad_arr, axis=0))
        d3 = zip(numbers, norm_sMRI_trad)
        self.sMRI_trad_dict = dict(d3)

        # (4) sMRI deep brainspace (256, 256, 256)
        d4 = zip(numbers, tr_data_subjid[:, 3])
        self.sMRI_deep_brainspace_dict = dict(d4)

        # (5) sMRI deep mnispace (181, 217, 181)
        d5 = zip(numbers, tr_data_subjid[:, 4])
        self.sMRI_deep_mnispace_dict = dict(d5)

        # (6) label
        d6 = zip(numbers, tr_label)  # tr_label is corresponding label
        self.behavioral_dict = dict(d6)

    def __len__(self):
        return len(self.full_subject_list)

    def __getitem__(self, idx):
        subject = self.full_subject_list[idx]
        label = self.behavioral_dict[int(subject)]

        subjid = self.subjid_dict[int(subject)]
        fMRI_timeseries = self.fMRIseries_dict[subject]
        sMRI_trad_freesurfer = self.sMRI_trad_dict[subject].astype('float64')
        sMRI_deep_brainspace = self.sMRI_deep_brainspace_dict[subject].astype('float64')
        sMRI_deep_mnispace = self.sMRI_deep_mnispace_dict[subject].astype('float64')

        if label == 0.0:
            label = tensor(0)
        elif label == 1.0:
            label = tensor(1)
        else:
            raise

        return {'idx': subject, 'subjid:': subjid, 'fMRI': tensor(fMRI_timeseries, dtype=float32), 'sMRI_trad': tensor(sMRI_trad_freesurfer, dtype=float32), 'sMRI_deep_brainspace': tensor(sMRI_deep_brainspace, dtype=float32), 'sMRI_deep_mnispace': tensor(sMRI_deep_mnispace, dtype=float32), 'label': label}


class Dataset_Test(Dataset):
    def __init__(self, te_data_subjid, te_label):
        # te_data_subjid: numpy array with shape (subjnum, 5), here 5 is a tuple
        # containing 1) subjid, 2) fMRI (230, 116), 3) sMRI trad (4858), 4) sMRI deep brainspace (256, 256, 256), 5) sMRI deep mnispace (181, 217, 181)
        # te_label: numpy array with shape (subjnum, )

        super().__init__()
        subj_num = te_data_subjid.shape[0]
        numbers = [int(x) for x in range(subj_num)]

        # (1) subject id
        d1 = zip(numbers, te_data_subjid[:, 0])
        self.subjid_dict = dict(d1)

        # (2) fMRI data
        d2 = zip(numbers, te_data_subjid[:, 1])
        self.fMRIseries_dict = dict(d2)
        self.full_subject_list = list(self.fMRIseries_dict.keys())

        # (3) sMRI traditional data
        # handcrafted features
        norm_sMRI_trad_init = te_data_subjid[:, 2]
        norm_sMRI_trad_lst = []
        for i in range(norm_sMRI_trad_init.shape[0]):
            norm_sMRI_trad_lst.append(norm_sMRI_trad_init[i])
        norm_sMRI_trad_arr = np.array(norm_sMRI_trad_lst)
        norm_sMRI_trad = (norm_sMRI_trad_arr - np.min(norm_sMRI_trad_arr, axis=0)) / (np.max(norm_sMRI_trad_arr, axis=0) - np.min(norm_sMRI_trad_arr, axis=0))
        d3 = zip(numbers, norm_sMRI_trad)
        self.sMRI_trad_dict = dict(d3)

        # (4) sMRI deep brainspace (256, 256, 256)
        d4 = zip(numbers, te_data_subjid[:, 3])
        self.sMRI_deep_brainspace_dict = dict(d4)

        # (5) sMRI deep mnispace (181, 217, 181)
        d5 = zip(numbers, te_data_subjid[:, 4])
        self.sMRI_deep_mnispace_dict = dict(d5)

        # (6) label
        d6 = zip(numbers, te_label)  # te_label is corresponding label
        self.behavioral_dict = dict(d6)

    def __len__(self):
        return len(self.full_subject_list)

    def __getitem__(self, idx):
        subject = self.full_subject_list[idx]
        label = self.behavioral_dict[int(subject)]

        subjid = self.subjid_dict[int(subject)]
        fMRI_timeseries = self.fMRIseries_dict[subject]
        sMRI_trad_freesurfer = self.sMRI_trad_dict[subject].astype('float64')
        sMRI_deep_brainspace = self.sMRI_deep_brainspace_dict[subject].astype('float64')
        sMRI_deep_mnispace = self.sMRI_deep_mnispace_dict[subject].astype('float64')

        if label == 0.0:
            label = tensor(0)
        elif label == 1.0:
            label = tensor(1)
        else:
            raise

        return {'idx': subject, 'subjid:': subjid, 'fMRI': tensor(fMRI_timeseries, dtype=float32), 'sMRI_trad': tensor(sMRI_trad_freesurfer, dtype=float32), 'sMRI_deep_brainspace': tensor(sMRI_deep_brainspace, dtype=float32), 'sMRI_deep_mnispace': tensor(sMRI_deep_mnispace, dtype=float32), 'label': label}