import torch
import numpy as np
from random import randrange


# randomly select 70% Aas training data, and remaining 30% as test data
def rand_div(cls1_data_subjid, cls1_label, cls0_data_subjid, cls0_label, train_pct, normalized=False):
    cls1_tr_idx = np.sort(np.random.choice(np.arange(0, cls1_data_subjid.shape[0]), size=int(train_pct*cls1_data_subjid.shape[0]), replace=False))
    cls1_data_subjid_tr, cls1_label_tr = cls1_data_subjid[cls1_tr_idx], cls1_label[cls1_tr_idx]
    cls1_te_idx = np.sort(np.setdiff1d(np.arange(0, cls1_data_subjid.shape[0]), cls1_tr_idx))
    cls1_data_subjid_te, cls1_label_te = cls1_data_subjid[cls1_te_idx], cls1_label[cls1_te_idx]

    cls0_tr_idx = np.sort(np.random.choice(np.arange(0, cls0_data_subjid.shape[0]), size=int(train_pct*cls0_data_subjid.shape[0]), replace=False))
    cls0_data_subjid_tr, cls0_label_tr = cls0_data_subjid[cls0_tr_idx], cls0_label[cls0_tr_idx]
    cls0_te_idx = np.sort(np.setdiff1d(np.arange(0, cls0_data_subjid.shape[0]), cls0_tr_idx))
    cls0_data_subjid_te, cls0_label_te = cls0_data_subjid[cls0_te_idx], cls0_label[cls0_te_idx]

    tr_data_subjid = np.concatenate((cls1_data_subjid_tr, cls0_data_subjid_tr), axis=0)
    tr_label = np.concatenate((cls1_label_tr, cls0_label_tr), axis=0)
    te_data_subjid = np.concatenate((cls1_data_subjid_te, cls0_data_subjid_te), axis=0)
    te_label = np.concatenate((cls1_label_te, cls0_label_te), axis=0)

    if normalized:
        # TODO
        print('TODO')
        # tr_data = (tr_data - np.min(tr_data, axis=0)) / (np.max(tr_data, axis=0) - np.min(tr_data, axis=0))
        # te_data = (te_data - np.min(te_data, axis=0)) / (np.max(te_data, axis=0) - np.min(te_data, axis=0))

        # timeseries (each subject) = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / np.std(timeseries, axis=0, keepdims=True)


    return tr_data_subjid, tr_label, te_data_subjid, te_label
