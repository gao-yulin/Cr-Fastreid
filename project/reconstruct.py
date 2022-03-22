# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:15:19 2022

@author: yling
"""

import os
import glob
import numpy as np
import pickle
import torch
from project.AE import AutoEncoder


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    fea.astype('<f4').tofile(path)
    return True


def reconstruct(bytes_rate):
    """
    reconstruct features back to size 2048 given compressed features
    """
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(bytes_rate)
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    compressed_query_fea_paths = glob.glob(os.path.join(compressed_query_fea_dir, '*.*'))
    assert (len(compressed_query_fea_paths) != 0)
    names = []
    X = []

    # read compressed feature
    for compressed_query_fea_path in compressed_query_fea_paths:
        query_basename = get_file_basename(compressed_query_fea_path)
        reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, query_basename + '.dat')
        fea = np.fromfile(compressed_query_fea_path, dtype='<f2')
        assert fea.ndim == 1 and fea.dtype == np.float16
        fea = fea.astype('float32')
        X.append(fea)
        names.append(reconstructed_fea_path)

    with open('project/AEmodel/AutoEncoder_' + str(bytes_rate) + '_fp16' + '.pkl', 'rb') as f:
        Coder = AutoEncoder(int(bytes_rate))
        Coder.load_state_dict(torch.load(f))
        X = np.vstack(X)
        tensor_X = torch.tensor(np.expand_dims(X, axis=1))
        decoded = Coder.decoder(tensor_X)
        reconstructed_fea = np.squeeze(decoded.cpu().detach().numpy().astype('float32'), 1)

    for path, decompressed_feature in zip(names, reconstructed_fea):
        write_feature_file(decompressed_feature, path)

    print('Reconstruction Done' + bytes_rate)


if __name__ == '__main__':
    reconstruct('64')
