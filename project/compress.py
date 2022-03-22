# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:06:05 2022

@author: Gao Yulin
"""

import os
import glob
import numpy as np
import torch
from project.AE import AutoEncoder


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_feature_file(path: str) -> np.ndarray:
    return np.fromfile(path, dtype='<f4')


def mse(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def compress(bytes_rate):
    """
    compress feature files under dir 'query_feature' and save in 'compressed_query_feature/{bytes_rate}'
    """
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    print("compressing query features with bytes_rate {}".format(bytes_rate))
    query_fea_dir = 'query_feature'
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    # iterate over query features
    query_fea_paths = glob.glob(os.path.join(query_fea_dir, '*.*'))
    assert (len(query_fea_paths) != 0)
    X = []  # list of query features
    fea_paths = []  # list of compressed query feature's file paths
    for query_fea_path in query_fea_paths:
        query_basename = get_file_basename(query_fea_path)  # 00056451
        fea = read_feature_file(query_fea_path)  # (2048, )  float32
        assert fea.ndim == 1 and fea.dtype == np.float32
        X.append(fea)
        compressed_fea_path = os.path.join(compressed_query_fea_dir, query_basename + '.dat')
        fea_paths.append(compressed_fea_path)
    input_feature_size = X[0].size
    print('Feature size is {}'.format(input_feature_size))
    print("Start doing Autoencoder compression...")
    with open('project/AEmodel/AutoEncoder_' + str(bytes_rate) + '_fp16' + '.pkl', 'rb') as f:
        model = AutoEncoder(int(bytes_rate))
        model.load_state_dict(torch.load(f))
        X = np.vstack(X)
        tensor_X = torch.Tensor(np.expand_dims(X, axis=1))

        encoded, decoded = model(tensor_X)
        # print(encoded.size())  # torch.Size([batch_size, 1, 16])
        compressed_X = np.squeeze(encoded.cpu().detach().numpy(), 1)  # (batch_size, 16)
        c = np.squeeze(decoded.cpu().detach().numpy(), 1).astype('float32')  # (batch_size, 2048)

        loss = mse(X, c)
        print("The reconstructed loss is {}".format(loss))
        print("Start writing compressed feature")
        for path, compressed_fea in zip(fea_paths, compressed_X):
            with open(path, 'wb') as f:
                f.write(compressed_fea.astype('<f2').tostring())
        print('Compression Done for bytes_rate' + str(bytes_rate))


if __name__ == '__main__':
    compress('64')
    # compress("../test.zip", '128')
    # compress("../test.zip", '256')
