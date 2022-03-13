import os
import glob
import zipfile
import numpy as np
import shutil
from sklearn import decomposition
import pickle

def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_feature_file(path: str) -> np.ndarray:
    return np.fromfile(path, dtype='<f4')


def mse(a, b):
    return np.sqrt(np.sum((a-b)**2))

def compress(bytes_rate):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    query_fea_dir = 'query_feature'
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    query_fea_paths = glob.glob(os.path.join(query_fea_dir, '*.*'))
    assert(len(query_fea_paths) != 0)
    X = []
    fea_paths = []
    for query_fea_path in query_fea_paths:
        query_basename = get_file_basename(query_fea_path)
        fea = read_feature_file(query_fea_path)
        assert fea.ndim == 1 and fea.dtype == np.float32
        X.append(fea)
        compressed_fea_path = os.path.join(compressed_query_fea_dir, query_basename + '.dat')
        fea_paths.append(compressed_fea_path)
    input_feature_size = X[0].size
    print('Feature size is {}'.format(input_feature_size))
    print('Sample feature: {}'.format(X[0]))
    print("Start doing PCA...")
    with open('project/PCAmodel/pca' + str(bytes_rate) + '.pickle', 'rb') as f:
        pca = pickle.load(f)
        compressed_X = pca.transform(X)
        c = pca.inverse_transform(compressed_X)
        c = c.astype('float32')
        loss = mse(X, c)
        # np.savetxt("./reconstructed_data.txt", c, delimiter=',')
        print("The reconstructed loss is {}".format(loss))
        print("Start writing compressed feature")
        for path, compressed_fea in zip(fea_paths, compressed_X):
            with open(path, 'wb') as f:
                f.write(int(input_feature_size).to_bytes(4, byteorder='little', signed=False))
                f.write(compressed_fea.astype('<f2').tostring())
        print('Compression Done for bytes_rate' + str(bytes_rate))


if __name__ == '__main__':
    compress('64')
    # compress("../test.zip", '128')
    # compress("../test.zip", '256')