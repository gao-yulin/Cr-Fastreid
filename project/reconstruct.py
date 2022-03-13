import os
import glob
import numpy as np
from sklearn import decomposition
import pickle



def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    fea.astype('<f4').tofile(path)
    return True


def reconstruct_feature(path: str) -> np.ndarray:
    fea = np.fromfile(path, dtype='<f4')
    fea = np.concatenate(
        [fea, np.zeros(2048 - fea.shape[0], dtype='<f4')], axis=0
    )
    return fea


def reconstruct(bytes_rate):
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(bytes_rate)
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    compressed_query_fea_paths = glob.glob(os.path.join(compressed_query_fea_dir, '*.*'))
    assert(len(compressed_query_fea_paths) != 0)
    names = []
    X = []
    feature_len = 0
    for compressed_query_fea_path in compressed_query_fea_paths:
        query_basename = get_file_basename(compressed_query_fea_path)
        names.append(query_basename)
        with open(compressed_query_fea_path, 'rb') as f:
            feature_len = int.from_bytes(f.read(4), byteorder='little', signed=False)
            fea = np.frombuffer(f.read(), dtype='<f2')
            X.append(fea)
    # Do decompress
    print("Do PCA reconstruct to feature length {}".format(feature_len))
    with open('project/PCAmodel/pca' + bytes_rate + '.pickle', 'rb') as f:
        pca = pickle.load(f)
        c = pca.inverse_transform(X)
        c = c.astype('float32')
    # reconstructed_fea = decompress_feature(compressed_query_fea_path)

    # np.savetxt("./reconstructed_data.txt", c, delimiter=',')
    for path, decompressed_feature in zip(names, c):
        reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, path + '.dat')
        write_feature_file(decompressed_feature, reconstructed_fea_path)

    print('Reconstruction Done' + bytes_rate)

if __name__ == '__main__':
    reconstruct('64')