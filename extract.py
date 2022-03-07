import os
import glob

import numpy as np
from PIL import Image


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    fea.astype('<f4').tofile(path)
    return True


def extract_feature(im_path: str) -> np.ndarray:
    im = Image.open(im_path)
    fea = np.asarray(im)[::4, ::4, 2].reshape(-1).astype('<f4') / 255
    return fea


def extract():
    # img_dir = 'image'
    img_dir = 'C:/Users/yling/Documents/naic/train/train_picture'
    # fea_dir = 'feature'
    fea_dir = 'datasets/NAIC2021Reid/train_feature'
    os.makedirs(fea_dir, exist_ok=True)
    img_paths = glob.glob(os.path.join(img_dir, '*.*'))
    assert(len(img_paths) != 0)
    for im_path in img_paths:
        basename = get_file_basename(im_path)
        fea = extract_feature(im_path)
        write_feature_file(fea, os.path.join(fea_dir, basename + '.dat'))

    print('Extraction Done')


if __name__ == '__main__':
    extract()