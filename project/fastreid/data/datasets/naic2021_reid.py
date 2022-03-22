# encoding: utf-8
import os
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class NAIC2021ReidTrain(ImageDataset):
    _junk_pids = [-1]
    dataset_dir = 'NAIC2021Reid'
    dataset_url = ''
    dataset_name = "NAIC2021ReidTrain"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(osp.join('project', self.root), self.dataset_dir)
        train_set, query_set, gallery_set = self.gen_sample_sets()
        super(NAIC2021ReidTrain, self).__init__(train_set, query_set, gallery_set, **kwargs)

    def gen_sample_sets(self):
        train_dir = osp.join(self.dataset_dir, 'train_feature')
        query_dir = osp.join(self.dataset_dir, 'train_feature')
        gallery_dir = osp.join(self.dataset_dir, 'train_feature')
        required_files = [
            train_dir,
            query_dir,
            gallery_dir,
        ]
        self.check_before_run(required_files)

        train_set = lambda: self.process_dir(
            train_dir, osp.join(self.dataset_dir, 'sub_train_list.txt'), 0
        )
        query_set = lambda: self.process_dir(
            query_dir, osp.join(self.dataset_dir, 'val_query_list.txt'), 1, False
        )
        gallery_set = lambda: self.process_dir(
            gallery_dir, osp.join(self.dataset_dir, 'val_gallery_list.txt'), 2, False
        )
        return train_set, query_set, gallery_set

    def process_dir(self, dir_path, fn_list_path, pseudo_camid, is_train=True):
        im_names = []
        im_pids = []
        if fn_list_path is None:
            im_names = os.listdir(dir_path)
            im_pids = [-1] * len(im_names)
        else:
            with open(fn_list_path, 'r') as f:
                for line in f:
                    im_name, im_pid = line.strip().split(' ')
                    im_names.append(im_name)
                    im_pids.append(int(im_pid))
        data_list = []
        for im_name, im_pid in zip(im_names, im_pids):
            im_camid = pseudo_camid
            if is_train:
                im_pid = self.dataset_name + "_" + str(im_pid)
                im_camid = self.dataset_name + "_" + str(im_camid)
            data_list.append((osp.join(dir_path, im_name), im_pid, im_camid))
        return data_list


@DATASET_REGISTRY.register()
class NAIC2021ReidTestA(NAIC2021ReidTrain):
    _junk_pids = [-1]
    dataset_dir = 'NAIC2021Reid'
    dataset_url = ''
    dataset_name = "NAIC2021ReidTestA"

    def gen_sample_sets(self):
        query_dir = osp.join(self.dataset_dir, 'query_feature_A')
        gallery_dir = osp.join(self.dataset_dir, 'gallery_feature_A')
        required_files = [
            query_dir,
            gallery_dir,
        ]
        self.check_before_run(required_files)

        query_set = lambda: self.process_dir(
            query_dir, None, 1, False
        )
        gallery_set = lambda: self.process_dir(
            gallery_dir, None, 2, False
        )
        return [], query_set, gallery_set

@DATASET_REGISTRY.register()
class NAIC2022ReidTest(NAIC2021ReidTrain):
    _junk_pids = [-1]
    query_dir = 'reconstructed_query_feature'
    bytes_rate = os.getenv("BYTES_RATE")
    dataset_url = ''
    dataset_name = "NAIC2022ReidTest"

    def gen_sample_sets(self):
        query_dir = osp.join(self.query_dir, self.bytes_rate)
        gallery_dir = 'gallery_feature'
        required_files = [
            query_dir,
            gallery_dir,
        ]
        self.check_before_run(required_files)

        query_set = lambda: self.process_dir(
            query_dir, None, 1, False
        )
        gallery_set = lambda: self.process_dir(
            gallery_dir, None, 2, False
        )
        return [], query_set, gallery_set

@DATASET_REGISTRY.register()
class NAIC2022ReidTest128(NAIC2021ReidTrain):
    _junk_pids = [-1]
    query_dir = 'reconstructed_query_feature'
    dataset_url = ''
    dataset_name = "NAIC2022ReidTest128"

    def gen_sample_sets(self):
        query_dir = osp.join(self.query_dir, '128')
        gallery_dir = 'gallery_feature'
        required_files = [
            query_dir,
            gallery_dir,
        ]
        self.check_before_run(required_files)

        query_set = lambda: self.process_dir(
            query_dir, None, 1, False
        )
        gallery_set = lambda: self.process_dir(
            gallery_dir, None, 2, False
        )
        return [], query_set, gallery_set

@DATASET_REGISTRY.register()
class NAIC2022ReidTest256(NAIC2021ReidTrain):
    _junk_pids = [-1]
    query_dir = 'reconstructed_query_feature'
    dataset_url = ''
    dataset_name = "NAIC2022ReidTest256"

    def gen_sample_sets(self):
        query_dir = osp.join(self.query_dir, '256')
        gallery_dir = 'gallery_feature'
        required_files = [
            query_dir,
            gallery_dir,
        ]
        self.check_before_run(required_files)

        query_set = lambda: self.process_dir(
            query_dir, None, 1, False
        )
        gallery_set = lambda: self.process_dir(
            gallery_dir, None, 2, False
        )
        return [], query_set, gallery_set
