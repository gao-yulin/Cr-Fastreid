#!/usr/bin/env python
# encoding: utf-8
"""
@author:  yulin.gao
@contact: gaoyulin@sjtu.edu.cn
"""
import os
import sys
from project.tools.train_net import *
import json
import numpy as np




def reid(bytes_rate):
    # set the environment variable
    os.environ["BYTES_RATE"] = bytes_rate
    # configure the arguments
    args = default_argument_parser().parse_args(['--config-file', 'project/configs/NAIC2021Reid/sbs_mlp2x_inference.yml'
                                                 , '--infer-only', 'MODEL.DEVICE', 'cuda:0'])
    print("Command Line Args:", args)
    # start inference
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    # reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(bytes_rate)
    # gallery_fea_dir = 'gallery_feature'
    # reid_results_path = 'reid_results/{}.json'.format(bytes_rate)
    # os.makedirs(os.path.dirname(reid_results_path), exist_ok=True)
    #
    # query_names = os.listdir(reconstructed_query_fea_dir)
    # gallery_names = os.listdir(gallery_fea_dir)
    # query_num = len(query_names)
    # gallery_num = len(gallery_names)
    # assert(query_num != 0 and gallery_num != 0)
    #
    # # reconstructed_query_fea_list = []
    # # gallery_fea_list = []
    # # for query_name in query_names:
    # #     reconstructed_query_fea_list.append(
    # #         read_feature_file(os.path.join(reconstructed_query_fea_dir, query_name))
    # #     )
    # # for gallery_name in gallery_names:
    # #     gallery_fea_list.append(
    # #         read_feature_file(os.path.join(gallery_fea_dir, gallery_name))
    # #     )
    #
    # # reconstructed_query_fea_all = np.stack(reconstructed_query_fea_list, axis=0).reshape((query_num, 1, -1))
    # # gallery_fea_all = np.stack(gallery_fea_list, axis=0).reshape((1, gallery_num, -1))
    #
    # top_num = min(100, gallery_num)
    # # dists = np.linalg.norm(reconstructed_query_fea_all - gallery_fea_all, ord=2, axis=2)[:, :top_num]
    # # indexes = np.argsort(dists, axis=1)
    # np.random.seed(0)
    # indexes = np.stack([np.random.choice(top_num, top_num, replace=False) for _ in range(query_num)])
    #
    # result_dict = {}
    # gallery_names_array = np.array(list(map(lambda _: _.rsplit('.', 1)[0] + '.png', gallery_names)))
    # del gallery_names
    # for query_idx, query_name in enumerate(query_names):
    #     query_name = query_name.rsplit('.', 1)[0] + '.png'
    #     result_dict[query_name] = gallery_names_array[indexes[query_idx]].tolist()
    #
    # with open(reid_results_path, 'w', encoding='UTF8') as f:
    #     f.write(json.dumps(result_dict, indent=2, sort_keys=False))

    print('ReID Done')


if __name__ == '__main__':
    reid('64')
