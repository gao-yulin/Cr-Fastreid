# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
import time
import itertools
from collections import OrderedDict
import os
import os.path as osp
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
import faiss

from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist
from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank_cylib import compile_helper

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self._cpu_device = torch.device('cpu')

        self._predictions = []
        self._compile_dependencies()

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        prediction = {
            'feats': outputs.to(self._cpu_device, torch.float32),
            'pids': inputs['targets'].to(self._cpu_device),
            'camids': inputs['camids'].to(self._cpu_device)

        }
        self._predictions.append(prediction)

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        features = []
        pids = []
        camids = []
        for prediction in predictions:
            features.append(prediction['feats'])
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])

        features = torch.cat(features, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids, dim=0).numpy()
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = pids[:self._num_query]
        query_camids = camids[:self._num_query]

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = pids[self._num_query:]
        gallery_camids = camids[self._num_query:]

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        from .rank import evaluate_rank
        if self.cfg.TEST.USE_FAISS_FOR_INDEX is True:
            if self.cfg.TEST.RERANK.ENABLED:
                raise NotImplementedError
            if self.cfg.TEST.METRIC == "euclidean":
                faiss_index = faiss.IndexFlatL2(features.shape[1])
                faiss_index.add(gallery_features.numpy())
                indexes = faiss_index.search(query_features.numpy(), gallery_features.shape[0])[1]
            elif self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, p=2, dim=1)
                gallery_features = F.normalize(gallery_features, p=2, dim=1)
                faiss_index = faiss.IndexFlatIP(features.shape[1])
                faiss_index.add(gallery_features.numpy())
                indexes = faiss_index.search(query_features.numpy(), gallery_features.shape[0])[1]
            else:
                raise NotImplementedError
            cmc, all_AP, all_INP = evaluate_rank(
                indexes, query_pids, gallery_pids, query_camids, gallery_camids, use_distmat=False
            )

        else:
            dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

            if self.cfg.TEST.RERANK.ENABLED:
                logger.info("Test with rerank setting")
                k1 = self.cfg.TEST.RERANK.K1
                k2 = self.cfg.TEST.RERANK.K2
                lambda_value = self.cfg.TEST.RERANK.LAMBDA

                if self.cfg.TEST.METRIC == "cosine":
                    query_features = F.normalize(query_features, dim=1)
                    gallery_features = F.normalize(gallery_features, dim=1)

                rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
                dist = rerank_dist * (1 - lambda_value) + dist * lambda_value
            cmc, all_AP, all_INP = evaluate_rank(
                dist, query_pids, gallery_pids, query_camids, gallery_camids, use_distmat=True
            )

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100
        self._results["metric"] = (mAP + cmc[0]) / 2 * 100

        if self.cfg.TEST.ROC.ENABLED:
            if self.cfg.TEST.USE_FAISS_FOR_INDEX is True:
                raise NotImplementedError
            from .roc import evaluate_roc
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)

    def _compile_dependencies(self):
        # Since we only evaluate results in rank(0), so we just need to compile
        # cython evaluation tool on rank(0)
        if comm.is_main_process():
            try:
                from .rank_cylib.rank_cy import evaluate_cy
            except ImportError:
                start_time = time.time()
                logger.info("> compiling reid evaluation cython tool")

                compile_helper()

                logger.info(
                    ">>> done with reid evaluation cython tool. Compilation time: {:.3f} "
                    "seconds".format(time.time() - start_time))
        comm.synchronize()


def partition_arg_topk(matrix, k):
    a_part = np.argpartition(matrix, k, axis=1)
    column_index = np.arange(matrix.shape[0])[:, None]
    a_sec_argsort_k = np.argsort(matrix[column_index, a_part[:, :k]], axis=1)
    return a_part[:, :k][column_index, a_sec_argsort_k]


class ReidPredictor(ReidEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        super(ReidPredictor, self).__init__(cfg, num_query, output_dir)

    def process(self, inputs, outputs):
        prediction = {
            'feats': outputs.to(self._cpu_device, torch.float32),
            'img_paths': inputs['img_paths']
        }
        self._predictions.append(prediction)

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        features = torch.cat([_['feats'] for _ in predictions], dim=0)
        sample_names = []
        for prediction in predictions:
            sample_names.extend([osp.split(_)[1] for _ in prediction['img_paths']])
        query_names = sample_names[:self._num_query]
        gallery_names = sample_names[self._num_query:]
        query_features = features[:self._num_query]
        gallery_features = features[self._num_query:]

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Inference with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        logger.info('Start computing distances...')
        if self.cfg.TEST.USE_FAISS_FOR_INDEX is True:
            if self.cfg.TEST.RERANK.ENABLED:
                raise NotImplementedError
            if self.cfg.TEST.METRIC == "euclidean":
                faiss_index = faiss.IndexFlatL2(features.shape[1])
                faiss_index.add(gallery_features.numpy())
                indexes = faiss_index.search(query_features.numpy(), self.cfg.TEST.QUERY_RES_NUM)[1]
            elif self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, p=2, dim=1)
                gallery_features = F.normalize(gallery_features, p=2, dim=1)
                faiss_index = faiss.IndexFlatIP(features.shape[1])
                faiss_index.add(gallery_features.numpy())
                indexes = faiss_index.search(query_features.numpy(), self.cfg.TEST.QUERY_RES_NUM)[1]
            else:
                raise NotImplementedError

        else:
            dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

            if self.cfg.TEST.RERANK.ENABLED:
                logger.info("Test with rerank setting")
                k1 = self.cfg.TEST.RERANK.K1
                k2 = self.cfg.TEST.RERANK.K2
                lambda_value = self.cfg.TEST.RERANK.LAMBDA

                if self.cfg.TEST.METRIC == "cosine":
                    query_features = F.normalize(query_features, dim=1)
                    gallery_features = F.normalize(gallery_features, dim=1)

                rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
                dist = rerank_dist * (1 - lambda_value) + dist * lambda_value
            if self.cfg.TEST.QUERY_RES_NUM > 0:
                indexes = partition_arg_topk(dist, self.cfg.TEST.QUERY_RES_NUM)
            else:
                indexes = dist.argsort(axis=1)

        logger.info('Start generating json file...')
        result_dict = {}
        gallery_names_array = np.array(gallery_names)
        for query_idx, query_name in enumerate(query_names):
            result_dict[query_name] = gallery_names_array[indexes[query_idx]].tolist()
        bytes_rate = os.getenv("BYTES_RATE")
        output_path = (self._output_dir or self.cfg.OUTPUT_DIR) + '/{}.json'.format(bytes_rate)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info('Output path :' + output_path)
        with open(output_path, 'w', encoding='UTF8') as f:
            f.write(json.dumps(result_dict, indent=2, sort_keys=False))
        logger.info('Done')

        return OrderedDict()
