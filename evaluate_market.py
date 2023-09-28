import copy
import logging
import os
from typing import Dict

import numpy as np
import torch
from learnware import specification
from learnware.market import BaseUserInfo
from torch.utils.data import TensorDataset, DataLoader

from build_market import user_semantic
from preprocess.dataloader import ImageDataLoader
from utils.clerk import Clerk, get_custom_logger
from utils.ntk_rkme import RKMEStatSpecification
from utils.reuse import AveragingReuser


def evaluate_market_performance(args, easy_market) -> Dict:
    logger = get_custom_logger()

    data_root = os.path.join(args.data_root, 'learnware_market_data', args.data)
    dataloader = ImageDataLoader(data_root, args.n_users, train=False)
    acc = []
    for i, (test_X, test_y) in enumerate(dataloader):
        if args.spec == "rbf":
            stat_spec = specification.utils.generate_rkme_spec(X=test_X, reduced_set_size=args.K, gamma=0.1, cuda_idx=0)
        elif args.spec == "ntk":
            stat_spec = RKMEStatSpecification(n_models=8, **args.__dict__)
            stat_spec.generate_stat_spec_from_data(test_X, reduce=True, steps=args.ntk_steps, K=args.K)
        else:
            raise NotImplementedError()

        user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMEStatSpecification": stat_spec})

        sorted_score_list, single_learnware_list,\
            mixture_score, mixture_learnware_list = easy_market.search_learnware(user_info, max_search_num=1)
        reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list, mode="vote")
        ensemble_predict_y = np.argmax(reuse_ensemble.predict(user_data=test_X), axis=-1)

        curr_acc = np.mean(ensemble_predict_y == test_y)
        acc.append(curr_acc)
        logger.debug("Accuracy for user {:d} with {} kernel: {:.3f}".format(i, args.spec, curr_acc))

    logger.info("Accuracy {:.3f}({:.3f})".format(np.mean(acc), np.std(acc)))

    return {
        "Accuracy": {
            "Mean": np.mean(acc),
            "Std": np.std(acc)
        }
    }