import os
from typing import Dict

import numpy as np
from learnware import specification
from learnware.market import BaseUserInfo

from build_market import user_semantic
from preprocess.dataloader import ImageDataLoader
from utils.clerk import Clerk, get_custom_logger
from utils.ntk_rkme import RKMEStatSpecification
from utils.reuse import AveragingReuser


def evaluate_market_performance(args, market, clerk: Clerk=None) -> Dict:
    logger = get_custom_logger()

    data_root = os.path.join(args.data_root, 'learnware_market_data', "{}_{:d}".format(args.data, args.data_id))
    dataloader = ImageDataLoader(data_root, args.n_users, train=False)
    acc = []
    for i, (test_X, test_y) in enumerate(dataloader):
        if args.spec == "rbf":
            stat_spec = specification.utils.generate_rkme_spec(X=test_X, reduced_set_size=args.K, gamma=0.1, cuda_idx=0)
        elif args.spec == "ntk":
            stat_spec = RKMEStatSpecification(rkme_id=i+args.n_uploaders, **args.__dict__)
            stat_spec.generate_stat_spec_from_data(test_X, reduce=True, steps=args.ntk_steps, K=args.K)
        else:
            raise NotImplementedError()

        user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMEStatSpecification": stat_spec})

        sorted_score_list, single_learnware_list, _, _= market.search_learnware(user_info, max_search_num=args.max_search_num)

        reuse_ensemble = AveragingReuser(learnware_list=single_learnware_list, mode="vote")
        ensemble_predict_y = np.argmax(reuse_ensemble.predict(user_data=test_X), axis=-1)

        curr_acc = np.mean(ensemble_predict_y == test_y)
        acc.append(curr_acc)
        if clerk:
            clerk.rkme_performance(curr_acc)

        logger.debug("Accuracy for user {:d}: {:.3f}; {:.3f} on average up to now.".format(i, curr_acc, np.mean(acc)))

    logger.info("Accuracy {:.3f}({:.3f})".format(np.mean(acc), np.std(acc)))

    return {
        "Accuracy": {
            "Mean": np.mean(acc),
            "Std": np.std(acc)
        }
    }