import logging
import os

import numpy as np
import torch
from learnware import specification
from learnware.market import BaseUserInfo
from torch.utils.data import TensorDataset, DataLoader

from build_market import user_semantic
from preprocess.dataloader import ImageDataLoader
from utils.ntk_rkme import RKMEStatSpecification
from utils.reuse import AveragingReuser


def user_test(data_X, data_y, model, batch_size=128, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total, correct = 0, 0
    dataset = TensorDataset(torch.from_numpy(data_X).to(device), torch.from_numpy(data_y).to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for i, (X, y) in enumerate(dataloader):
        out = model.predict(X)
        _, predicted = torch.max(out.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    acc = correct/total * 100
    # print("Accuracy: {:.2f}".format(acc))

    return acc

def models_test(test_X, test_y, model_list, device):
    acc_list = []
    for model in model_list:
        acc = user_test(test_X, test_y, model, device=device)
        acc_list.append(acc)
    return acc_list


def evaluate_market_performance(args, easy_market):
    data_root = os.path.join(args.data_root, 'learnware_market_data', args.data)
    dataloader = ImageDataLoader(data_root, args.n_users, train=False)
    acc = []
    for i, (test_X, test_y) in enumerate(dataloader):
        if args.spec == "rbf":
            stat_spec = specification.utils.generate_rkme_spec(X=test_X, reduced_set_size=args.K, gamma=0.1, cuda_idx=0)
        elif args.spec == "ntk":
            stat_spec = RKMEStatSpecification(model_channel=args.model_channel,
                                                n_features=args.n_features,
                                                activation=args.activation,
                                                cuda_idx=args.cuda_idx)
            stat_spec.generate_stat_spec_from_data(test_X, reduce=True, K=args.K)
        else:
            raise NotImplementedError()

        user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMEStatSpecification": stat_spec})

        sorted_score_list, single_learnware_list,\
            mixture_score, mixture_learnware_list = easy_market.search_learnware(user_info, max_search_num=1)
        reuse_ensemble = AveragingReuser(learnware_list=mixture_learnware_list, mode="vote")
        ensemble_predict_y = np.argmax(reuse_ensemble.predict(user_data=test_X), axis=-1)

        curr_acc = np.mean(ensemble_predict_y == test_y)
        acc.append(curr_acc)
        print("Accuracy for user {:d} with {} kernel:".format(i, args.spec), curr_acc)

    print("Accuracy:", np.mean(acc), np.std(acc))