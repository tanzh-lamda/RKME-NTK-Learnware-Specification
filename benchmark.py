import copy
import json
import os
from typing import List

import numpy as np
import torch
from learnware.specification.rkme import choose_device

from diagram.plot_accuracy import _evaluate_performance_by_user
from diagram.plot_spec import load_market
from preprocess.dataloader import ImageDataLoader
from preprocess.model import ConvModel
from utils.clerk import Clerk


def cal_best_match(args, k=1):
    data_root = os.path.join(args.data_root, "learnware_market_data",
                             "{}_{:d}".format(args.data, args.data_id))

    with open(os.path.join(data_root, "information.json")) as info_file:
        info = json.load(info_file)

    user_weights_record = info["user_weights_record"][:args.n_users]
    uploader_weights_record = info["uploader_weights_record"][:args.n_uploaders]

    similarities = [[np.sum(np.minimum(np.asarray(user_weight), np.asarray(uploader_weight)))
                  for uploader_weight in uploader_weights_record]
        for user_weight in user_weights_record
    ]

    if k == 1:
        best_match = [[np.argmax(np.asarray(u))] for u in similarities]
    else:
        best_match = [np.argsort(-np.asarray(u))[:k] for u in similarities]

    return best_match

def best_match_performance(args, clerk: Clerk=None):
    device = choose_device(args.cuda_idx)
    data_root = os.path.join(args.data_root, "learnware_market_data",
                             "{}_{:d}".format(args.data, args.data_id))

    best_match_by_user = cal_best_match(args, k=args.max_search_num)
    dataloader = ImageDataLoader(data_root, args.n_users, train=False)
    input_channel = dataloader[0][0].shape[1]

    models = []
    for model_file in (os.path.join(data_root, "models", "uploader_{:d}.pth".format(i))
                       for i in range(args.n_uploaders)):
        model = ConvModel(channel=input_channel, n_random_features=10)
        model.load_state_dict(torch.load(model_file))
        model.to(device).eval()

        models.append(model)

    acc = []
    for i, (test_X, test_y) in enumerate(dataloader):
        test_X, test_y = torch.asarray(test_X, device=device),\
            torch.asarray(test_y, device=device)

        predict_y = torch.argmax(torch.sum(torch.stack(
            [models[m](test_X) for m in best_match_by_user[i]],
            dim=-1), dim=-1), dim=-1)

        curr_acc = np.mean((predict_y == test_y).cpu().detach().numpy())
        acc.append(curr_acc)
        if clerk:
            clerk.best_performance(curr_acc)
        else:
            print("Accuracy for user {:d} with best match: {:.2f}".format(i, curr_acc))

    if clerk is None:
        print("Accuracy: {:.2f} ({:.2f})".format(np.mean(acc), np.std(acc)))

def average_performance_totally(args, ids: List[int], data_ids: List[int]):
    accuracies = []
    for id, data_id in zip(ids, data_ids):
        args_ = copy.deepcopy(args)
        args_.data_id = data_id
        args_.id = id

        rbf_market, ntk_market = load_market(args)
        acc = _evaluate_performance_by_user(args, rbf_market)
        acc = np.asarray(acc)

        accuracies.append(acc)

    accuracy_totally = np.stack(accuracies)
    print("Average Case: {:.5f} {:.5f}".format(np.mean(accuracy_totally), np.std(accuracy_totally)))
    print("{:.5f}".format(np.std(np.mean(accuracy_totally, axis=(1,2)))))
