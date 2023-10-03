import json
import os

import numpy as np
import torch
from learnware.specification.rkme import choose_device

from preprocess.dataloader import ImageDataLoader
from preprocess.model import ConvModel


def cal_best_match(args, k=1):
    data_root = os.path.join(args.data_root, "learnware_market_data", args.data)

    with open(os.path.join(data_root, "information.json")) as info_file:
        info = json.load(info_file)

    user_weights_record = info["user_weights_record"]
    uploader_weights_record = info["uploader_weights_record"]

    similarities = [[np.sum(np.minimum(np.asarray(user_weight), np.asarray(uploader_weight)))
                  for uploader_weight in uploader_weights_record]
        for user_weight in user_weights_record
    ]

    if k == 1:
        best_match = [np.argmax(np.asarray(u)) for u in similarities]
    else:
        best_match = [np.argsort(-np.asarray(u))[:k] for u in similarities]

    return best_match

def best_match_performance(args):
    device = choose_device(args.cuda_idx)
    data_root = os.path.join(args.data_root, "learnware_market_data", args.data)

    best_match_by_user = cal_best_match(args, k=args.max_search_num)

    models = []
    for model_file in (os.path.join(data_root, "models", "uploader_{:d}.pth".format(i))
                       for i in range(args.n_uploaders)):
        model = ConvModel(channel=3, n_random_features=10)
        model.load_state_dict(torch.load(model_file))
        model.to(device).eval()

        models.append(model)

    dataloader = ImageDataLoader(data_root, args.n_users, train=False)

    acc = []
    for i, (test_X, test_y) in enumerate(dataloader):
        test_X, test_y = torch.asarray(test_X, device=device),\
            torch.asarray(test_y, device=device)

        predict_y = torch.argmax(torch.sum(torch.stack(
            [models[m](test_X) for m in best_match_by_user[i]],
            dim=-1), dim=-1), dim=-1)

        curr_acc = np.mean((predict_y == test_y).cpu().detach().numpy())
        acc.append(curr_acc)
        print("Accuracy for user {:d} with best match: {:.2f}".format(i, curr_acc))

    print("Accuracy: {:.2f} ({:.2f})".format(np.mean(acc), np.std(acc)))