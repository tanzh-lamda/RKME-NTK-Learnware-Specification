import copy
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from learnware import specification
from learnware.market import BaseUserInfo
from learnware.specification.rkme import choose_device
from tqdm import tqdm

from build_market import user_semantic
from evaluate_market import evaluate_market_performance
from preprocess.dataloader import ImageDataLoader
from utils import ntk_rkme
from utils.market import DummyMarket


def load_users(args):
    market_root = args.market_root
    rbf_spec_path_list = [os.path.join(market_root, args.data, "{}_{:d}".format("rbf", args.id),
                            "user_{:d}".format(i), "spec.json") for i in range(args.n_uploaders)]
    ntk_spec_path_list = [os.path.join(market_root, args.data, "{}_{:d}".format("ntk", args.id),
                            "user_{:d}".format(i), "spec.json") for i in range(args.n_uploaders)]

    rbf_specs, ntk_specs = [], []

    for path in rbf_spec_path_list:
        spec = specification.RKMEStatSpecification(gamma=0.1, cuda_idx=args.cuda_idx)
        spec.load(path)
        rbf_specs.append(spec)

    for i, path in enumerate(ntk_spec_path_list):
        spec = ntk_rkme.RKMEStatSpecification(rkme_id=i+args.n_uploaders, **args.__dict__)
        spec.load(path)
        ntk_specs.append(spec)

    return rbf_specs, ntk_specs

def _evaluate_performance_by_user(args, market: DummyMarket):
    models = [market.learnware_list[key] for key in market.learnware_list]

    device = choose_device(args.cuda_idx)
    data_root = os.path.join(args.data_root, 'learnware_market_data', "{}_{:d}".format(args.data, args.data_id))
    dataloader = ImageDataLoader(data_root, args.n_users, train=False)

    accuracies = []
    for i, (test_X, test_y) in tqdm(enumerate(dataloader), total=args.n_users):
        test_X, test_y = torch.asarray(test_X, device=device), \
            torch.asarray(test_y, device=device)

        acc = []
        for model in models:
            predict_y = torch.argmax(model.predict(test_X), dim=-1)
            curr_acc = np.mean((predict_y == test_y).cpu().detach().numpy())
            acc.append(curr_acc)

        accuracies.append(acc)

    return accuracies

def _evaluate_performance_with_spec(args, market: DummyMarket, spec: str):
    args = copy.deepcopy(args)
    args.spec = spec
    return evaluate_market_performance(args, market, regenerate=False)["Accuracy"]["All"]


def plot_accuracy_diagram(args, rbf_market: DummyMarket, ntk_market: DummyMarket,
                          rbf_specs: List[specification.RKMEStatSpecification],
                          ntk_specs: List[ntk_rkme.RKMEStatSpecification]):
    accuracy_by_user = _evaluate_performance_by_user(args, rbf_market)
    accuracy_by_user = np.asarray(accuracy_by_user)
    max_accuracy_by_user = np.max(accuracy_by_user, axis=-1)
    min_accuracy_by_user = np.min(accuracy_by_user, axis=-1)
    mean_accuracy_by_user = np.mean(accuracy_by_user, axis=-1)

    print("Average Case: {:.5f} {:.5f}".format(np.mean(accuracy_by_user), np.std(accuracy_by_user)))

    rbf_accuracy_by_user = np.asarray(_evaluate_performance_with_spec(args, rbf_market, "rbf"))
    ntk_accuracy_by_user = np.asarray(_evaluate_performance_with_spec(args, ntk_market, "ntk"))

    plt.yticks(np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
               ["0%", "20%", "40%", "60%", "80%", "100%"])
    plt.xlabel("目标任务")
    plt.ylabel("投票复用准确率")
    plt.ylim(0.0, 1.0)
    x_user = range(1, args.n_users + 1)
    plt.fill_between(x_user, max_accuracy_by_user, min_accuracy_by_user,
                     alpha=0.35, step='mid', color="lightblue", label="全部学件")

    plt.step(x_user, mean_accuracy_by_user, "--", where='mid', color="tab:green", label="平均表现")
    plt.step(x_user, rbf_accuracy_by_user, "--", where='mid', color="tab:blue", label="RBF表现")
    plt.step(x_user, ntk_accuracy_by_user, "-", where='mid', color="tab:red", label="NTK表现")

    plt.fill_between(x_user, ntk_accuracy_by_user, rbf_accuracy_by_user,
                     where=ntk_accuracy_by_user>rbf_accuracy_by_user,
                     alpha=0.35, step='mid', color="tab:red")

    plt.legend()
    plt.savefig("accuracy_diagram.png", dpi=600)
    plt.show()