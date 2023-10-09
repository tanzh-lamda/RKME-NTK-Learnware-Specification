import argparse
import copy
import logging
import os

from learnware.market import easy
import torch.multiprocessing as mp

from benchmark import best_match_performance
from build_market import build_from_preprocessed, upload_to_easy_market
from evaluate_market import evaluate_market_performance
from preprocess.split_data import generate
from preprocess.train_model import train_model
from searcher import MultiProcessSearcher
from utils import ntk_rkme
from utils.clerk import get_custom_logger, Clerk

parser = argparse.ArgumentParser(description='NTK-RF Experiments Remake')

parser.add_argument('--mode', type=str, default="regular", required=False)

# train
parser.add_argument('--cuda_idx', type=int, default=0,
                    help='ID of device')
parser.add_argument('--no_reduce', default=False, action=argparse.BooleanOptionalAction, help='whether to reduce')

# learnware
parser.add_argument('--id', type=int, default=0,
                    help='Used for parallel training')
parser.add_argument('--spec', type=str, default='ntk',
                    help='Specification, options: [rbf, NTK]')
parser.add_argument('--market_root', type=str, default='market',
                    help='Path of Market')
parser.add_argument('--max_search_num', type=int, default=3,
                    help='Number of Max Search Learnware to ensemble')
parser.add_argument('-K', type=int, default=50,
                    help='number of reduced points')

# data
parser.add_argument('--resplit', default=False, action=argparse.BooleanOptionalAction,
                    help='Resplit datasets')
parser.add_argument('--data', type=str, default='cifar10', help='dataset type')
parser.add_argument('--data_root', type=str, default=r"image_models",
                    help='The path of images and models')
parser.add_argument('--n_uploaders', type=int, default=50, help='Number of uploaders')
parser.add_argument('--n_users', type=int, default=50, help='Number of users')
parser.add_argument('--data_id', type=int, default=0, help='market data id')

#ntk
parser.add_argument('--model', type=str,
                    default="conv", help='The model used to generate random features')
parser.add_argument('--n_random_features', type=int, default=4096,
                    help='out features of random model')
parser.add_argument('--net_width', type=int, default=128,
                    help='# of inner channels of random model')
parser.add_argument('--net_depth', type=int, default=3,
                    help='network depth of conv')
parser.add_argument('--activation', type=str,
                    default='relu', help='activation of random model')
parser.add_argument('--ntk_steps', type=int,
                    default=40, help='steps of optimization')
parser.add_argument('--sigma', type=float, default=None, help='standard variance of random models')

args = parser.parse_args()

CANDIDATES = {
    "model": ['conv', 'resnet'],
    "ntk_steps": [5, 10, 20, 30, 40, 50, 60, 70],
    "sigma": [0.003, 0.004, 0.005, 0.006, 0.01, 0.025, 0.05, 0.1],
    "n_random_features": [32, 64, 96, 128, 196, 256],
    "net_width": [32, 64, 96, 128, 160, 196],
    "data_id": [0, 1, 2, 3, 4, 5, 6, 7],
    "net_depth": [3, 3, 4, 4, 5, 5, 6, 6]
}

AUTO_PARAM = "data_id"

# setattr
def _grid_search_mode():
    mp.set_start_method("spawn")
    logger = get_custom_logger()
    clerk = Clerk()

    available_cuda_idx = [1, 2, 3, 4, 5, 6, 7]
    available_id = [1, 2, 3, 4, 5, 6, 7]

    for key, vals in CANDIDATES.items():
        args_list = []
        for i, val in enumerate(vals):
            buffer = copy.deepcopy(args)
            setattr(buffer, key, val)
            setattr(buffer, "cuda_idx", available_cuda_idx[i])
            setattr(buffer, "id", available_id[i])
            args_list.append(buffer)

        mps = MultiProcessSearcher(args_list, clerk)
        mps.dispatch()

        best_case = clerk.latest_best_case()
        logger.info("Best {} is {} with Accuracy {:.3f}({:.3f})".format(
            key, best_case["Args"][key], best_case["Accuracy"]["Mean"], best_case["Accuracy"]["Std"]
        ))
    # TODO: 细致处理一下Logger的记录
    # TODO: Clerk类记录精度，有利于网格搜索


def _regular_mode():
    easy.logger.setLevel(logging.WARNING)

    if args.resplit:
        _re_split_mode()

    learnware_list = build_from_preprocessed(args, regenerate=True)
    market = upload_to_easy_market(args, learnware_list)
    evaluate_market_performance(args, market)

    logger = get_custom_logger()

    logger.info("=" * 45)
    for k, v in args.__dict__.items():
        logger.info("{:<10}:{}".format(k, v))
    logger.info("=" * 45)

def _re_split_mode():
    setattr(args, "data_id", CANDIDATES["data_id"][args.id])
    generate(args)
    train_model(args)
    best_match_performance(args)

def _auto_mode(search_key):
    logger = get_custom_logger()

    available_cuda_idx = [1, 2, 3, 4, 5, 6, 7, 0]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(available_cuda_idx[args.id % len(available_cuda_idx)])
    if args.id >= len(CANDIDATES[search_key]):
        return
    setattr(args, search_key, CANDIDATES[search_key][args.id])

    if args.resplit:
        _re_split_mode()

    logger.info("=" * 45)
    for k, v in args.__dict__.items():
        logger.info("{:<10}:{}".format(k, v))
    logger.info("=" * 45)
    _regular_mode()



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_idx)
    args.cuda_idx = 0

    if args.mode == "resplit":
        _re_split_mode()
    elif args.mode == "grid":
        _grid_search_mode()
    elif args.mode == "regular":
        _regular_mode()
    elif args.mode == "auto":
        _auto_mode(AUTO_PARAM)
    else:
        raise NotImplementedError()




