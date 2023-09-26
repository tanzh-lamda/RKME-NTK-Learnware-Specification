import argparse
import logging
import sys

import fire
import torch
from learnware import get_module_logger
from learnware.market import easy

from benchmark import best_match_performance
from build_market import build_from_preprocessed, upload_to_easy_market
from evaluate_market import evaluate_market_performance
from preprocess.split_data import generate
from preprocess.train_model import train_model
from utils import ntk_rkme

parser = argparse.ArgumentParser(description='NTK-RF Experiments Remake')

parser.add_argument('--cuda_idx', type=int, required=False, default=0,
                    help='ID of device')
parser.add_argument('--resplit', default=False, action=argparse.BooleanOptionalAction,
                    help='Resplit datasets')
parser.add_argument('--retrain', default=False, action=argparse.BooleanOptionalAction,
                    help='Retrain models')
parser.add_argument('--regenerate', default=False, action=argparse.BooleanOptionalAction,
                    help='regenerate learnwares')
parser.add_argument('--no_reduce', default=False, action=argparse.BooleanOptionalAction, help='whether to reduce')

# learnware
parser.add_argument('--id', type=int, required=False, default=0,
                    help='Used for parallel training')
parser.add_argument('--spec', type=str, required=False, default='rbf',
                    help='Specification, options: [rbf, NTK]')
parser.add_argument('--market_root', type=str, required=False, default='market',
                    help='Path of Market')
parser.add_argument('-K', type=int, required=False, default=100,
                    help='number of reduced points')

# data
parser.add_argument('--data', type=str, required=False, default='cifar10', help='dataset type')
parser.add_argument('--data_root', type=str, required=False, default=r"image_models",
                    help='The path of images and models')
parser.add_argument('--n_uploaders', type=int, required=False, default=50, help='Number of uploaders')
parser.add_argument('--n_users', type=int, required=False, default=50, help='Number of users')

#ntk
parser.add_argument('--model', type=str, required=False,
                    default="resnet", help='The model used to generate randome features')
parser.add_argument('--model_channel', type=int, required=False,
                    default=32, help='channel of random model')
parser.add_argument('--n_features', type=int, required=False, default=64,
                    help='out features of random model')
parser.add_argument('--activation', type=str, required=False,
                    default='relu', help='activation of random model')
parser.add_argument('--ntk_steps', type=int, required=False,
                    default=40, help='steps of optimization')
parser.add_argument('--sigma', type=float, required=False,
                    default=0.1, help='standard variance of random models')

args = parser.parse_args()

logger = logging.getLogger("ntk-experiment")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('Grid Search.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

if __name__ == "__main__":
    easy.logger.setLevel(logging.WARNING)

    # 避免误触
    # if args.resplit:
    #     generate('cifar10')
    # if args.resplit or args.retrain:
    #     train_model()

    # best_match_performance(args)
    learnware_list = build_from_preprocessed(args, regenerate=args.regenerate)
    market = upload_to_easy_market(args, learnware_list)
    evaluate_market_performance(args, market)

    logger.debug("一共GENERATE: {:d}".format(ntk_rkme.RKMEStatSpecification.GENERATE_COUNT))

    # TODO: 细致处理一下Logger的记录
    # TODO: Clerk类记录精度，有利于网格搜索
    # logger.info("=" * 20 + "ARGS" + "=" * 20)
    for k, v in args.__dict__.items():
        logger.info("{:<10}:{}".format(k, v))
    logger.info("=" * 45)