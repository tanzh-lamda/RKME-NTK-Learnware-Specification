import argparse
import copy
import logging
import os
from functools import partial

from matplotlib import rcParams
from learnware.market import easy

from benchmark import best_match_performance, average_performance_totally
from build_market import build_from_preprocessed, upload_to_easy_market
from evaluate_market import evaluate_market_performance
from diagram.plot_accuracy import plot_accuracy_diagram, load_users
from diagram.plot_spec import load_market, plot_comparison_diagram
from preprocess.split_data import generate
from preprocess.train_model import train_model
from utils import ntk_rkme
from utils.clerk import get_custom_logger, Clerk

parser = argparse.ArgumentParser(description='NTK-RF Experiments Remake')

# AUTO_PARAM = "data_id"
parser.add_argument('--mode', type=str, default="regular")
parser.add_argument('--token', default=None, help='Used for auto.bash')
parser.add_argument('--auto_param', type=str, default=None, help='search param in auto model, None for regular mode')

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
parser.add_argument('--regenerate', type=bool, default=True, help='whether to regenerate specs and learnwares')

parser.add_argument('--data', type=str, default='cifar10', help='dataset type')
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--data_root', type=str, default=r"image_models",
                    help='The path of images and models')
parser.add_argument('--n_uploaders', type=int, default=50, help='Number of uploaders')
parser.add_argument('--n_users', type=int, default=50, help='Number of users')
parser.add_argument('--data_id', type=int, default=0, help='market data id')

#ntk
parser.add_argument('--model', type=str,
                    default="conv", help='The model used to generate random features')
parser.add_argument('--n_models', type=int, default=16,
                    help='# of random models')
parser.add_argument('--n_random_features', type=int, default=4096,
                    help='out features of random model')
parser.add_argument('--net_width', type=int, default=128,
                    help='# of inner channels of random model')
parser.add_argument('--net_depth', type=int, default=3,
                    help='network depth of conv')
parser.add_argument('--activation', type=str,
                    default='relu', help='activation of random model')
parser.add_argument('--ntk_steps', type=int,
                    default=100, help='steps of optimization')
parser.add_argument('--ntk_factor', type=float, # TODO: Why this is not working ?
                    default=1, help='factor of steps of optimization')
parser.add_argument('--sigma', type=float, default=None, help='standard variance of random models')

args = parser.parse_args()

CANDIDATES = {
    "model": ['conv', 'resnet'],
    "ntk_steps": [70, 80, 90, 100, 110, 120, 130],
    # "ntk_factor": [0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4],
    "sigma": [0.003, 0.004, 0.005, 0.006, 0.01, 0.025, 0.05, 0.1],
    "n_random_features": [32, 64, 96, 128, 196, 256],
    "net_width": [32, 64, 96, 128, 160, 196],
    "data_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "net_depth": [3, 3, 4, 4, 5, 5, 6, 6]
}

def _regular_mode(clerk=None):

    if args.resplit:
        _re_split_mode()
    learnware_list = build_from_preprocessed(args, regenerate=args.regenerate)
    market = upload_to_easy_market(args, learnware_list)
    evaluate_market_performance(args, market, clerk=clerk, regenerate=args.regenerate)

    best_match_performance(args, clerk=clerk)
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

def _auto_mode(search_key, clerk=None):
    logger = get_custom_logger()

    available_cuda_idx = [0, 1, 2, 3, 4, 5, 6, 7]

    if search_key is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(available_cuda_idx[args.id % len(available_cuda_idx)])
        if args.id >= len(CANDIDATES[search_key]):
            return
        setattr(args, search_key, CANDIDATES[search_key][args.id])

    logger.info("=" * 45)
    for k, v in args.__dict__.items():
        logger.info("{:<10}:{}".format(k, v))
    logger.info("=" * 45)

    _regular_mode(clerk=clerk)
    print(ntk_rkme.RKMEStatSpecification.INNER_PRODUCT_COUNT)

def _plot_spec_mode():
    rbf_market, ntk_market = load_market(args)
    plot_comparison_diagram(args, 10, rbf_market, ntk_market)

def _plot_accuracy_mode():
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = 'SimHei'

    rbf_market, ntk_market = load_market(args)
    rbf_specs, ntk_specs = load_users(args)
    plot_accuracy_diagram(args, rbf_market, ntk_market, rbf_specs, ntk_specs)

def _average_performance():
    average_performance_totally(args, list(range(8)), list(range(8)))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_idx)
    args.cuda_idx = 0
    easy.logger.setLevel(logging.WARNING)

    performance_clerk = Clerk()
    behaviour_by_mode = {
        "resplit": _re_split_mode,
        "regular": partial(_regular_mode, clerk=performance_clerk),
        "auto": partial(_auto_mode, args.auto_param, clerk=performance_clerk),
        "plot_spec": _plot_spec_mode,
        "plot_accuracy": _plot_accuracy_mode,
        "average_performance": _average_performance,
    }

    if args.mode not in behaviour_by_mode:
        raise NotImplementedError()
    behaviour_by_mode[args.mode]()

    print(performance_clerk)





