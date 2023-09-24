import argparse
import logging

import fire
import torch

from build_market import build_from_preprocessed, upload_to_easy_market
from evaluate import evaluate_market_performance

parser = argparse.ArgumentParser(description='NTK-RF Experiments Remake')

parser.add_argument('--cuda_idx', type=int, required=False, default=0,
                    help='ID of device')
# learnware
parser.add_argument('--spec', type=str, required=False, default='gaussian',
                    help='Specification, options: [Gaussian, NTK]')
parser.add_argument('--market_root', type=str, required=False, default='market',
                    help='Path of Market')

# data
parser.add_argument('--data', type=str, required=False, default='cifar10', help='dataset type')
parser.add_argument('--data_root', type=str, required=False, default=r"image_models",
                    help='The path of images and models')
parser.add_argument('--n_uploaders', type=int, required=False, default=50, help='Number of uploaders')
parser.add_argument('--n_users', type=int, required=False, default=50, help='Number of users')

#ntk
parser.add_argument('--model_channel', type=int, required=False,
                    default=32, help='channel of random model')
parser.add_argument('--n_features', type=int, required=False, default=64,
                    help='out features of random model')
parser.add_argument('--activation', type=str, required=False,
                    default='relu', help='activation of random model')

args = parser.parse_args()


if __name__ == "__main__":
    # TODO: 不依赖这个补丁，从模型内部修改类型
    # torch.set_default_dtype(torch.float64)

    logging.basicConfig(level=logging.WARNING)

    learnware_list = build_from_preprocessed(args, regenerate=True)
    market = upload_to_easy_market(args, learnware_list)
    evaluate_market_performance(args, market)
