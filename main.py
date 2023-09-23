import argparse
import logging

import fire

from build_market import build_from_preprocessed, upload_to_easy_market
from evaluate import evaluate_market_performance

parser = argparse.ArgumentParser(description='NTK-RF Experiments Remake')

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

args = parser.parse_args()


if __name__ == "__main__":
    # fire.Fire(ExperimentsWorkflow)
    logging.basicConfig(level=logging.WARNING)

    learnware_list = build_from_preprocessed(args, regenerate=True)
    market = upload_to_easy_market(args, learnware_list)
    evaluate_market_performance(args, market)
