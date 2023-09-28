import copy
import json
import logging

from joblib import Parallel, delayed

import torch
import torch.multiprocessing as mp
from learnware.market import easy

from build_market import build_from_preprocessed, upload_to_easy_market
from evaluate_market import evaluate_market_performance
from utils.clerk import Clerk


def process_runner(args):
    easy.logger.setLevel(logging.WARNING)

    learnware_list = build_from_preprocessed(args, regenerate=args.regenerate)
    market = upload_to_easy_market(args, learnware_list)
    performance = evaluate_market_performance(args, market)
    performance["Args"] = args.__dict__

    return json.dumps(performance)

class MultiProcessSearcher:
    def __init__(self, args_list, clerk):
        self.args_list = args_list
        self.results = []
        self.clerk = clerk

    def dispatch(self):
        tasks = []
        with mp.Pool(processes=len(self.args_list)) as pool:
            for args in self.args_list:
                task = pool.apply_async(process_runner, copy.deepcopy((args,)))
                tasks.append(task)

            for task in tasks:
                self.results.append(json.loads(task.get()))

        # 记录搜素情况
        self.clerk.phrase(self.results)
