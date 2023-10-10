import sys
import logging

import numpy as np

logger = logging.getLogger("ntk-experiment")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('Grid Search.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

def get_custom_logger():
    return logger


class Clerk:

    def __init__(self):
        self.best = []
        self.rkme = []

    def best_performance(self, accuracy):
        self.best.append(accuracy)

    def rkme_performance(self, accuracy):
        self.rkme.append(accuracy)

    def __str__(self):
        best_acc = np.asarray(self.best)
        rkme_acc = np.asarray(self.rkme)

        return "\n".join([
            "Best Accuracy {:.5f}({:.3f})".format(np.mean(best_acc), np.std(best_acc)),
            "RKME Accuracy {:.5f}({:.3f})".format(np.mean(rkme_acc), np.std(rkme_acc)),
            "Pearson {:5.f}".format(np.corrcoef(np.stack([best_acc, rkme_acc])))
        ])