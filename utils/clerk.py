import sys
import logging

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

def get_custom_logger():
    return logger


class Clerk:

    def __init__(self):
        self.phrases = []

    def phrase(self, performances):
        self.phrases.append(
            performances
        )

    def latest_best_case(self):
        if len(self.phrases) == 0:
            raise KeyError("No Phrase at now")

        return max(self.phrases[-1], key=lambda p: p["Accuracy"]["Mean"])