from typing import List

import numpy as np
import torch
from learnware.learnware import BaseReuser, Learnware
from scipy.special import softmax


class AveragingReuser(BaseReuser):
    """Baseline Multiple Learnware Reuser uing Ensemble Method"""

    def __init__(self, learnware_list: List[Learnware], mode="mean"):
        """The initialization method for ensemble reuser

        Parameters
        ----------
        learnware_list : List[Learnware]
            The learnware list
        """
        super(AveragingReuser, self).__init__(learnware_list)
        self.mode = mode

    def predict(self, user_data: np.ndarray) -> np.ndarray:
        """Give prediction for user data using baseline ensemble method

        Parameters
        ----------
        user_data : np.ndarray
            Raw user data.

        Returns
        -------
        np.ndarray
            Prediction given by ensemble method
        """
        mean_pred_y = None

        for idx in range(len(self.learnware_list)):
            pred_y = self.learnware_list[idx].predict(user_data)
            if isinstance(pred_y, torch.Tensor):
                pred_y = pred_y.detach().cpu().numpy()
            # elif isinstance(pred_y, tf.Tensor):
            #     pred_y = pred_y.numpy()

            if not isinstance(pred_y, np.ndarray):
                raise TypeError(f"Model output must be np.ndarray or torch.Tensor")

            if self.mode == "mean":
                if mean_pred_y is None:
                    mean_pred_y = pred_y
                else:
                    mean_pred_y += pred_y
            elif self.mode == "vote":
                # TODO: 修改learnware包的代码
                softmax_pred = softmax(pred_y, axis=-1)
                # softmax_pred = pred_y
                if mean_pred_y is None:
                    mean_pred_y = softmax_pred
                else:
                    mean_pred_y += softmax_pred

        mean_pred_y /= len(self.learnware_list)

        return mean_pred_y