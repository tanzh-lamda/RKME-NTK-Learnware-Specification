import codecs
import copy
import json
import os
from typing import Any, Union

import faiss
import numpy as np
import torch
from learnware.specification import BaseStatSpecification
from learnware.specification.rkme import choose_device, setup_seed, solve_qp, RKMEStatSpecification


class NTKStatSpecification(RKMEStatSpecification):
    def __init__(self, sigma: float = 0.1, cuda_idx: int = -1):
        """Initializing RKME parameters.

        Parameters
        ----------
        sigma : float

        cuda_idx : int
            A flag indicating whether use CUDA during RKME computation. -1 indicates CUDA not used.
        """
        super().__init__(cuda_idx=cuda_idx)
        self.sigma = sigma

    def generate_stat_spec_from_data(
            self,
            X: np.ndarray,
            K: int = 100,
            step_size: float = 0.1,
            steps: int = 3,
            nonnegative_beta: bool = True,
            reduce: bool = True,
    ):
        """Construct reduced set from raw dataset using iterative optimization.

        Parameters
        ----------
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format.
        K : int
            Size of the construced reduced set.
        step_size : float
            Step size for gradient descent in the iterative optimization.
        steps : int
            Total rounds in the iterative optimization.
        nonnegative_beta : bool, optional
            True if weights for the reduced set are intended to be kept non-negative, by default False.
        reduce : bool, optional
            Whether shrink original data to a smaller set, by default True
        """
        alpha = None
        self.num_points = X.shape[0]
        X_shape = X.shape
        Z_shape = tuple([K] + list(X_shape)[1:])
        X = X.reshape(self.num_points, -1)

        # Check data values
        X[np.isinf(X) | np.isneginf(X) | np.isposinf(X) | np.isneginf(X)] = np.nan
        if np.any(np.isnan(X)):
            for col in range(X.shape[1]):
                is_nan = np.isnan(X[:, col])
                if np.any(is_nan):
                    if np.all(is_nan):
                        raise ValueError(f"All values in column {col} are exceptional, e.g., NaN and Inf.")
                    # Fill np.nan with np.nanmean
                    col_mean = np.nanmean(X[:, col])
                    X[:, col] = np.where(is_nan, col_mean, X[:, col])

        if not reduce:
            self.z = X.reshape(X_shape)
            self.beta = 1 / self.num_points * np.ones(self.num_points)
            self.z = torch.from_numpy(self.z).double().to(self.device)
            self.beta = torch.from_numpy(self.beta).double().to(self.device)
            return

        # Initialize Z by clustering, utiliing faiss to speed up the process.
        self._init_z_by_faiss(X, K)
        self._update_beta(X, nonnegative_beta)

        # Alternating optimize Z and beta
        for i in range(steps):
            self._update_z(alpha, X, step_size)
            self._update_beta(X, nonnegative_beta)

        # Reshape to original dimensions
        self.z = self.z.reshape(Z_shape)

    def _update_beta(self, X: Any, nonnegative_beta: bool = True):
        """Fix Z and update beta using its closed-form solution.

        Parameters
        ----------
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format or torch.tensor format.
        nonnegative_beta : bool, optional
            True if weights for the reduced set are intended to be kept non-negative, by default False.
        """
        Z = self.z
        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self.device).double()

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self.device).double()
        # TODO: 修改成
        K = torch_rbf_kernel(Z, Z, gamma=self.gamma).to(self.device)
        C = torch_rbf_kernel(Z, X, gamma=self.gamma).to(self.device)
        C = torch.sum(C, dim=1) / X.shape[0]

        if nonnegative_beta:
            beta = solve_qp(K, C).to(self.device)
        else:
            beta = torch.linalg.inv(K + torch.eye(K.shape[0]).to(self.device) * 1e-5) @ C

        self.beta = beta

    def _update_z(self, alpha: float, X: Any, step_size: float):
        """Fix beta and update Z using gradient descent.

        Parameters
        ----------
        alpha : int
            Normalization factor.
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format or torch.tensor format.
        step_size : float
            Step size for gradient descent.
        """
        sigma = self.sigma
        Z = self.z
        beta = self.beta

        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self.device).double()

        if not torch.is_tensor(beta):
            beta = torch.from_numpy(beta)
        beta = beta.to(self.device).double()

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self.device).double()

        grad_Z = torch.zeros_like(Z)


        self.z = Z

    def _generate_models(self, model_class, n_models, n_features_per_model, fixed_seed=None):
        models = []
        for m in range(n_models):
            if fixed_seed is not None:
                torch.manual_seed(fixed_seed[m])
            model = model_class()
            models.append(model)

        return models

    def _generate_random_feature(self, X, batch_size=300):
        X_features_list = []
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self.device)

        for m in range(self.n_models):
            model = self.model_list[m]
            model.to(self.device)
            model.eval()
            curr_features_list = []
            for i in range(math.ceil(X.shape[0] / batch_size)):
                out = model(X[batch_size * i: batch_size * (i + 1)])
                # print(out.size())
                curr_features_list.append(out)
            curr_features = torch.cat(curr_features_list, 0)
            X_features_list.append(curr_features)
        X_features = torch.cat(X_features_list, 1)
        X_features = X_features / math.sqrt(self.n_models * self.n_features_per_model)
        return X_features

    def _calc_ntk_from_raw(self, x1, x2, batch_size=128):

        x1_feature = self.generate_random_feature(x1, batch_size=batch_size)
        x2_feature = self.generate_random_feature(x2, batch_size=batch_size)

        # x1_feature = x1_feature/math.sqrt(self.n_models*self.n_features_per_model)
        # x2_feature = x2_feature/math.sqrt(self.n_models*self.n_features_per_model)
        K_12 = x1_feature @ x2_feature.T + 0.01
        return K_12

    def _calc_ntk_from_feature(self, x1_feature, x2_feature):
        # print(type(x1_feature), type(x2_feature))
        # print(x1_feature.size(), x2_feature.size())
        K_12 = x1_feature @ x2_feature.T + 0.01
        return K_12


    def _inner_prod_with_X(self, X: Any) -> float:
        """Compute the inner product between RKME specification and X

        Parameters
        ----------
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format or torch.tensor format.

        Returns
        -------
        float
            The inner product between RKME specification and X
        """
        beta = self.beta.reshape(1, -1).double().to(self.device)
        Z = self.z.double().to(self.device)
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self.device).double()




        return v.detach().cpu().numpy()

    def _sampling_candidates(self, N: int) -> np.ndarray:
        """Generate a large set of candidates as preparation for herding

        Parameters
        ----------
        N : int
            The number of herding candidates.

        Returns
        -------
        np.ndarray
            The herding candidates.
        """
        raise NotImplementedError()

    def inner_prod(self, Phi2: RKMEStatSpecification) -> float:
        """Compute the inner product between two RKME specifications

        Parameters
        ----------
        Phi2 : RKMEStatSpecification
            The other RKME specification.

        Returns
        -------
        float
            The inner product between two RKME specifications.
        """
        beta_1 = self.beta.reshape(1, -1).double().to(self.device)
        beta_2 = Phi2.beta.reshape(1, -1).double().to(self.device)
        Z1 = self.z.double().reshape(self.z.shape[0], -1).to(self.device)
        Z2 = Phi2.z.double().reshape(Phi2.z.shape[0], -1).to(self.device)
        v = torch.sum(torch_rbf_kernel(Z1, Z2, self.gamma) * (beta_1.T @ beta_2))

        return float(v)

    def herding(self, T: int) -> np.ndarray:
        """Iteratively sample examples from an unknown distribution with the help of its RKME specification

        Parameters
        ----------
        T : int
            Total iteration number for sampling.

        Returns
        -------
        np.ndarray
            A collection of examples which approximate the unknown distribution.
        """
        raise NotImplementedError()