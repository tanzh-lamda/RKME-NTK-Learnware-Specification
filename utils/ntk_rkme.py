import codecs
import copy
import json
import os
from typing import Any, Union

import faiss
import numpy as np
import torch
from learnware.specification.rkme import choose_device, setup_seed, solve_qp, RKMEStatSpecification

from utils.random_feature_model import build_model

MODEL_TYPE = ""

class NTKStatSpecification(RKMEStatSpecification):
    def __init__(self, sigma: float = 0.1, n_models=8, cuda_idx: int = -1, **kwargs):
        """Initializing RKME parameters.

        Parameters
        ----------
        sigma : float

        cuda_idx : int
            A flag indicating whether use CUDA during RKME computation. -1 indicates CUDA not used.
        """
        super().__init__(cuda_idx=cuda_idx)
        self.sigma = sigma
        self.n_models = n_models
        self.random_models = None
        self.kwargs = kwargs

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

        if self.random_models is None:
            model_args = {'input_dim': -1, 'n_channels': -1,
                          'n_random_features': self.kwargs["n_features"],
                          'net_depth': 3, 'random__act': self.kwargs["activation"],
                          'mu': 0, 'sigma': self.sigma}

            model_class = build_model("resnet", **model_args)
            self.random_models = self._generate_models(model_class)

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

    def _generate_models(self, model_class, fixed_seed=None):
        models = []
        for m in range(self.n_models):
            if fixed_seed is not None:
                torch.manual_seed(fixed_seed[m])
            model = model_class()
            models.append(model)

        return models

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

        z_random_features = self._generate_random_feature(Z)
        x_random_features = self._generate_random_feature(X)

        K = self._calc_ntk_from_feature(z_random_features, z_random_features).to(self.device)
        C = self._calc_ntk_from_feature(z_random_features, x_random_features).to(self.device)
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
        # TODO

        self.z = Z

    def _generate_random_feature(self, X, batch_size=300):
        X_features_list = []
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self.device)

        for m in range(self.n_models):
            model = self.random_models[m]
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
        x1_feature = self._generate_random_feature(x1, batch_size=batch_size)
        x2_feature = self._generate_random_feature(x2, batch_size=batch_size)

        return self._calc_ntk_from_feature(x1_feature, x2_feature)

    def _calc_ntk_from_feature(self, x1_feature, x2_feature):
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

        return self._calc_ntk_from_raw(Z, X)

    def _sampling_candidates(self, N: int) -> np.ndarray:
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
        v = torch.sum(self._calc_ntk_from_raw(Z1, Z2) * (beta_1.T @ beta_2))

        return float(v)

    def herding(self, T: int) -> np.ndarray:
        raise NotImplementedError()