import codecs
import copy
import json
import os
from math import sqrt
from typing import Any, Union

import faiss
import learnware
import numpy as np
import torch
import torch_optimizer
from learnware.specification import BaseStatSpecification
from learnware.specification.rkme import solve_qp, choose_device, setup_seed
from torch.utils.data import TensorDataset, DataLoader

from utils.random_feature_model import build_model

class RKMEStatSpecification(BaseStatSpecification):
    # Lazy Mode
    # 好像要单例模式了, 只能有一个网络实例,不然算的似乎不行
    ntk_calculator = None

    def __init__(self, sigma: float = 0.1,
                 n_models=8, cuda_idx: int = -1,
                 **kwargs):
        self.z = None
        self.beta = None
        self.num_points = 0
        self.cuda_idx = cuda_idx
        torch.cuda.empty_cache()
        self.device = choose_device(cuda_idx=cuda_idx)

        self.sigma = sigma
        self.n_models = n_models
        self.kwargs = kwargs
        self.n_features = kwargs["n_features"]
        self.random_models = None

        setup_seed(0)

    def generate_stat_spec_from_data(
        self,
        X: np.ndarray,
        K: int = 100,
        step_size: float = 0.1,
        steps: int = 3,
        nonnegative_beta: bool = True,
        reduce: bool = True,
    ):
        alpha = None
        self.num_points = X.shape[0]
        X_shape = X.shape
        Z_shape = tuple([K] + list(X_shape)[1:])
        X = X.reshape(self.num_points, -1)

        self.random_models = self._generate_models(self.kwargs, self.sigma,
                                                   self.n_models, self.device)

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
            self.z = torch.from_numpy(self.z).to(self.device)
            self.beta = torch.from_numpy(self.beta).to(self.device)
            return

        # Initialize Z by clustering, utiliing faiss to speed up the process.
        self._init_z_by_faiss(X, K)

        # Reshape to original dimensions
        X = X.reshape(X_shape)
        self.z = self.z.reshape(Z_shape).float()
        with torch.no_grad():
            x_features = self._generate_random_feature(X)
        self._update_beta(x_features, nonnegative_beta)
        optimizer = torch_optimizer.AdaBelief([{"params": [self.z]}],
                                              lr=step_size, eps=1e-16)
        # Alternating optimize Z and beta
        for i in range(steps):
            self._update_z(alpha, x_features, optimizer)
            self._update_beta(x_features, nonnegative_beta)


    def _init_z_by_faiss(self, X: Union[np.ndarray, torch.tensor], K: int):
        """Intialize Z by faiss clustering.

        Parameters
        ----------
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format or torch.tensor format.
        K : int
            Size of the construced reduced set.
        """
        X = X.astype("float32")
        numDim = X.shape[1]
        kmeans = faiss.Kmeans(numDim, K, niter=100, verbose=False)
        kmeans.train(X)
        center = torch.from_numpy(kmeans.centroids)
        self.z = center

    @classmethod
    def _generate_models(cls, kwargs, sigma, n_models, device, fixed_seed=None):
        if cls.ntk_calculator is not None:
            return cls.ntk_calculator

        # 由于一些历史原因，这个input_dim其实对应了数据的channel
        # 而这里的channel，指的其实是模型的宽度
        model_args = {'input_dim': 3, 'n_channels': kwargs["model_channel"],
                      'n_random_features': kwargs["n_features"],
                      'net_depth': 3, 'net_act': kwargs["activation"],
                      'mu': 0, 'sigma': sigma}
        model_class = build_model("resnet", **model_args)

        models = []
        for m in range(n_models):
            if fixed_seed is not None:
                torch.manual_seed(fixed_seed[m])
            model = model_class()
            models.append(model.to(device))

        cls.ntk_calculator = models

        return models

    @torch.no_grad()
    def _update_beta(self, x_features: Any, nonnegative_beta: bool = True):
        Z = self.z
        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self.device)

        if not torch.is_tensor(x_features):
            x_features = torch.from_numpy(x_features)
        x_features = x_features.to(self.device)

        z_features = self._generate_random_feature(Z)

        K = self._calc_ntk_from_feature(z_features, z_features).to(self.device)
        C = self._calc_ntk_from_feature(z_features, x_features).to(self.device)
        C = torch.sum(C, dim=1) / x_features.shape[0]

        if nonnegative_beta:
            beta = solve_qp(K.double(), C.double()).to(self.device)
        else:
            beta = torch.linalg.inv(K + torch.eye(K.shape[0]).to(self.device) * 1e-5) @ C

        self.beta = beta

    def _update_z(self, alpha: float, x_features: Any, optimizer):
        Z = self.z
        beta = self.beta

        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self.device).float()

        if not torch.is_tensor(beta):
            beta = torch.from_numpy(beta)
        beta = beta.to(self.device)

        if not torch.is_tensor(x_features):
            x_features = torch.from_numpy(x_features)
        x_features = x_features.to(self.device).float()

        with torch.no_grad():
            beta = beta.unsqueeze(0)
        # print('Update Z')
        for i in range(3):
            z_features = self._generate_random_feature(Z)
            K_z = self._calc_ntk_from_feature(z_features, z_features)
            K_zx = self._calc_ntk_from_feature(x_features, z_features)
            term_1 = torch.sum(K_z * (beta.T @ beta))
            term_2 = torch.sum(K_zx * beta / self.num_points)
            loss = term_1 - 2 * term_2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def _generate_random_feature(self, data_X, batch_size=300) -> torch.Tensor:
        X_features_list = []
        if not torch.is_tensor(data_X):
            data_X = torch.from_numpy(data_X)
        data_X = data_X.to(self.device)

        dataset = TensorDataset(data_X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for m, model in enumerate(self.random_models):
            model.eval()
            curr_features_list = []
            for i, (X,) in enumerate(dataloader):
                out = model(X)
                curr_features_list.append(out)
            curr_features = torch.cat(curr_features_list, 0)
            X_features_list.append(curr_features)
        X_features = torch.cat(X_features_list, 1)
        X_features = X_features / sqrt(self.n_models * self.n_features)

        return X_features

    def _inner_prod_with_X(self, X: Any) -> float:
        beta = self.beta.reshape(1, -1).to(self.device)
        Z = self.z.to(self.device)

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self.device)

        v = self._calc_ntk_from_raw(Z, X) * beta.T
        v = torch.sum(v, dim=0)

        return v.detach().cpu().numpy()

    def inner_prod(self, Phi2) -> float:
        beta_1 = self.beta.reshape(1, -1).to(self.device)
        beta_2 = Phi2.beta.reshape(1, -1).to(self.device)
        Z1 = self.z.to(self.device)
        Z2 = Phi2.z.to(self.device)
        v = torch.sum(self._calc_ntk_from_raw(Z1, Z2) * (beta_1.T @ beta_2))

        return float(v)

    def dist(self, Phi2, omit_term1: bool = False) -> float:
        if omit_term1:
            term1 = 0
        else:
            term1 = self.inner_prod(self)
        term2 = self.inner_prod(Phi2)
        term3 = Phi2.inner_prod(Phi2)

        return float(term1 - 2 * term2 + term3)

    def _calc_ntk_from_raw(self, x1, x2, batch_size=128):
        x1_feature = self._generate_random_feature(x1, batch_size=batch_size)
        x2_feature = self._generate_random_feature(x2, batch_size=batch_size)

        return self._calc_ntk_from_feature(x1_feature, x2_feature)

    def _calc_ntk_from_feature(self, x1_feature, x2_feature):
        K_12 = x1_feature @ x2_feature.T + 0.01
        return K_12

    def herding(self, T: int) -> np.ndarray:
        raise NotImplementedError()

    def _sampling_candidates(self, N: int) -> np.ndarray:
        raise NotImplementedError()

    def save(self, filepath: str):
        """Save the computed RKME specification to a specified path in JSON format.

        Parameters
        ----------
        filepath : str
            The specified saving path.
        """
        save_path = filepath
        rkme_to_save = copy.deepcopy(self.__dict__)
        if torch.is_tensor(rkme_to_save["z"]):
            rkme_to_save["z"] = rkme_to_save["z"].detach().cpu().numpy()
        rkme_to_save["z"] = rkme_to_save["z"].tolist()
        if torch.is_tensor(rkme_to_save["beta"]):
            rkme_to_save["beta"] = rkme_to_save["beta"].detach().cpu().numpy()
        rkme_to_save["beta"] = rkme_to_save["beta"].tolist()
        rkme_to_save["device"] = "gpu" if rkme_to_save["cuda_idx"] != -1 else "cpu"
        rkme_to_save["random_models"] = "<dynamically generated>"
        json.dump(
            rkme_to_save,
            codecs.open(save_path, "w", encoding="utf-8"),
            separators=(",", ":"),
        )

    def load(self, filepath: str) -> bool:
        """Load a RKME specification file in JSON format from the specified path.

        Parameters
        ----------
        filepath : str
            The specified loading path.

        Returns
        -------
        bool
            True if the RKME is loaded successfully.
        """
        # Load JSON file:
        load_path = filepath
        if os.path.exists(load_path):
            with codecs.open(load_path, "r", encoding="utf-8") as fin:
                obj_text = fin.read()
            rkme_load = json.loads(obj_text)
            rkme_load["device"] = choose_device(rkme_load["cuda_idx"])
            rkme_load["z"] = torch.from_numpy(np.array(rkme_load["z"])).float()
            rkme_load["beta"] = torch.from_numpy(np.array(rkme_load["beta"]))
            rkme_load["random_models"] = self._generate_models(rkme_load["kwargs"],
                                                               rkme_load["sigma"],
                                                               rkme_load["n_models"],
                                                               rkme_load["device"])

            for d in self.__dir__():
                if d in rkme_load.keys():
                    setattr(self, d, rkme_load[d])
            return True
        else:
            return False

    def get_beta(self) -> np.ndarray:
        return self.beta.detach().cpu().numpy()

    def get_z(self) -> np.ndarray:
        return self.z.detach().cpu().numpy()