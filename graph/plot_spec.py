import copy
import os

import learnware
import numpy as np
import torch
from learnware.specification.rkme import choose_device, torch_rbf_kernel
from matplotlib import pyplot as plt

from build_market import user_semantic, upload_to_easy_market
from preprocess import data_downloader
from preprocess.data_downloader import get_cifar100
from utils.market import DummyMarket

def load_market(args):
    market_root = args.market_root
    rbf_zip_path_list = [os.path.join(market_root, args.data, "{}_{:d}".format("rbf", args.id),
                                "learnware_{:d}.zip".format(i)) for i in range(args.n_uploaders)]
    ntk_zip_path_list = [os.path.join(market_root, args.data, "{}_{:d}".format("ntk", args.id),
                                "learnware_{:d}.zip".format(i)) for i in range(args.n_uploaders)]

    return upload_to_easy_market(args, rbf_zip_path_list, market_id="plot_rbf"), \
        upload_to_easy_market(args, ntk_zip_path_list, market_id="plot_ntk")

def min_max_norm(X):
    lower, upper = np.min(X), np.max(X)
    return (X - lower) / (upper - lower)

def inverse_to_image(args, images, X_train, inv_whitening_mat):
    if args.data != "cifar10":
        raise NotImplementedError()

    mean, std = torch.mean(X_train, [0, 2, 3], keepdim=True).squeeze(dim=0).cpu().numpy(),\
        torch.std(X_train, [0, 2, 3], keepdim=True).squeeze(dim=0).cpu().numpy()

    origin_images = []
    for i, (image, beta) in enumerate(images):
        image_flat = image.reshape(-1)
        origin_image = image_flat @ inv_whitening_mat
        origin_image = origin_image.reshape(*image.shape)
        origin_image = origin_image * std + mean
        origin_image = min_max_norm(origin_image)

        origin_images.append(origin_image)

    return origin_images

def rbf_most_similar(Z, X, device):
    Z = torch.asarray(Z).double().reshape(Z.shape[0], -1).to(device)
    X = torch.asarray(X).double().reshape(X.shape[0], -1).to(device)
    # Equivalent to rbf
    # similarity = torch.sum(Z**2, 1, keepdim=True) - 2 * Z @ X.T + torch.sum(X**2, 1, keepdim=True).T
    similarity = torch_rbf_kernel(Z, X, gamma=0.1)
    return torch.argmax(similarity, dim=1)

def plot_comparison_diagram(args, uploader_id, rbf_market: DummyMarket, ntk_market: DummyMarket):
    rbf = list(rbf_market.learnware_list.values())[uploader_id].\
        specification.stat_spec["RKMEStatSpecification"]
    ntk = list(ntk_market.learnware_list.values())[uploader_id].\
        specification.stat_spec["RKMEStatSpecification"]
    X_train, _, _, _ = data_downloader.get_cifar10(output_channels=3, z_score=False)
    X_train_transformed, _, _, _ = data_downloader.get_cifar10(output_channels=3, z_score=True)
    inv_whitening_mat = data_downloader.get_zca_matrix_inv(
        X_train_transformed, reg_coef=0.1).numpy()
    whitening_mat = data_downloader.get_zca_matrix(
        X_train_transformed, reg_coef=0.1).numpy()
    X_train_transformed = data_downloader.transform_data(X_train_transformed, whitening_mat)

    device = choose_device(args.cuda_idx)

    rbf_z, ntk_z = rbf.z.detach().cpu().numpy(), ntk.z.detach().cpu().numpy()
    rbf_top = list(sorted(zip(rbf_z, rbf.beta), key=lambda t: t[1], reverse=True))[:8]
    ntk_top = list(sorted(zip(ntk_z, ntk.beta), key=lambda t: t[1], reverse=True))[:8]

    rbf_images = inverse_to_image(args, rbf_top, X_train, inv_whitening_mat)
    ntk_images = inverse_to_image(args, ntk_top, X_train, inv_whitening_mat)

    fig, axs = plt.subplots(3, 8, figsize=(12, 6))

    for i, im in enumerate(rbf_images):
        axs[0][i].imshow(np.transpose(im, [1,2,0]))
        axs[0][i].set_visible(False)
        axs[0][i].set_visible(False)

    most_similar_idxes = rbf_most_similar(np.stack([z for z, beta in rbf_top]), X_train_transformed, device)
    for i, idx in enumerate(most_similar_idxes):
        axs[1][i].imshow(np.transpose(X_train[idx] / 255, [1,2,0]))
        axs[1][i].set_visible(False)
        axs[1][i].set_visible(False)

    for i, im in enumerate(ntk_images):
        axs[2][i].imshow(np.transpose(im, [1,2,0]))
        axs[2][i].set_visible(False)
        axs[2][i].set_visible(False)

    plt.show()