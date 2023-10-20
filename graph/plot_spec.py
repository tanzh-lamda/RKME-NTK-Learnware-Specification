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

def rbf_most_similar(Z, X):
    Z = torch.asarray(Z).double().reshape(Z.shape[0], -1)
    X = torch.asarray(X).double().reshape(X.shape[0], -1)
    # L2-Norm
    similarity = torch.sum(Z ** 2, 1, keepdim=True) - 2 * Z @ X.T + torch.sum(X ** 2, 1, keepdim=True).T
    # similarity = torch_rbf_kernel(Z, X, gamma=0.1)
    return torch.argmin(similarity, dim=1), torch.min(similarity, dim=1)

def plot_comparison_diagram(args, uploader_id, rbf_market: DummyMarket, ntk_market: DummyMarket):
    rbf = list(rbf_market.learnware_list.values())[uploader_id].\
        specification.stat_spec["RKMEStatSpecification"]
    ntk = list(ntk_market.learnware_list.values())[uploader_id].\
        specification.stat_spec["RKMEStatSpecification"]
    if args.data == "cifar10":
        X_train, _, _, _ = data_downloader.get_cifar10(output_channels=3, z_score=False)
        X_train_transformed, _, _, _ = data_downloader.get_cifar10(output_channels=3, z_score=True)
        inv_whitening_mat = data_downloader.get_zca_matrix_inv(
            X_train_transformed, reg_coef=0.1).numpy()
        whitening_mat = data_downloader.get_zca_matrix(
            X_train_transformed, reg_coef=0.1).numpy()
        X_train_transformed = data_downloader.transform_data(X_train_transformed, whitening_mat)
    elif args.data == "fashion":
        X_train, _, _, _ = data_downloader.get_fashion_mnist(
            output_channels = 1, image_size = args.image_size, z_score=False)
        X_train_transformed, _, _, _ = data_downloader.get_cifar10(
            output_channels = 1, image_size = args.image_size, z_score=True)
        inv_whitening_mat = np.identity(X_train.shape[1] * X_train.shape[2] * X_train.shape[3])

    rbf_z, ntk_z = rbf.z.detach().cpu().numpy(), ntk.z.detach().cpu().numpy()
    rbf_top = list(sorted(zip(rbf_z, rbf.beta), key=lambda t: t[1], reverse=True))[:8]
    ntk_top = list(sorted(zip(ntk_z, ntk.beta), key=lambda t: t[1], reverse=True))[:8]

    rbf_images = inverse_to_image(args, rbf_top, X_train, inv_whitening_mat)
    ntk_images = inverse_to_image(args, ntk_top, X_train, inv_whitening_mat)

    fig, axs = plt.subplots(3, 8, figsize=(12, 6))

    for i, im in enumerate(rbf_images):
        if im.shape[0] == 3:
            axs[0][i].imshow(np.transpose(im, [1,2,0]))
        else:
            axs[0][i].imshow(np.transpose(im, [1,2,0]), cmap = 'gray')
        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[0][i].set_xlabel("β={:.3f}".format(rbf_top[i][1]))
    axs[0][0].set_ylabel("RBF")

    if args.data == "cifar10":
        most_similar_idxes, distances = rbf_most_similar(np.stack([z for z, beta in rbf_top]), X_train_transformed)
    else: # fashion
        flat = X_train.reshape(X_train.shape[0], -1)
        most_similar_idxes, distances = rbf_most_similar(
            np.stack(rbf_images), (flat - torch.min(flat, dim=1, keepdim=True).values) /
                                  (torch.max(flat, dim=1, keepdim=True).values - torch.min(flat, dim=1, keepdim=True).values))

    for i, idx in enumerate(most_similar_idxes):
        if X_train[idx].shape[0] == 3:
            axs[1][i].imshow(np.transpose(X_train[idx].cpu().numpy() / 255, [1,2,0]))
        else:
            axs[1][i].imshow(np.transpose(X_train[idx].cpu().numpy() / 255, [1, 2, 0]), cmap='gray')
        axs[1][i].set_xticks([])
        axs[1][i].set_yticks([])
        axs[1][i].set_xlabel("{:.4e}".format(
            np.sum((min_max_norm(X_train[idx].cpu().numpy())-rbf_images[i]) ** 2) ** 0.5
        ))
    axs[1][0].set_ylabel("Original")

    for i, im in enumerate(ntk_images):
        if im.shape[0] == 3:
            axs[2][i].imshow(np.transpose(im, [1,2,0]))
        else:
            axs[2][i].imshow(np.transpose(im, [1, 2, 0]), cmap="gray")
        axs[2][i].set_xticks([])
        axs[2][i].set_yticks([])
        axs[2][i].set_xlabel("β={:.3f}".format(ntk_top[i][1]))
    axs[2][0].set_ylabel("NTK")

    plt.tight_layout(pad=2.0)
    plt.savefig("comparison_diagram.png", dpi=600)
    plt.show()