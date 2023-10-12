import json
from functools import reduce

import preprocess.data_downloader as data_downloader
import torch
import numpy as np
import random
import os

SAVE_ROOT = os.path.join("image_models", "learnware_market_data")
def sample_by_labels(labels: torch.Tensor, weights, total_num):
    weights = np.asarray(weights)

    norm_factor = np.sum(weights)
    last_non_zero = np.argwhere(weights > 0)[-1].item()
    category_nums = [int(w * total_num / norm_factor) for w in weights[:last_non_zero]]
    category_nums += [total_num - sum(category_nums)]
    category_nums += [0] * (weights.shape[0] - last_non_zero - 1)

    selected_cls_indexes = [
        random.sample(list(torch.where(labels == c)[0]), k=n)
            for c, n in enumerate(category_nums)
    ]

    return selected_cls_indexes


def generate_uploader(data_x, data_y, train_size, val_size, weights, n_uploaders=50, data_save_root=None):
    if data_save_root is None:
        raise NotImplementedError("data_save_root can't be None")
    os.makedirs(data_save_root, exist_ok=True)

    weights = np.asarray(weights)
    orders_record, weights_record = [], []
    for i in range(n_uploaders):
        order = list(range(len(weights)))
        random.shuffle(order)

        selected_data_indexes = reduce(lambda x, y: x+y, sample_by_labels(data_y, weights[order], train_size))
        selected_data_indexes = torch.stack(selected_data_indexes)
        selected_X = data_x[selected_data_indexes].numpy()
        selected_y = data_y[selected_data_indexes].numpy()

        X_save_dir = os.path.join(data_save_root, 'uploader_{:d}_train_X.npy'.format(i))
        y_save_dir = os.path.join(data_save_root, 'uploader_{:d}_train_y.npy'.format(i))
        np.save(X_save_dir, selected_X)
        np.save(y_save_dir, selected_y)
        if i == 0:
            print('Saving train set to {} with size'.format(data_save_root), selected_X.shape, selected_y.shape)

        selected_data_indexes = reduce(lambda x, y: x + y, sample_by_labels(data_y, weights[order], val_size))
        selected_data_indexes = torch.stack(selected_data_indexes)
        selected_X = data_x[selected_data_indexes].numpy()
        selected_y = data_y[selected_data_indexes].numpy()

        X_save_dir = os.path.join(data_save_root, 'uploader_{:d}_val_X.npy'.format(i))
        y_save_dir = os.path.join(data_save_root, 'uploader_{:d}_val_y.npy'.format(i))
        np.save(X_save_dir, selected_X)
        np.save(y_save_dir, selected_y)
        if i == 0:
            print('Saving val set to {} with size'.format(data_save_root), selected_X.shape, selected_y.shape)

        orders_record.append(order)
        weights_record.append([int(v) for v in weights[order]])

    return orders_record, weights_record

def generate_user(data_x, data_y, size, weights, n_users=50, data_save_root=None):
    if data_save_root is None:
        raise NotImplementedError("data_save_root can't be None")
    os.makedirs(data_save_root, exist_ok=True)

    weights = np.asarray(weights)
    orders_record, weights_record = [], []
    for i in range(n_users):
        order = list(range(len(weights)))
        random.shuffle(order)
        selected_data_indexes = reduce(lambda x, y: x+y, sample_by_labels(data_y, weights[order], size))
        selected_data_indexes = torch.stack(selected_data_indexes)
        selected_X = data_x[selected_data_indexes].numpy()
        selected_y = data_y[selected_data_indexes].numpy()

        X_save_dir = os.path.join(data_save_root, 'user_{:d}_X.npy'.format(i))
        y_save_dir = os.path.join(data_save_root, 'user_{:d}_y.npy'.format(i))
        np.save(X_save_dir, selected_X)
        np.save(y_save_dir, selected_y)
        if i == 0:
            print('Saving user set to {} with size'.format(data_save_root), selected_X.shape, selected_y.shape)

        orders_record.append(order)
        weights_record.append([int(v) for v in weights[order]])

    return orders_record, weights_record

USER_WEIGHTS = [3, 3, 1, 1, 1, 1, 0, 0, 0, 0]
UPLOADER_WEIGHTS = [4, 4, 1, 1, 0, 0, 0, 0, 0, 0]
def generate(args):
    dataset = args.data

    curr_save_root = os.path.join(SAVE_ROOT, "{}_{:d}".format(dataset, args.data_id))
    if dataset == 'cifar10':
        train_X, train_y, test_X, test_y = data_downloader.get_cifar10(output_channels = 3, image_size = 32)

        whitening_mat = data_downloader.get_zca_matrix(train_X, reg_coef=0.1)
        train_X = data_downloader.transform_data(train_X, whitening_mat)
        test_X = data_downloader.transform_data(test_X, whitening_mat)

    elif dataset == 'fashion':
        train_X, train_y, test_X, test_y = data_downloader.get_fashion_mnist(output_channels = 1, image_size = args.image_size)
    else:
        raise NotImplementedError("No Support for", dataset)

    user_orders_record, user_weights_record = generate_user(test_X, test_y, 3000, USER_WEIGHTS, n_users=args.n_users,
                                                            data_save_root=os.path.join(curr_save_root, 'user'))
    uploader_orders_record, uploader_weights_record = generate_uploader(train_X, train_y, 12500, 2000, UPLOADER_WEIGHTS,
                                                                        n_uploaders=args.n_uploaders,
                                                                        data_save_root=os.path.join(curr_save_root, 'uploader'))

    with open(os.path.join(curr_save_root, 'information.json'), "w") as f:
        json.dump({
            "user_orders_record": user_orders_record,
            "user_weights_record": user_weights_record,
            "uploader_orders_record": uploader_orders_record,
            "uploader_weights_record": uploader_weights_record
        }, f)


if __name__ == '__main__':
    generate('cifar10')