from functools import reduce

import get_data
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
    # if data_save_root is None:
    #     return
    # os.makedirs(data_save_root, exist_ok=True)
    # for i in range(n_uploaders):
    #     random_class_num = random.randint(6,10)
    #     cls_indexes = list(range(10))
    #     random.shuffle(cls_indexes)
    #     selected_cls_indexes = cls_indexes[:random_class_num]
    #     rest_cls_indexes = cls_indexes[random_class_num:]
    #     selected_data_indexes = []
    #     for cls in selected_cls_indexes:
    #         data_indexes = list(torch.where(data_y == cls)[0])
    #         # print(type(data_indexes))
    #         random.shuffle(data_indexes)
    #         data_num = random.randint(800, 2000)
    #         selected_indexes = data_indexes[:data_num]
    #         selected_data_indexes = selected_data_indexes + selected_indexes
    #     for cls in rest_cls_indexes:
    #         flag = random.randint(0,1)
    #         if flag == 0:
    #             continue
    #         data_indexes = list(torch.where(data_y == cls)[0])
    #         random.shuffle(data_indexes)
    #         data_num = random.randint(20, 80)
    #         selected_indexes = data_indexes[:data_num]
    #         selected_data_indexes = selected_data_indexes + selected_indexes
    #     # print('Total Index:', len(selected_data_indexes))
    #     selected_X = data_x[selected_data_indexes].numpy()
    #     selected_y = data_y[selected_data_indexes].numpy()
    #     print(selected_X.dtype, selected_y.dtype)
    #     print(selected_X.shape, selected_y.shape)
    #     X_save_dir = os.path.join(data_save_root, 'uploader_{:d}_X.npy'.format(i))
    #     y_save_dir = os.path.join(data_save_root, 'uploader_{:d}_y.npy'.format(i))
    #     np.save(X_save_dir, selected_X)
    #     np.save(y_save_dir, selected_y)
    #     print('Saving to {}'.format(X_save_dir))
    if data_save_root is None:
        raise NotImplementedError("data_save_root can't be None")
    os.makedirs(data_save_root, exist_ok=True)
    for i in range(n_uploaders):
        random.shuffle(weights)

        selected_data_indexes = reduce(lambda x, y: x+y, sample_by_labels(data_y, weights, train_size))
        selected_data_indexes = torch.stack(selected_data_indexes)
        selected_X = data_x[selected_data_indexes].numpy()
        selected_y = data_y[selected_data_indexes].numpy()

        X_save_dir = os.path.join(data_save_root, 'uploader_{:d}_train_X.npy'.format(i))
        y_save_dir = os.path.join(data_save_root, 'uploader_{:d}_train_y.npy'.format(i))
        np.save(X_save_dir, selected_X)
        np.save(y_save_dir, selected_y)
        if i == 0:
            print('Saving train set to {} with size'.format(data_save_root), selected_X.shape, selected_y.shape)

        selected_data_indexes = reduce(lambda x, y: x + y, sample_by_labels(data_y, weights, val_size))
        selected_data_indexes = torch.stack(selected_data_indexes)
        selected_X = data_x[selected_data_indexes].numpy()
        selected_y = data_y[selected_data_indexes].numpy()

        X_save_dir = os.path.join(data_save_root, 'uploader_{:d}_val_X.npy'.format(i))
        y_save_dir = os.path.join(data_save_root, 'uploader_{:d}_val_y.npy'.format(i))
        np.save(X_save_dir, selected_X)
        np.save(y_save_dir, selected_y)
        if i == 0:
            print('Saving val set to {} with size'.format(data_save_root), selected_X.shape, selected_y.shape)

def generate_user(data_x, data_y, size, weights, n_users=50, data_save_root=None):
    if data_save_root is None:
        raise NotImplementedError("data_save_root can't be None")
    os.makedirs(data_save_root, exist_ok=True)
    for i in range(n_users):
        random.shuffle(weights)
        selected_data_indexes = reduce(lambda x, y: x+y, sample_by_labels(data_y, weights, size))
        selected_data_indexes = torch.stack(selected_data_indexes)
        selected_X = data_x[selected_data_indexes].numpy()
        selected_y = data_y[selected_data_indexes].numpy()

        X_save_dir = os.path.join(data_save_root, 'user_{:d}_X.npy'.format(i))
        y_save_dir = os.path.join(data_save_root, 'user_{:d}_y.npy'.format(i))
        np.save(X_save_dir, selected_X)
        np.save(y_save_dir, selected_y)
        if i == 0:
            print('Saving user set to {} with size'.format(data_save_root), selected_X.shape, selected_y.shape)


USER_WEIGHTS = [3, 3, 1, 1, 1, 1, 0, 0, 0, 0]
UPLOADER_WEIGHTS = [4, 4, 1, 1, 0, 0, 0, 0, 0, 0]
def generate(dataset='cifar10'):
    curr_save_root = os.path.join(SAVE_ROOT, dataset)
    if dataset == 'cifar10':
        train_X, train_y, test_X, test_y = get_data.get_cifar10(output_channels = 3, image_size = 32)
        # print(train_X.dtype, train_y.dtype)
        # print(train_X.size(), train_y, test_X.size(), test_y)
    elif dataset == 'fashion':
        train_X, train_y, test_X, test_y = get_data.get_fashion_mnist(output_channels = 1, image_size = 32)
        print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)
        print(train_y[:10])
    else:
        raise NotImplementedError("No Support for", dataset)

    generate_user(test_X, test_y, 3000, USER_WEIGHTS, data_save_root=os.path.join(curr_save_root, 'user'))
    generate_uploader(train_X, train_y, 12500, 2000, UPLOADER_WEIGHTS, data_save_root=os.path.join(curr_save_root, 'uploader'))


if __name__ == '__main__':
    generate('cifar10')