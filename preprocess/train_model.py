import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from learnware.specification.rkme import choose_device
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import TensorDataset, DataLoader

matplotlib.rc('pdf', fonttype=42)
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd
import random
import datetime
import math

from preprocess.model import ConvModel
from preprocess.dataloader import ImageDataLoader

# Train Uploaders' models
def uploader_train(train_X, train_y, val_X, val_y, out_classes, epochs=35, batch_size=128, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_feature = train_X.shape[1]
    model = ConvModel(channel=input_feature, n_random_features=out_classes).to(device)
    model.train()

    # Adam optimizer with learning rate 1e-3
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # SGD optimizer with learning rate 1e-2
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    # mean-squared error loss
    criterion = nn.CrossEntropyLoss()
    # Prepare DataLoader
    dataset = TensorDataset(torch.from_numpy(train_X).to(device),
                            torch.from_numpy(train_y).to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Optimizing...
    for epoch in range(epochs):
        running_loss = []
        for i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        if (epoch + 1) % 5 == 0:
            print('Epoch: {}, Average Loss: {:.3f}'.format(epoch+1, np.mean(running_loss)))

    # Train Accuracy and print
    val_acc = user_test(val_X, val_y, model, device=device)
    train_acc = user_test(train_X, train_y, model, device=device)
    print("Train Accuracy: {:.2f}\tVal Accuracy: {:.2f}".format(train_acc, val_acc))
    return model

def user_test(data_X, data_y, model, batch_size=128, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    total, correct = 0, 0
    dataset = TensorDataset(torch.from_numpy(data_X).to(device), torch.from_numpy(data_y).to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for i, (X, y) in enumerate(dataloader):
        out = model(X)
        _, predicted = torch.max(out.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    acc = correct/total * 100
    # print("Accuracy: {:.2f}".format(acc))

    return acc

def models_test(test_X, test_y, model_list, device=None):
    acc_list = []
    for model in model_list:
        acc = user_test(test_X, test_y, model, device=device)
        acc_list.append(acc)
    return acc_list

def train_model(args):
    device = choose_device(args.cuda_idx)
    n_uploaders = 50
    n_users = 50
    n_classes = 10
    dataset = args.data
    data_root = os.path.join("image_models", 'learnware_market_data', "{}_{:d}".format(dataset, args.data_id))
    dataloader = ImageDataLoader(data_root, n_uploaders, train=True)

    model_save_root = os.path.join(data_root, 'models')
    os.makedirs(model_save_root, exist_ok=True)
    model_list = []
    for i, data in enumerate(dataloader):
        print("=" * 40)
        print('Train on uploader {:d}, with size'.format(i), data[0].shape, data[2].shape)
        model = uploader_train(*data, out_classes=n_classes, device=device)
        model_save_path = os.path.join(model_save_root, 'uploader_{:d}.pth'.format(i))
        torch.save(model.state_dict(), model_save_path)
        print("Model saved to '{}'".format(model_save_path))
        model_list.append(model)

    test_dataloader = ImageDataLoader(data_root, n_users, train=False)
    acc_matrix = []
    for i, (test_X, test_y) in enumerate(test_dataloader):
        print('Evaluate on user: %d' % i)
        acc_list = models_test(test_X, test_y, model_list, device=device)
        print('Results: Max accuracy: %.2f, Min accuracy: %.2f, Average accuracy: %.2f'%(np.max(acc_list),
                                                                                        np.min(acc_list),
                                                                                        np.average(acc_list)))
        acc_matrix.append(acc_list)

    np_acc_matrix = np.asarray(acc_matrix)
    # TODO: Check This
    print("Accuracy Totally {:.2f} ({:.2f})".format(np.mean(np_acc_matrix), np.std(np_acc_matrix)))
    eval_results = os.path.join(model_save_root, 'eval_results.txt')
    np.savetxt(eval_results, np_acc_matrix)
    print("Eval Results Saved to '{}'".format(eval_results))


if __name__ == '__main__':
    # main()
    pass