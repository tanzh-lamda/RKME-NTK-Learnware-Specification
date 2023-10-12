import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from scipy.ndimage.interpolation import rotate as scipyrotate

import numpy as np

DATA_ROOT = r'image_models/data'
def get_fashion_mnist(output_channels = 1, image_size = 28):
    print(image_size)
    ds_train = datasets.FashionMNIST(DATA_ROOT, train = True, download = True, transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()]))
    ds_test = datasets.FashionMNIST(DATA_ROOT, train = False, download = True, transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()]))
    train_size = len(ds_train)
    test_size = len(ds_test)
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    for i in range(train_size):
        X, label = ds_train[i]
        train_X.append(X)
        train_y.append(label)
    
    for i in range(test_size):
        X, label = ds_test[i]
        test_X.append(X)
        test_y.append(label)
        # print(label.dtype)

    X_train = torch.cat(train_X, 0)
    X_test = torch.cat(test_X, 0)
    print(X_train.size(), X_test.size())
    X_train = X_train[:, None, :, :].float()
    X_test = X_test[:, None, :, :].float()
    y_train = torch.tensor(train_y, dtype=int)
    y_test = torch.tensor(test_y, dtype=int)
    if(output_channels > 1):
        X_train = torch.cat([X_train for i in range(output_channels)], 1)
        X_test = torch.cat([X_test for i in range(output_channels)], 1)
        
    X_test = (X_test - torch.mean(X_train, [0,2,3], keepdim = True))/(torch.std(X_train, [0,2,3], keepdim = True))
    X_train = (X_train - torch.mean(X_train, [0,2,3], keepdim = True))/(torch.std(X_train, [0,2,3], keepdim = True))

    return X_train, y_train, X_test, y_test


def get_mnist(output_channels = 1, image_size = 28):
    ds_train = datasets.MNIST(DATA_ROOT, train = True, download = True, transform = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor()]))
    X_train = []
    
    for x,_ in ds_train:
        X_train.append(x)
    X_train = torch.stack(X_train)
    
    y_train = ds_train.targets
    ds_test = datasets.MNIST(DATA_ROOT, train = False, download = True, transform = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor()]))
    
    
    X_test = []
    
    for x,_ in ds_test:
        X_test.append(x)
    X_test = torch.stack(X_test)
    
    y_test = ds_test.targets
    
    if output_channels > 1:
        X_train = torch.cat([X_train for i in range(output_channels)], 1)
        X_test = torch.cat([X_test for i in range(output_channels)], 1)
    
    X_test = (X_test - torch.mean(X_train, [0,2,3], keepdim = True))/(torch.std(X_train, [0,2,3], keepdim = True))
    X_train = (X_train - torch.mean(X_train, [0,2,3], keepdim = True))/(torch.std(X_train, [0,2,3], keepdim = True))

    return X_train, y_train, X_test, y_test

def get_cifar10(output_channels = 1, image_size = 32, z_score=True):
    ds_train = datasets.CIFAR10(DATA_ROOT, train = True, download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([image_size, image_size])]))
    X_train = ds_train.data
    y_train = ds_train.targets
    ds_test = datasets.CIFAR10(DATA_ROOT, train = False, download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([image_size, image_size])]))
    
    X_test = ds_test.data
    y_test = ds_test.targets
    
    X_train = torch.Tensor(np.moveaxis(X_train, 3, 1))
    y_train = torch.Tensor(y_train).long()
    X_test = torch.Tensor(np.moveaxis(X_test, 3, 1))
    y_test = torch.Tensor(y_test).long()
    
    if output_channels == 1:
        X_train = torch.mean(X_train, 1, keepdim = True)
        X_test = torch.mean(X_test, 1, keepdim = True)

    if z_score:
        X_test = (X_test - torch.mean(X_train, [0,2,3], keepdim = True))/(torch.std(X_train, [0,2,3], keepdim = True))
        X_train = (X_train - torch.mean(X_train, [0,2,3], keepdim = True))/(torch.std(X_train, [0,2,3], keepdim = True))
    
    return X_train, y_train, X_test, y_test

def get_svhn(output_channels = 1, image_size = 32):
    ds_train = datasets.SVHN(DATA_ROOT, split='train', download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([image_size, image_size])]))
    X_train = ds_train.data
    y_train = ds_train.labels
    ds_test = datasets.SVHN(DATA_ROOT, split = 'test', download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([image_size, image_size])]))
    
    X_test = ds_test.data
    y_test = ds_test.labels 
    
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train).long()
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test).long()
    
    if output_channels == 1:
        X_train = torch.mean(X_train, 1, keepdim = True)
        X_test = torch.mean(X_test, 1, keepdim = True)
        
        
    X_test = (X_test - torch.mean(X_train, [0,2,3], keepdim = True))/(torch.std(X_train, [0,2,3], keepdim = True))
    X_train = (X_train - torch.mean(X_train, [0,2,3], keepdim = True))/(torch.std(X_train, [0,2,3], keepdim = True))
    
    return X_train, y_train, X_test, y_test


def get_cifar100(output_channels = 3, image_size = 32):
    ds_train = datasets.CIFAR100('./data/', train = True, download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([image_size, image_size])]))
    X_train = ds_train.data
    y_train = ds_train.targets
    ds_test = datasets.CIFAR100('./data/', train = False, download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([image_size, image_size])]))
    
    X_test = ds_test.data
    y_test = ds_test.targets
    
    
    X_train = torch.Tensor(np.moveaxis(X_train, 3, 1))
    y_train = torch.Tensor(y_train).long()
    X_test = torch.Tensor(np.moveaxis(X_test, 3, 1))
    y_test = torch.Tensor(y_test).long()
    
    if output_channels == 1:
        X_train = torch.mean(X_train, 1, keepdim = True)
        X_test = torch.mean(X_test, 1, keepdim = True)
        
        
    X_test = (X_test - torch.mean(X_train, [0,2,3], keepdim = True))/(torch.std(X_train, [0,2,3], keepdim = True))
    X_train = (X_train - torch.mean(X_train, [0,2,3], keepdim = True))/(torch.std(X_train, [0,2,3], keepdim = True))
    
    return X_train, y_train, X_test, y_test


def get_zca_matrix(X, reg_coef = 0.1):
    X_flat = X.reshape(X.shape[0], -1)
    cov = (X_flat.T @ X_flat)/X_flat.shape[0]
    reg_amount = reg_coef * torch.trace(cov) / cov.shape[0]
    u, s, _ = torch.svd(cov.cuda() + reg_amount * torch.eye(cov.shape[0]).cuda())
    inv_sqrt_zca_eigs = s**(-0.5)
    whitening_transform = torch.einsum(
      'ij,j,kj->ik', u, inv_sqrt_zca_eigs, u)
    
    return whitening_transform.cpu()


def get_zca_matrix_inv(X, reg_coef=0.1):
    X_flat = X.reshape(X.shape[0], -1)
    cov = (X_flat.T @ X_flat) / X_flat.shape[0]
    reg_amount = reg_coef * torch.trace(cov) / cov.shape[0]
    u, s, _ = torch.svd(cov.cuda() + reg_amount * torch.eye(cov.shape[0]).cuda())

    sqrt_zca_eigs = torch.diag(s) ** 0.5
    inv_whitening_transform = (u @ sqrt_zca_eigs) @ u.T

    return inv_whitening_transform.cpu()

def layernorm_data(X):
    X_processed = (X - torch.mean(X, [1,2,3], keepdim = True))
    X_processed = X_processed/torch.sqrt(torch.sum(X_processed**2, [1,2,3], keepdim = True))
    
    return X_processed
        
def transform_data(X, whitening_transform):
    if len(whitening_transform.shape) == 2:
        X_flat = X.reshape(X.shape[0], -1)
        X_flat = X_flat @ whitening_transform
        return X_flat.view(*X.shape)
    else:
        X_flat = X.reshape(X.shape[0], -1)
        X_flat = torch.einsum('nd, ndi->ni', X_flat, whitening_transform)
        return X_flat.view(*X.shape)
    
    
def scale_to_zero_one(X):
    mins = torch.min(X.view(X.shape[0], -1), 1)[0].view(-1, 1, 1, 1)
    maxes = torch.max(X.view(X.shape[0], -1), 1)[0].view(-1, 1, 1, 1)
    return (X - mins)/(maxes-mins)



def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images


# def augment(images):
    