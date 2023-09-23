import numpy as np
import os

class ImageDataLoader:
    def __init__(self, data_root, size, train:bool=True):
        self.data_root = data_root
        self.train = train
        self.size = size

    def __getitem__(self, idx):
        if self.train:
            X_train_path = os.path.join(self.data_root, 'uploader', 'uploader_{:d}_train_X.npy'.format(idx))
            y_train_path = os.path.join(self.data_root, 'uploader', 'uploader_{:d}_train_y.npy'.format(idx))
            X_val_path = os.path.join(self.data_root, 'uploader', 'uploader_{:d}_val_X.npy'.format(idx))
            y_val_path = os.path.join(self.data_root, 'uploader', 'uploader_{:d}_val_y.npy'.format(idx))

            if not (os.path.exists(X_train_path) and os.path.exists(y_train_path)):
                raise IndexError("Index Error")
            train_X = np.load(X_train_path)
            train_y = np.load(y_train_path)
            if not (os.path.exists(X_val_path) and os.path.exists(y_val_path)):
                raise IndexError("Index Error")
            val_X = np.load(X_val_path)
            val_y = np.load(y_val_path)

            return train_X, train_y, val_X, val_y
        else:
            X_path = os.path.join(self.data_root, 'user', 'user_{:d}_X.npy'.format(idx))
            y_path = os.path.join(self.data_root, 'user', 'user_{:d}_y.npy'.format(idx))
            if not (os.path.exists(X_path) and os.path.exists(y_path)):
                raise Exception("Index Error")
            X = np.load(X_path)
            y = np.load(y_path)

            return X, y

    def __len__(self):
        return self.size

    def __iter__(self):
        return (self[i] for i in range(len(self)))

