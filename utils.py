import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset.

    @param dataset Numpy array representing the dataset.
    """
    def __init__(self, dataset:np.ndarray):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index:int):
        return self.dataset[index]


class LabeledDataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset and its associated labels.

    @param dataset Numpy array representing the dataset.
    @param labels One-dimensional array of the same length as dataset with
           non-negative int values.
    """
    def __init__(self, dataset:np.ndarray, labels:np.ndarray):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index:int):
        return self.dataset[index], self.labels[index]

def z_normalize(X:np.ndarray):
    """
    Z-normalizes a sequence X

    `X` is a 1D ndarray with the shape (`L`) where `L` is the length of the input
    Outputs a tuple with
        - (1), a z-normalized ndarray of length (`L`)
        - (2), the mean of X as a scalar
        - (3), the std of X as a scalar
    """
    mean = np.mean(X)
    std = np.std(X)
    if std < 1e-5:
        std = 0
    if std != 0:
        z_normalized = (X - mean) / std
    else:
        z_normalized = X - mean
    return z_normalized, mean, std

def z_normalize_dataset(X:np.ndarray, is_global:bool=True):
    """
    Z-normalizes a multivariate time series dataset X

    `X` is a 3D ndarray with the shape (`N`, `C`, `L`), where `N` is the
    number of samples, `C` is the number of input channels, and `L` is the length of
    the input.

    When `is_global` is set to `True`:
    Outputs a tuple with
        - (1), a 3D ndarray of the normalized data (`N`, `C`, `L`),
        - (2), a 3D ndarray of associated means (`1`, `C`, `1`), and
        - (3), a 3D ndarray of associated standard deviations (`1`, `C`, `1`)

    When `is_global` is set to `False`:
    Outputs a tuple with
        - (1), a 3D ndarray of the normalized data (`N`, `C`, `L`),
        - (2), a 3D ndarray of means by sample by channel (`N`, `C`, `1`), and
        - (3), a 3D ndarray of standard deviations by sample by channel (`N`, `C`, `1`)

    Note: This function assumes that all input samples have the same number of
    channels and the same length.
    """
    if is_global:
        mean = np.mean(X,axis=(0,2), keepdims=True)
        std = np.std(X,axis=(0,2), keepdims=True)
    else:
        mean = np.mean(X, axis=2, keepdims=True)
        std = np.std(X, axis=2, keepdims=True)
    z_normalized = (X - mean) / std
    return z_normalized, mean, std

def z_denormalize_dataset(X_norm:np.ndarray, means:np.ndarray, stds:np.ndarray):
    """
    Z-denormalizes a dataset X_norm based on means and standard deviations

    `X_norm` is a 3D ndarray with the shape (`N`, `C`, `L`), where `N` is the
    number of samples, `C` is the number of input channels, and `L` is the length of
    the input.

    `means` is a 3D ndarray with the shape (`N`, `C`, `1`) or (`1`, `C`, `1`)
    containing the means of the dataset by sample by channel or by channel

    `stds` is a 3D ndarray with the shape (`N`, `C`, `1`) or (`1`, `C`, `1`)
    containing the standard deviations of the dataset by sample by channel or by channel

    When `is_global` is set to `True`:
    Outputs a 3D ndarray of the denormalized data (`N`, `C`, `L`)

    Note: This function assumes that all input samples have the same number of
    channels and the same length.
    """
    return (X_norm * stds) + means

