import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def normalize_data(data):
    return data / np.max(np.abs(data), axis=1, keepdims=True)


def hermitian_to_real_imag(matrices):
    """Convert Hermitian matrices to concatenated real-imaginary matrices, using only upper triangle for imag."""
    real_part = np.real(matrices)
    imag_part = np.imag(matrices)
    # Combine real and upper-triangular imaginary parts into a new real matrix
    return np.tril(real_part, k=0) + np.triu(imag_part, k=1)


def real_imag_to_hermitian(matrices):
    real_part = np.tril(matrices, k=0) + np.tril(matrices, k=1).T
    imag_part = np.triu(matrices, k=1) - np.triu(matrices, k=1).T
    return real_part + 1j * imag_part




def normalize_and_process(data):
    """Normalize complex Hermitian matrices and convert to real-valued format."""
    data_normalized = normalize_data(data)
    data_processed = hermitian_to_real_imag(data_normalized)
    data_reshaped = data_processed.reshape(data_processed.shape[0], 1, data_processed.shape[1], data_processed.shape[2])
    return data_reshaped


class CovarianceDataset(Dataset):
    def __init__(self, XR, YR):
        # Process and convert data before storing in the dataset
        self.XR = normalize_and_process(XR)
        self.YR = normalize_and_process(YR)

    def __len__(self):
        return self.XR.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.XR[idx], dtype=torch.float), torch.tensor(self.YR[idx], dtype=torch.float)

