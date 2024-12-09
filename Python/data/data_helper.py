import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


def extract_data(path):
    data = scipy.io.loadmat(path)
    Xw = data['Xw']
    Yw = data['Yw']
    XR = data['XR']
    XRd = data['XRd']
    YR = data['YR']
    Ydoa = data['Ydoa']
    params = data['params']

    return Xw, Yw, XR, XRd, YR, Ydoa, params


def create_dataloaders(XR, XRd, YR, Yw, Ydoa, batch_size=1024, test_size=0.2, val_size=0.1, random_state=42,
                       num_workers=0):
    indices = np.arange(XR.shape[0])

    # Split the data into training and testing sets
    XR_train, XR_test, XRd_train, XRd_test, YR_train, YR_test, Yw_train, Yw_test, Ydoa_train, Ydoa_test, idx_train, idx_test = (
        train_test_split(XR, XRd, YR, Yw, Ydoa, indices, test_size=test_size, random_state=random_state))

    # Further split the training data into training and validation sets
    XR_train, XR_val, XRd_train, XRd_val, YR_train, YR_val, Yw_train, Yw_val, Ydoa_train, Ydoa_val, idx_train, idx_val = (
        train_test_split(XR_train, XRd_train, YR_train, Yw_train, Ydoa_train, idx_train,
                         test_size=val_size / (1 - test_size), random_state=random_state))

    # Create datasets
    cov_train_dataset = CovarianceDataset(XR_train, XRd_train, YR_train)
    cov_val_dataset = CovarianceDataset(XR_val, XRd_val, YR_val)
    cov_test_dataset = CovarianceDataset(XR_test, XRd_test, YR_test)

    weights_train_dataset = WeightsDataset(Yw_train)
    weights_val_dataset = WeightsDataset(Yw_val)
    weights_test_dataset = WeightsDataset(Yw_test)

    doa_train_dataset = DoaDataset(Ydoa_train)
    doa_val_dataset = DoaDataset(Ydoa_val)
    doa_test_dataset = DoaDataset(Ydoa_test)

    # Create dataloaders
    cov_train_loader = DataLoader(cov_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    cov_val_loader = DataLoader(cov_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    cov_test_loader = DataLoader(cov_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    weights_train_loader = DataLoader(weights_train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers)
    weights_val_loader = DataLoader(weights_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    weights_test_loader = DataLoader(weights_test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers)

    doa_train_loader = DataLoader(doa_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    doa_val_loader = DataLoader(doa_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    doa_test_loader = DataLoader(doa_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return (cov_train_loader, cov_val_loader, cov_test_loader,
            weights_train_loader, weights_val_loader, weights_test_loader,
            doa_train_loader, doa_val_loader, doa_test_loader,
            idx_train, idx_val, idx_test)



def normalize_data(data):
    if len(data.shape) == 3:
        return data / np.max(np.abs(data), axis=(1, 2), keepdims=True)
    else:
        return data / np.max(np.abs(data), axis=1, keepdims=True)


def hermitian_to_real_imag(data):
    """Convert Hermitian matrices to concatenated real-imaginary matrices, using only upper triangle for imag."""
    real_part = np.real(data)
    imag_part = np.imag(data)
    # Combine real and upper-triangular imaginary parts into a new real matrix
    return np.tril(real_part, k=0) + np.triu(imag_part, k=1)


def real_imag_to_hermitian(matrices):
    real_part = np.tril(matrices, k=0) + np.transpose(np.tril(matrices, k=-1), (0,2,1))
    imag_part = np.triu(matrices, k=1) - np.transpose(np.triu(matrices, k=1), (0,2,1))
    return real_part + 1j * imag_part


class CovarianceDataset(Dataset):
    def __init__(self, XR, XRd, YR):
        # Process and convert data before storing in the dataset
        self.XR = XR
        self.YR = YR
        self.XRd = XRd
        self.normalize_and_process()

    def __len__(self):
        return self.XR.shape[0]

    def __getitem__(self, idx):
        inputs = torch.tensor(self.XRd[idx], dtype=torch.float)
        labels = torch.cat((torch.tensor(self.YR[idx], dtype=torch.float), torch.tensor(self.XR[idx], dtype=torch.float)), dim=0)
        return inputs, labels

    def normalize_and_process(self):
        """Normalize complex Hermitian matrices and convert to real-valued format."""
        XR = normalize_data(self.XR)
        YR = normalize_data(self.YR)
        XRd = normalize_data(self.XRd)
        XR_processed = hermitian_to_real_imag(XR)
        YR_processed = hermitian_to_real_imag(YR)
        XRd_processed = hermitian_to_real_imag(XRd)

        XR_processed = XR_processed.reshape(XR_processed.shape[0], 1, XR_processed.shape[1],
                                           XR_processed.shape[2])
        YR_processed = YR_processed.reshape(YR_processed.shape[0], 1, YR_processed.shape[1],
                                           YR_processed.shape[2])
        XRd_processed = XRd_processed.reshape(XRd_processed.shape[0], 1, XRd_processed.shape[1],
                                              XRd_processed.shape[2])

        self.XR = XR_processed
        self.YR = YR_processed
        self.XRd = XRd_processed


class WeightsDataset(Dataset):
    def __init__(self, Yw):
        # Process and convert data before storing in the dataset
        self.Yw = Yw
        self.normalize_and_process()

    def __len__(self):
        return self.Yw.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.Yw[idx], dtype=torch.float)

    def normalize_and_process(self):
        Yw = normalize_data(self.Yw)
        Yw_real = np.real(Yw)
        Yw_imag = np.imag(Yw)
        Yw_processed = np.concatenate((Yw_real, Yw_imag), axis=1)

        self.Yw = Yw_processed


class DoaDataset(Dataset):
    def __init__(self, Ydoa):
        # Process and convert data before storing in the dataset
        self.Ydoa = Ydoa
        self.normalize_and_process()

    def __len__(self):
        return self.Ydoa.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.Ydoa[idx], dtype=torch.float)

    def normalize_and_process(self):
        """Normalize complex Hermitian matrices and convert to real-valued format."""
        Ydoa =  normalize_data(self.Ydoa)
        Ydoa_real = np.real(Ydoa)
        Ydoa_imag = np.imag(Ydoa)

        Ydoa_processed = np.concatenate((Ydoa_real, Ydoa_imag), axis=1)

        self.Ydoa = Ydoa_processed
