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


def create_dataloaders_cov(XR, YR, XRd, pre_method, batch_size=32, test_size=0.2, random_state=42, num_workers=0):
    # Split data into training and testing
    indices = np.arange(XR.shape[0])
    XR_train, XR_test, YR_train, YR_test, idx_train, idx_test =\
        train_test_split(XR, YR, indices, test_size=test_size, random_state=random_state)

    XRd_train, XRd_test = XRd[idx_train], XRd[idx_test]

    # Create datasets
    train_dataset = CovarianceDataset(XR_train, YR_train, XRd_train, pre_method)
    test_dataset = CovarianceDataset(XR_test, YR_test, XRd_test, pre_method)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, idx_train, idx_test


def create_dataloaders_weights(Xw, Yw, batch_size=32, test_size=0.2, random_state=42, num_workers=0):
    # Split data into training and testing
    indices = np.arange(Xw.shape[0])
    Xw_train, Xw_test, Yw_train, Yw_test, idx_train, idx_test = train_test_split(Xw, Yw, indices, test_size=test_size, random_state=random_state)

    # Create datasets
    train_dataset = WeightsDataset(Xw_train, Yw_train)
    test_dataset = WeightsDataset(Xw_test, Yw_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, idx_train, idx_test


def create_dataloaders_doa(Ydoa, batch_size=32, test_size=0.2, random_state=42, num_workers=0):
    # Split data into training and testing
    Ydoa_train, Ydoa_test = train_test_split(Ydoa, test_size=test_size, random_state=random_state)

    # Create datasets
    train_dataset = DoaDataset(Ydoa_train)
    test_dataset = DoaDataset(Ydoa_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def plot_losses(train_losses, val_losses):
    # Set up the figure and axes
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')

    # Adding title and labels
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Add legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # Display the plot
    plt.show()


def normalize_data(data):
    if len(data.shape) == 3:
        return data / np.max(np.abs(data), axis=(1, 2), keepdims=True)
    else:
        return data / np.max(np.abs(data), axis=1, keepdims=True)


def abs_phase_split(data):
    abs_part = np.abs(data)
    phase_part = np.angle(data)
    return np.stack([abs_part, phase_part], axis=1)


def abs_phase_rejoin(data):
    abs_part = np.squeeze(data[:, 0, :, :])
    phase_part = np.squeeze(data[:, 1, :, :])
    return abs_part * np.exp(1j * phase_part)


def abs_phase_split_hermitian(data):
    abs_part = np.abs(data) / np.max(np.abs(data), axis=(1, 2), keepdims=True)
    phase_part = np.angle(data) / np.pi
    return np.tril(abs_part, k=0) + np.triu(phase_part, k=1)


def abs_phase_rejoin_hermitian(data):
    abs_part = np.tril(data, k=0) + np.transpose(np.tril(data, k=-1), (0,2,1))
    phase_part = np.triu(data, k=1) - np.transpose(np.triu(data, k=1), (0,2,1))
    return abs_part * np.exp(1j * np.pi * phase_part)


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
    def __init__(self, XR, YR, XRd, pre_method):
        # Process and convert data before storing in the dataset
        self.XR = XR
        self.YR = YR
        self.XRd = XRd
        self.normalize_and_process(pre_method)

    def __len__(self):
        return self.XR.shape[0]

    def __getitem__(self, idx):
        inputs = torch.tensor(self.XR[idx], dtype=torch.float)
        labels = torch.cat((torch.tensor(self.YR[idx], dtype=torch.float), torch.tensor(self.XRd[idx], dtype=torch.float)), dim=0)
        return inputs, labels


    def normalize_and_process(self, pre_method):
        """Normalize complex Hermitian matrices and convert to real-valued format."""
        if pre_method == 1:
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

        elif pre_method == 2:
            XR = abs_phase_split(self.XR)
            YR = abs_phase_split(self.YR)
            XR[:, 0, :, :] = normalize_data(XR[:, 0, :, :])
            YR[:, 0, :, :] = normalize_data(YR[:, 0, :, :])
            XR_processed = XR
            YR_processed = YR

        else:
            XR_processed = abs_phase_split_hermitian(self.XR)
            YR_processed = abs_phase_split_hermitian(self.YR)
            # abs_phase_split_hermitian includes normalization
            XR_processed = XR_processed.reshape(XR_processed.shape[0], 1, XR_processed.shape[1],
                                               XR_processed.shape[2])
            YR_processed = YR_processed.reshape(YR_processed.shape[0], 1, YR_processed.shape[1],
                                               YR_processed.shape[2])

        self.XR = XR_processed
        self.YR = YR_processed
        self.XRd = XRd_processed


class WeightsDataset(Dataset):
    def __init__(self, Xw, Yw):
        # Process and convert data before storing in the dataset
        self.Xw = Xw
        self.Yw = Yw
        self.normalize_and_process()

    def __len__(self):
        return self.Xw.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.Xw[idx], dtype=torch.float), torch.tensor(self.Yw[idx], dtype=torch.float)

    def normalize_and_process(self):
        """Normalize complex Hermitian matrices and convert to real-valued format."""
        Xw = self.Xw
        Yw = self.Yw
        Xw_real = np.real(Xw)
        Yw_real = np.real(Yw)
        Xw_imag = np.imag(Xw)
        Yw_imag = np.imag(Yw)

        Xw_processed = np.concatenate((Xw_real, Xw_imag), axis=1)
        Yw_processed = np.concatenate((Yw_real, Yw_imag), axis=1)

        self.Xw = Xw_processed
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
        Ydoa = self.Ydoa
        Ydoa_real = np.real(Ydoa)
        Ydoa_imag = np.imag(Ydoa)

        Ydoa_processed = np.concatenate((Ydoa_real, Ydoa_imag), axis=1)

        self.Ydoa = Ydoa_processed
