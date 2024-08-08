import numpy as np
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
    YR = data['YR']
    params = data['params']

    return Xw, Yw, XR, YR, params


def create_dataloaders_cov(XR, YR, batch_size=32, test_size=0.2, random_state=42):
    # Split data into training and testing
    indices = np.arange(XR.shape[0])
    XR_train, XR_test, YR_train, YR_test, idx_train, idx_test = train_test_split(XR, YR, indices, test_size=test_size, random_state=random_state)

    # Create datasets
    train_dataset = CovarianceDataset(XR_train, YR_train)
    test_dataset = CovarianceDataset(XR_test, YR_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, idx_train, idx_test

def create_dataloaders_weights(Xw, Yw, batch_size=32, test_size=0.2, random_state=42):
    # Split data into training and testing
    indices = np.arange(Xw.shape[0])
    Xw_train, Xw_test, Yw_train, Yw_test, idx_train, idx_test = train_test_split(Xw, Yw, indices, test_size=test_size, random_state=random_state)

    # Create datasets
    train_dataset = WeightsDataset(Xw_train, Yw_train)
    test_dataset = WeightsDataset(Xw_test, Yw_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, idx_train, idx_test

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
    return data / np.max(np.abs(data), axis=(1, 2), keepdims=True)


def hermitian_to_real_imag(matrices):
    """Convert Hermitian matrices to concatenated real-imaginary matrices, using only upper triangle for imag."""
    real_part = np.real(matrices)
    imag_part = np.imag(matrices)
    # Combine real and upper-triangular imaginary parts into a new real matrix
    return np.tril(real_part, k=0) + np.triu(imag_part, k=1)


def real_imag_to_hermitian(matrices):
    real_part = np.tril(matrices, k=0) + np.transpose(np.tril(matrices, k=-1), (0,2,1))
    imag_part = np.triu(matrices, k=1) - np.transpose(np.triu(matrices, k=1), (0,2,1))
    return real_part + 1j * imag_part


class CovarianceDataset(Dataset):
    def __init__(self, XR, YR):
        # Process and convert data before storing in the dataset
        self.XR = XR
        self.YR = YR
        self.normalize_and_process()

    def __len__(self):
        return self.XR.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.XR[idx], dtype=torch.float), torch.tensor(self.YR[idx], dtype=torch.float)

    def normalize_and_process(self):
        """Normalize complex Hermitian matrices and convert to real-valued format."""
        XR = normalize_data(self.XR)
        YR = normalize_data(self.YR)
        XR_processed = hermitian_to_real_imag(XR)
        YR_processed = hermitian_to_real_imag(YR)

        XR_reshaped = XR_processed.reshape(XR_processed.shape[0], 1, XR_processed.shape[1],
                                               XR_processed.shape[2])
        YR_reshaped = YR_processed.reshape(YR_processed.shape[0], 1, YR_processed.shape[1],
                                               YR_processed.shape[2])
        self.XR = XR_reshaped
        self.YR = YR_reshaped



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
        Xw = normalize_data(self.Xw)
        Yw = normalize_data(self.Yw)
        Xw_real = np.real(Xw)
        Yw_real = np.real(Yw)
        Xw_imag = np.imag(Xw)
        Yw_imag = np.imag(Yw)

        Xw_processed = np.concatenate((Xw_real, Xw_imag), axis=1)
        Yw_processed = np.concatenate((Yw_real, Yw_imag), axis=1)


        self.XR = Xw_processed
        self.YR = Yw_processed

