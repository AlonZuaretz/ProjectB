import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from data_helper import CovarianceDataset
from net import Stage1Network
from train import train_and_validate


def extract_data(path):
    data = scipy.io.loadmat(path)
    Xw = data['Xw']
    Yw = data['Yw']
    XR = data['XR']
    YR = data['YR']
    params = data['params']

    return Xw, Yw, XR, YR, params


def create_dataloaders(XR, YR, batch_size=32, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    # Split data into training and testing
    XR_train, XR_test, YR_train, YR_test = train_test_split(XR, YR, test_size=test_size, random_state=random_state)

    # Create datasets
    train_dataset = CovarianceDataset(XR_train, YR_train)
    test_dataset = CovarianceDataset(XR_test, YR_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":

    path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\generatedDataV1_simple\dataForPython.mat"
    Xw, Yw, XR, YR, params = extract_data(path)

    # Training parameters
    batch_size = 128
    epochs_num = 100
    learning_rate = 1e-3

    train_loader, test_loader = create_dataloaders(XR, YR, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Stage1Network().to(device)
    # model.load_state_dict(torch.load('best_model.pth'))
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    train_and_validate(model, train_loader, test_loader, criterion, optimizer, device, epochs=epochs_num)










