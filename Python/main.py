import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from data_helper import CovarianceDataset
from neural_nets import Stage1Network
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


if __name__ == "__main__":

    path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV1\lowSIR\dataForPython.mat"
    save_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV1\lowSIR\netV1_results"
    save_flag = False
    Xw, Yw, XR, YR, params = extract_data(path)

    # Training parameters
    batch_size = 256
    epochs_num = 200
    learning_rate = 1e-3

    train_loader, test_loader = create_dataloaders(XR, YR, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Stage1Network().to(device)
    model.load_state_dict(torch.load('checkpoint.pth'))
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    train_losses, val_losses = train_and_validate(model, train_loader, test_loader,
                                                  criterion, optimizer, device, epochs_num, save_path, save_flag)

    plot_losses(train_losses, val_losses)

    # n_splits = 1
    # # Setup K-fold cross-validation
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # fold = 0
    #
    # accuracies = []
    # for train_idx, test_idx in kf.split(XR):
    #     print(f"Training on fold {fold + 1}/{n_splits}...")
    #     train_loader, test_loader = create_dataloaders(XR, YR, (train_idx, test_idx), batch_size=batch_size)
    #
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model = Stage1Network().to(device)
    #     criterion = nn.MSELoss()
    #     optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    #
    #     # Train and validate the model
    #     test_accuracy = train_and_validate(model, train_loader, test_loader, criterion, optimizer, device, epochs=epochs_num)
    #     accuracies.append(test_accuracy)
    #     fold += 1
    #
    # print(f"Mean Accuracy: {np.mean(accuracies)}")
    # print(f"Standard Deviation of Accuracy: {np.std(accuracies)}")










