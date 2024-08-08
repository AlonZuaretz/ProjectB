import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from sklearn.model_selection import KFold
from data_helper import extract_data, create_dataloaders_cov, create_dataloaders_weights, plot_losses
from neural_nets import Stage1Network
from train import train_and_validate

if __name__ == "__main__":

    path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV1_new\allSIR\dataForPython.mat"
    save_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV1_new\allSIR\netV1_results"
    save_flag = True
    Xw, Yw, XR, YR, params = extract_data(path)

    # Training parameters
    batch_size = 256
    epochs_num = 200
    learning_rate = 1e-3

    train_loader_cov, test_loader_cov, idx_train, idx_test = create_dataloaders_cov(XR, YR, batch_size=batch_size)
    # train_loader_weight, test_loader_weights = create_dataloaders_weights(Xw, Yw, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Stage1Network().to(device)
    # model.load_state_dict(torch.load('checkpoint.pth'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.98)  # Reduce lr by 5% each epoch
    train_losses, val_losses = train_and_validate(model, train_loader_cov, test_loader_cov,
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










