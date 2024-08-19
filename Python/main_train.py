import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from data_helper import extract_data, create_dataloaders_cov, create_dataloaders_weights, plot_losses
from neural_nets import Stage1Network, Stage2Network
from train import train_and_validate_cov, train_and_validate_weights

if __name__ == "__main__":
    path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV2\dataForPython_train.mat"
    save_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV2\netV3_results"
    save_flag = True
    Xw, Yw, XR, YR, params = extract_data(path)

    # Global parameters:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###### Stage 1 training: #######
    batch_size = 1024
    # epochs_num = 100
    # learning_rate = 5e-4
    pre_method = 1

    train_loader_cov, test_loader_cov, idx_train, idx_test = (
        create_dataloaders_cov(XR, YR, pre_method, batch_size=batch_size))
    stage1_model = Stage1Network().to(device)
    stage1_model.load_state_dict(torch.load('checkpoint_stage1_finetuned.pth'))
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(stage1_model.parameters(), lr=learning_rate)
    # scheduler = ExponentialLR(optimizer, gamma=0.95)
    # train_losses, val_losses = train_and_validate_cov(stage1_model, train_loader_cov, test_loader_cov,
    #                                                   criterion, optimizer, device, scheduler,
    #                                                   epochs_num, save_path, save_flag)
    #
    # plot_losses(train_losses, val_losses)

    ######## Stage 2 training: #######
    batch_size = 1024
    epochs_num = 50
    learning_rate = 5e-4

    train_loader_weights, test_loader_weights, _, _ = create_dataloaders_weights(Xw, Yw, batch_size=batch_size)
    stage2_model = Stage2Network().to(device)
    stage2_model.load_state_dict(torch.load('checkpoint_stage2_first_train.pth'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(stage2_model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    train_losses, val_losses = train_and_validate_weights(stage1_model, stage2_model, train_loader_weights, test_loader_weights,
                                                          train_loader_cov, test_loader_cov,
                                                          criterion, optimizer, scheduler, device,
                                                          epochs_num, save_path, save_flag)

    plot_losses(train_losses, val_losses)