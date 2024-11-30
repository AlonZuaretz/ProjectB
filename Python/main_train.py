import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR

from data_helper import (extract_data, create_dataloaders_cov, create_dataloaders_weights,
                         create_dataloaders_doa)
from neural_nets import Stage1Network, Stage2Network
from train_stage1 import train_and_validate_stage1
from train_stage2 import train_and_validate_stage2


def load_checkpoint(folder_path, model, optimizer, scheduler, device):
    # Construct the checkpoint path
    checkpoint1_path = os.path.join(folder_path, "checkpoint_stage1.pth")
    checkpoint2_path = os.path.join(folder_path, "checkpoint_stage2.pth")

    # Load the checkpoint
    if os.path.exists(checkpoint1_path):
        checkpoint = torch.load(checkpoint1_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded successfully from {checkpoint1_path}")
        last_epoch = checkpoint['epoch']
        return last_epoch
    elif os.path.exists(checkpoint2_path):
        checkpoint = torch.load(checkpoint2_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded successfully from {checkpoint2_path}")
        last_epoch = checkpoint['epoch']
        return last_epoch
    else:
        print(f"Checkpoint not found")
        return 0




if __name__ == "__main__":
    # path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4\dataForPython_train.mat"
    # save_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4\NN_results"

    path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV4\dataForPython_train.mat"
    save_path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV4\NN_results"
    stage1_load_path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV4\NN_results\stage1_run3_20241129_074726"
    stage2_load_path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV4\NN_results\stage2_run1_20241129_081706"

    save_flag = True
    stage1_load_flag = True
    stage2_load_flag = False

    Xw, Yw, XR, XRd, YR, Ydoa, params = extract_data(path)
    training_stage = [2]

    # Global parameters:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###### Stage 1 parameters ######:
    batch_size = 1024
    epochs_num = 200
    learning_rate = 5e-3
    pre_method = 1

    train_loader_cov, test_loader_cov, idx_train, idx_test = (
        create_dataloaders_cov(XR, YR, XRd, pre_method, batch_size=batch_size))

    stage1_model = Stage1Network().to(device)
    optimizer = optim.Adam(stage1_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Load from checkpoint:
    if stage1_load_flag:
        last_epoch = load_checkpoint(stage1_load_path, stage1_model, optimizer, scheduler, device)
    else:
        last_epoch = 0

    ###### Stage 1 training: #######
    if 1 in training_stage:
        train_and_validate_stage1(stage1_model, train_loader_cov, test_loader_cov,
                                                             criterion, optimizer, scheduler, device, last_epoch,
                                                             epochs_num, save_path, save_flag)

    ######## Stage 2 training: #######
    if 2 in training_stage:
        batch_size = 2048
        epochs_num = 100
        learning_rate = 1e-2
        num_workers = 0

        train_loader_cov, test_loader_cov, idx_train, idx_test = (
            create_dataloaders_cov(XR, YR, XRd, pre_method, batch_size=batch_size, num_workers=num_workers))
        train_loader_doa, test_loader_doa = create_dataloaders_doa(Ydoa, batch_size=batch_size, num_workers=num_workers)
        train_loader_weights, test_loader_weights, _, _ = create_dataloaders_weights(Xw, Yw, batch_size=batch_size, num_workers=num_workers)

        stage2_model = Stage2Network().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(stage2_model.parameters(), lr=learning_rate, momentum=0.5, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        if stage2_load_flag:
            last_epoch = load_checkpoint(stage2_load_path, stage2_model, optimizer, scheduler, device)
        else:
            last_epoch = 0


        train_and_validate_stage2(stage1_model, stage2_model,
                                         train_loader_weights, test_loader_weights,
                                         train_loader_cov, test_loader_cov,
                                         train_loader_doa, test_loader_doa,
                                         criterion, optimizer, scheduler, device,
                                         epochs_num, last_epoch, save_path, save_flag)

