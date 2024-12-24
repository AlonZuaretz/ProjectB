import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb

from torch.optim.lr_scheduler import StepLR
from data.data_helper import extract_data, create_dataloaders
from NNs.neural_nets import Stage1Network
from train_stage1 import train_and_val


def load_checkpoint(folder_path, model, optimizer, scheduler, device):
    # Construct the checkpoint path
    checkpoint_path = os.path.join(folder_path, "checkpoint_stage1.pth")

    # Load the checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded successfully from {checkpoint_path}")
        last_epoch = checkpoint['epoch']
        return last_epoch
    else:
        print(f"Checkpoint not found")
        return 0


if __name__ == "__main__":

    base_path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV6"
    # base_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV5"
    path = base_path + r"\dataForPython.mat"
    save_path = base_path + r"\NN_results"
    load_path = base_path + r""

    save_flag = True
    load_flag = False

    Xw, Yw, XR, XRd, YR, Ydoa, params = extract_data(path)

    # Global parameters:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1024
    epochs_num = 100
    learning_rate = 1e-3
    pre_method = 1

    cov_train_loader, cov_val_loader, _, _, _, _, _, _, _, _, _, _ = create_dataloaders(XR, XRd, YR, Yw, Ydoa)

    # wandb.init(project="Phased Array Project", name="V6 data, self attetion")
    # wandb.config = {
    #     "learning_rate": learning_rate,
    #     "batch_size": batch_size,
    #     "epochs": epochs_num
    # }


    model = Stage1Network().double().to(device)
    # wandb.watch(model, log="all", log_freq=10)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Load from checkpoint:
    if load_flag:
        first_epoch = load_checkpoint(load_path, model, optimizer, scheduler, device)
    else:
        first_epoch = 0



    train_and_val(model, cov_train_loader, cov_val_loader,
                  criterion, optimizer, scheduler, device, first_epoch,
                  epochs_num, save_path, save_flag)