import torch
import keyboard  # Import the keyboard module
import numpy as np
import time
import os
from scipy.io import savemat

import os
import time
import numpy as np
import torch
from scipy.io import savemat


def train_and_validate_stage1(model, train_loader, val_loader, criterion, optimizer, scheduler,
                           device, epochs, last_epoch, save_path, save_flag):
    model.train()

    # Create a new folder for each run
    if save_flag:
        run_folder = os.path.join(save_path, f"stage1_run_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_folder, exist_ok=True)
        # Save the model architecture to a text file
        with open(os.path.join(run_folder, "model_architecture.txt"), 'w') as f:
            f.write(str(model))

    # Lists to store loss metrics per epoch
    train_losses = []
    val_losses = []
    val_maes = []
    best_val_loss = float('inf')
    val_interval = 5

    def run_epoch(loader, is_train):
        running_loss = 0.0
        running_mae = 0.0
        model.train() if is_train else model.eval()

        with torch.set_grad_enabled(is_train):
            for inputs, labels in loader:
                # Check if ESC key was pressed during training
                if is_train and keyboard.is_pressed('esc'):
                    raise KeyboardInterrupt

                inputs, labels = inputs.to(device), labels.to(device)
                if is_train:
                    optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                mae = torch.nn.functional.l1_loss(outputs, labels, reduction='mean')
                if is_train:
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_mae += mae.item() * inputs.size(0)

        avg_loss = running_loss / len(loader.dataset)
        avg_mae = running_mae / len(loader.dataset)
        return avg_loss, avg_mae

    try:
        for epoch in range(last_epoch, epochs):
            start_time = time.time()  # Start timing the epoch

            # Run training epoch
            epoch_train_loss, _ = run_epoch(train_loader, is_train=True)
            train_losses.append(epoch_train_loss)
            print(f"Epoch {epoch + 1}, Train Loss: {epoch_train_loss}, Time: {time.time() - start_time:.2f} seconds")

            # Run validation epoch if it's the appropriate interval
            if (epoch + 1) % val_interval == 0:
                start_time = time.time()  # Start timing the epoch
                epoch_val_loss, epoch_val_mae = run_epoch(val_loader, is_train=False)
                val_losses.append(epoch_val_loss)
                val_maes.append(epoch_val_mae)

                # Print validation loss and MAE
                print(f"Epoch {epoch + 1}, Validation Loss: {epoch_val_loss}, MAE: {epoch_val_mae}, Time: {time.time() - start_time:.2f} seconds")

                # Check if the current validation loss is the best
                if save_flag and epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch + 1,
                    }, os.path.join(run_folder, "checkpoint_stage1.pth"))

            # Step the scheduler
            scheduler.step()

    except KeyboardInterrupt:
        print("Training interrupted. Saving progress and exiting.")

    finally:
        if save_flag:
            info = {
                'train_loss': np.array(train_losses),
                'val_loss': np.array(val_losses),
                'val_mae': np.array(val_maes),
            }
            savemat(os.path.join(run_folder, "train_info_stage1.mat"), info)
            print("Information saved.")


def train_and_validate_stage2(model1, model2, train_loader_weights, val_loader_weights,
                              train_loader_cov, val_loader_cov,
                              train_loader_doa, val_loader_doa,
                              criterion, optimizer,
                              scheduler, device, epochs, save_path, save_flag):
    model1.eval()

    # Create a new folder for each run
    run_folder = os.path.join(save_path, f"stage2_run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_folder, exist_ok=True)

    # Save the model architecture to a text file
    with open(os.path.join(run_folder, "model2_architecture.txt"), 'w') as f:
        f.write(str(model2))

    # Lists to store loss metrics per epoch
    train_losses = []
    val_losses = []
    val_maes = []
    best_val_loss = float('inf')
    val_interval = 5

    def run_epoch(model1, model2, loader_weights, loader_cov, loader_doa, is_train):
        running_loss = 0.0
        running_mae = 0.0
        model2.train() if is_train else model2.eval()

        with torch.set_grad_enabled(is_train):
            for (_, labels_weights), (inputs_cov, labels_cov), (labels_doa) in zip(loader_weights, loader_cov, loader_doa):
                # Check if ESC key was pressed during training
                if is_train and keyboard.is_pressed('esc'):
                    raise KeyboardInterrupt

                inputs_cov, labels_cov = inputs_cov.to(device), labels_cov.to(device)
                labels_weights = labels_weights.to(device)
                labels_doa = labels_doa.to(device)

                outputs_stage1 = model1(inputs_cov)
                outputs = model2(outputs_stage1)

                if is_train:
                    optimizer.zero_grad()

                loss = criterion(outputs, labels_weights)
                mae = torch.nn.functional.l1_loss(outputs, labels_weights, reduction='mean')

                if is_train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * labels_weights.size(0)
                running_mae += mae.item() * labels_weights.size(0)

        avg_loss = running_loss / len(loader_weights.dataset)
        avg_mae = running_mae / len(loader_weights.dataset)
        return avg_loss, avg_mae

    try:
        for epoch in range(epochs):
            start_time = time.time()  # Start timing the epoch

            # Run training epoch
            epoch_train_loss, _ = run_epoch(model1, model2, train_loader_weights, train_loader_cov, train_loader_doa, is_train=True)
            train_losses.append(epoch_train_loss)
            print(f"Epoch {epoch + 1}, Train Loss: {epoch_train_loss}, Time: {time.time() - start_time:.2f} seconds")

            # Run validation epoch if it's the appropriate interval
            if (epoch + 1) % val_interval == 0:
                start_time = time.time()
                val_start_time = time.time()  # Start timing the validation loop
                epoch_val_loss, epoch_val_mae = run_epoch(model1, model2, val_loader_weights, val_loader_cov, val_loader_doa, is_train=False)
                val_losses.append(epoch_val_loss)
                val_maes.append(epoch_val_mae)

                # Print validation loss and MAE
                print(f"Validation Loss: {epoch_val_loss}, MAE: {epoch_val_mae}, Validation Time: {time.time() - val_start_time:.2f} seconds")

                # Check if the current validation loss is the best
                if save_flag and epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    torch.save({
                        'model_state_dict': model2.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch + 1,
                    }, os.path.join(run_folder, "checkpoint_stage2.pth"))

            # Step the scheduler
            scheduler.step()

    except KeyboardInterrupt:
        print("Training interrupted. Saving progress and exiting.")

    finally:
        if save_flag:
            info = {
                'train_loss': np.array(train_losses),
                'val_loss': np.array(val_losses),
                'val_mae': np.array(val_maes),
            }
            savemat(os.path.join(run_folder, "train_info_stage2.mat"), info)
            print("Information saved.")

    return train_losses, val_losses



