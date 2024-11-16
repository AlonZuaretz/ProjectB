import torch
import keyboard  # Import the keyboard module
import numpy as np
import time
from scipy.io import savemat


def train_and_validate_cov(model, train_loader, val_loader, criterion, optimizer, scheduler,
                           device, epochs, save_path, save_flag):
    model.train()

    # Lists to store loss metrics per epoch
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epoch_times = []

    try:
        for epoch in range(epochs):
            start_time = time.time()  # Start timing the epoch
            model.train()  # Set model to training mode
            running_train_loss = 0.0

            # Training loop
            for inputs, labels in train_loader:
                # Check if ESC key was pressed
                if keyboard.is_pressed('esc'):
                    raise KeyboardInterrupt

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * inputs.size(0)

            scheduler.step()

            # Calculate average training loss for the epoch
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)

            # Validation loop
            model.eval()  # Set model to evaluation mode
            running_val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * inputs.size(0)

            # Calculate average validation loss for the epoch
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)

            end_time = time.time()  # End timing the epoch
            epoch_time = end_time - start_time
            epoch_times.append(epoch_time)

            print(f'Epoch {epoch + 1}, Train Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}, Time: {epoch_time:.2f} seconds')

            # Check if the current validation loss is the best
            if save_flag and epoch_val_loss < best_val_loss:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                }, save_path + r"\checkpoint_stage1.pth")

    except KeyboardInterrupt:
        print("Training interrupted. Saving progress and exiting.")

    finally:
        if save_flag:
            info = {
                'train_loss': np.array(train_losses),
                'val_loss': np.array(val_losses),
            }
            savemat(save_path + r"\train_info_stage1.mat", info)
            print("Information saved.")

    return train_losses, val_losses


def train_and_validate_weights(model1, model2, train_loader_weights, val_loader_weights,
                               train_loader_cov, val_loader_cov,
                               train_loader_doa, val_loader_doa,
                               criterion, optimizer,
                               scheduler, device, epochs, save_path, save_flag):
    model1.eval()

    # Lists to store loss metrics per epoch
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    try:
        for epoch in range(epochs):
            start_time = time.time()  # Start timing the epoch
            model2.train()  # Set model to training mode
            running_train_loss = 0.0

            # Training loop
            for (_, labels_weights), (inputs_cov, _), (labels_doa)\
                    in zip(train_loader_weights, train_loader_cov, train_loader_doa):
                # Check if ESC key was pressed
                if keyboard.is_pressed('esc'):
                    raise KeyboardInterrupt

                inputs_cov = inputs_cov.to(device)
                labels_weights = labels_weights.to(device)

                outputs_stage1 = model1(inputs_cov)
                outputs = model2(outputs_stage1)
                optimizer.zero_grad()
                loss = criterion(outputs, labels_weights)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * labels_weights.size(0)

            scheduler.step()

            # Calculate average training loss for the epoch
            epoch_train_loss = running_train_loss / len(train_loader_weights.dataset)
            train_losses.append(epoch_train_loss)
            end_time = time.time()  # End timing the training loop
            epoch_time = end_time - start_time
            print(f'Epoch {epoch + 1}, Train Loss: {epoch_train_loss}, Time: {epoch_time:.2f} seconds')


            # Validation loop
            if (epoch + 1) % 5 == 0:
                val_start_time = time.time()  # Start timing the validation loop
                model2.eval()  # Set model to evaluation mode
                running_val_loss = 0.0

                with torch.no_grad():
                    for (_, labels_weights), (inputs_cov, _), (labels_doa) \
                            in zip(val_loader_weights, val_loader_cov, val_loader_doa):

                        inputs_cov = inputs_cov.to(device)
                        labels_weights = labels_weights.to(device)

                        outputs_stage1 = model1(inputs_cov)
                        outputs = model2(outputs_stage1)
                        loss = criterion(outputs, labels_weights)

                        running_val_loss += loss.item() * labels_weights.size(0)

                # Calculate average validation loss for the epoch
                epoch_val_loss = running_val_loss / len(val_loader_weights.dataset)
                val_losses.append(epoch_val_loss)

                val_end_time = time.time()  # End timing the validation loop
                val_time = val_end_time - val_start_time

                print(f'Validation Loss: {epoch_val_loss}, Validation Time: {val_time:.2f} seconds')

                # Check if the current validation loss is the best
                if save_flag and epoch_val_loss < best_val_loss:
                    torch.save({
                        'model_state_dict': model2.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch + 1,
                    }, save_path + r"\checkpoint_stage2.pth")

    except KeyboardInterrupt:
        print("Training interrupted. Saving progress and exiting.")

    finally:
        if save_flag:
            info = {
                'train_loss': np.array(train_losses),
                'val_loss': np.array(val_losses),
            }
            savemat(save_path + r"\train_info_stage2.mat", info)
            print("Information saved.")

    return train_losses, val_losses
