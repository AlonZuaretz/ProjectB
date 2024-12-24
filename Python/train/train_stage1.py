import keyboard  # Import the keyboard module
import os
import time
import numpy as np
import torch
import wandb

from scipy.io import savemat

def train_and_val(model, train_loader, val_loader, criterion, optimizer, scheduler,
                              device, first_epoch, epochs, save_path, save_flag):
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
    best_model_state_dict = None

    # Lists to store inputs, outputs, and labels for final evaluation
    train_inputs = []
    train_outputs = []
    train_labels = []
    val_inputs = []
    val_outputs = []
    val_labels = []

    def run_epoch(loader, is_train, is_last_epoch=False):
        running_loss = 0.0
        running_mae = 0.0
        model.train() if is_train else model.eval()

        inputs_list = []
        outputs_list = []
        labels_list = []

        with torch.set_grad_enabled(is_train):
            for inputs, labels in loader:
                # Check if ESC key was pressed during training
                if is_train and keyboard.is_pressed('esc'):
                    raise KeyboardInterrupt

                inputs, labels = inputs.to(device), labels.to(device)
                if is_train:
                    optimizer.zero_grad()

                outputs = model(inputs)

                if is_last_epoch:
                    inputs_list.append(inputs.cpu().numpy())
                    outputs_list.append(outputs.cpu().numpy())
                    labels_list.append(labels.cpu().numpy())

                loss = criterion(outputs, labels)
                mae = torch.nn.functional.l1_loss(outputs, labels, reduction='mean')
                if is_train:
                    loss.backward()
                    optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_mae += mae.item() * inputs.size(0)

        avg_loss = running_loss / len(loader.dataset)
        avg_mae = running_mae / len(loader.dataset)
        return avg_loss, avg_mae, inputs_list, outputs_list, labels_list

    try:
        for epoch in range(first_epoch, epochs):
            start_time = time.time()  # Start timing the epoch

            # Run training epoch
            epoch_train_loss, _, _, _, _ = run_epoch(train_loader, is_train=True)
            train_losses.append(epoch_train_loss)
            print(f"Epoch {epoch + 1}, Train Loss: {epoch_train_loss}, Time: {time.time() - start_time:.2f} seconds")

            # wandb.log({"epoch": epoch + 1, "train_loss": epoch_train_loss})

            # Run validation epoch if it's the appropriate interval
            if (epoch + 1) % val_interval == 0:
                start_time = time.time()  # Start timing the epoch
                epoch_val_loss, epoch_val_mae, _, _, _ = run_epoch(val_loader, is_train=False)
                val_losses.append(epoch_val_loss)
                val_maes.append(epoch_val_mae)

                # Print validation loss and MAE
                print(f"Epoch {epoch + 1}, Validation Loss: {epoch_val_loss}, MAE: {epoch_val_mae}, Time: {time.time() - start_time:.2f} seconds")

                # wandb.log({
                #     "epoch": epoch + 1,
                #     "val_loss": epoch_val_loss,
                #     "val_mae": epoch_val_mae
                # })

                # Check if the current validation loss is the best
                if save_flag and epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_model_state_dict = model.state_dict()
                    torch.save({
                        'model_state_dict': best_model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch + 1,
                    }, os.path.join(run_folder, "checkpoint_stage1.pth"))

                # wandb.log({"best_val_loss": best_val_loss})

            # Step the scheduler
            scheduler.step()

    except KeyboardInterrupt:
        print("Training interrupted. Saving progress and exiting.")

    finally:
        if save_flag:
            print("Running final evaluation and saving inputs/outputs...")
            info = {
                'train_loss': np.array(train_losses),
                'val_loss': np.array(val_losses),
                'val_mae': np.array(val_maes),
            }
            savemat(os.path.join(run_folder, "train_info_stage1.mat"), info)

            _, _, train_inputs_list, train_outputs_list, train_labels_list = run_epoch(train_loader, is_train=False, is_last_epoch=True)
            if best_model_state_dict is not None:
                model.load_state_dict(best_model_state_dict)
                _, _, val_inputs_list, val_outputs_list, val_labels_list = run_epoch(val_loader, is_train=False, is_last_epoch=True)
            else:
                val_inputs_list, val_outputs_list, val_labels_list = [], [], []

            # Save all collected inputs/outputs and labels separately for train and validation
            train_inputs.extend(train_inputs_list)
            train_outputs.extend(train_outputs_list)
            train_labels.extend(train_labels_list)

            val_inputs.extend(val_inputs_list)
            val_outputs.extend(val_outputs_list)
            val_labels.extend(val_labels_list)

            data = {
                'train_inputs': np.vstack(train_inputs) if train_inputs else np.array([]),
                'train_outputs': np.vstack(train_outputs) if train_outputs else np.array([]),
                'train_labels': np.vstack(train_labels) if train_labels else np.array([]),
                'val_inputs': np.vstack(val_inputs) if val_inputs else np.array([]),
                'val_outputs': np.vstack(val_outputs) if val_outputs else np.array([]),
                'val_labels': np.vstack(val_labels) if val_labels else np.array([]),
            }
            savemat(os.path.join(run_folder, "data.mat"), data)
            print("Final inputs, outputs, and labels saved to data.mat.")
    return
