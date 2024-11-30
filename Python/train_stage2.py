# import keyboard  # Import the keyboard module
# import os
# import time
# import numpy as np
# import torch
# from scipy.io import savemat
#
# def train_and_validate_stage2(model1, model2, train_loader_weights, val_loader_weights,
#                               train_loader_cov, val_loader_cov,
#                               train_loader_doa, val_loader_doa,
#                               criterion, optimizer,
#                               scheduler, device, epochs, first_epoch, save_path, save_flag):
#     model1.eval()
#
#     if save_flag:
#         # Create a new folder for each run
#         run_folder = os.path.join(save_path, f"stage2_run_{time.strftime('%Y%m%d_%H%M%S')}")
#         os.makedirs(run_folder, exist_ok=True)
#         # Save the model architecture to a text file
#         with open(os.path.join(run_folder, "model2_architecture.txt"), 'w') as f:
#             f.write(str(model2))
#
#     # Lists to store loss metrics per epoch
#     train_losses = []
#     val_losses = []
#     val_maes = []
#     best_val_loss = float('inf')
#     val_interval = 5
#     best_model_state_dict = None
#
#     # Lists to store inputs, outputs, and labels for final evaluation
#     train_inputs_cov = []
#     train_outputs_stage1 = []
#     train_outputs = []
#     train_labels_weights = []
#     val_inputs_cov = []
#     val_outputs_stage1 = []
#     val_outputs = []
#     val_labels_weights = []
#
#     # Function to track gradient norms
#     def get_gradient_norms(model):
#         total_norm = 0.0
#         for p in model.parameters():
#             if p.grad is not None:
#                 param_norm = p.grad.data.norm(2)
#                 total_norm += param_norm.item() ** 2
#         total_norm = total_norm ** 0.5
#         return total_norm
#
#     # Function to track weights statistics
#     def get_weights_statistics(model):
#         weights_mean = 0.0
#         weights_var = 0.0
#         count = 0
#         for p in model.parameters():
#             if p.data is not None:
#                 weights_mean += p.data.mean().item()
#                 weights_var += p.data.var().item()
#                 count += 1
#         return weights_mean / count, weights_var / count
#
#     def run_epoch(model1, model2, loader_weights, loader_cov, loader_doa, is_train, is_last_epoch=False):
#         running_loss = 0.0
#         running_mae = 0.0
#         model2.train() if is_train else model2.eval()
#
#         inputs_list = []
#         outputs_stage1_list = []
#         outputs_list = []
#         labels_weights_list = []
#
#         with torch.set_grad_enabled(is_train):
#             for (_, labels_weights), (inputs_cov, labels_cov), (labels_doa) in zip(loader_weights, loader_cov, loader_doa):
#                 # Check if ESC key was pressed during training
#                 if is_train and keyboard.is_pressed('esc'):
#                     raise KeyboardInterrupt
#
#                 inputs_cov, labels_cov = inputs_cov.to(device), labels_cov.to(device)
#                 labels_weights = labels_weights.to(device)
#                 labels_doa = labels_doa.to(device)
#
#                 # outputs_stage1 = model1(inputs_cov)
#                 outputs = model2(labels_cov[:,0,:,:], labels_doa)
#
#                 if is_last_epoch:
#                     inputs_list.append(inputs_cov.cpu().numpy())
#                     outputs_stage1_list.append(outputs_stage1.cpu().numpy())
#                     outputs_list.append(outputs.cpu().numpy())
#                     labels_weights_list.append(labels_weights.cpu().numpy())
#
#                 if is_train:
#                     optimizer.zero_grad()
#
#                 loss = criterion(outputs, labels_weights)
#                 mae = torch.nn.functional.l1_loss(outputs, labels_weights, reduction='mean')
#
#                 if is_train:
#                     loss.backward()
#
#                     # Track gradient norms
#                     grad_norm = get_gradient_norms(model2)
#                     print(f"Epoch {epoch + 1}, Gradient Norm: {grad_norm}")
#
#                     # Optional: Clip gradients if they are exploding
#                     torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=5.0)
#
#                     optimizer.step()
#
#                 running_loss += loss.item() * labels_weights.size(0)
#                 running_mae += mae.item() * labels_weights.size(0)
#
#         avg_loss = running_loss / len(loader_weights.dataset)
#         avg_mae = running_mae / len(loader_weights.dataset)
#
#         return avg_loss, avg_mae, inputs_list, outputs_stage1_list, outputs_list, labels_weights_list
#
#     try:
#         for epoch in range(first_epoch, epochs):
#             start_time = time.time()  # Start timing the epoch
#
#             # Run training epoch
#             epoch_train_loss, _, _, _, _, _ = run_epoch(model1, model2, train_loader_weights, train_loader_cov, train_loader_doa, is_train=True)
#             train_losses.append(epoch_train_loss)
#             print(f"Epoch {epoch + 1}, Train Loss: {epoch_train_loss}, Time: {time.time() - start_time:.2f} seconds")
#
#             # Track weights statistics
#             weights_mean, weights_var = get_weights_statistics(model2)
#             print(f"Epoch {epoch + 1}, Weights Mean: {weights_mean}, Weights Variance: {weights_var}")
#
#             # Run validation epoch if it's the appropriate interval
#             if (epoch + 1) % val_interval == 0:
#                 start_time = time.time()
#                 val_start_time = time.time()  # Start timing the validation loop
#                 epoch_val_loss, epoch_val_mae, _, _, _, _ = run_epoch(model1, model2, val_loader_weights, val_loader_cov, val_loader_doa, is_train=False)
#                 val_losses.append(epoch_val_loss)
#                 val_maes.append(epoch_val_mae)
#
#                 # Print validation loss and MAE
#                 print(f"Validation Loss: {epoch_val_loss}, MAE: {epoch_val_mae}, Validation Time: {time.time() - val_start_time:.2f} seconds")
#
#                 # Check if the current validation loss is the best
#                 if save_flag and epoch_val_loss < best_val_loss:
#                     best_val_loss = epoch_val_loss
#                     best_model_state_dict = model2.state_dict()
#                     torch.save({
#                         'model_state_dict': best_model_state_dict,
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         'scheduler_state_dict': scheduler.state_dict(),
#                         'epoch': epoch + 1,
#                     }, os.path.join(run_folder, "checkpoint_stage2.pth"))
#
#             # Step the scheduler
#             scheduler.step()
#
#     except KeyboardInterrupt:
#         print("Training interrupted. Saving progress and exiting.")
#
#     finally:
#         if save_flag:
#             print("Running final evaluation and saving inputs/outputs...")
#             info = {
#                 'train_loss': np.array(train_losses),
#                 'val_loss': np.array(val_losses),
#                 'val_mae': np.array(val_maes),
#             }
#             savemat(os.path.join(run_folder, "train_info_stage2.mat"), info)
#
#             _, _, train_inputs_cov_list, train_outputs_stage1_list, train_outputs_list, train_labels_weights_list = (
#                 run_epoch(model1, model2, train_loader_weights, train_loader_cov, train_loader_doa, is_train=False, is_last_epoch=True))
#
#             # model2.load_state_dict(best_model_state_dict)
#             _, _, val_inputs_cov_list, val_outputs_stage1_list, val_outputs_list, val_labels_weights_list = (
#                 run_epoch(model1, model2, val_loader_weights, val_loader_cov, val_loader_doa, is_train=False, is_last_epoch=True))
#
#             # Save all collected inputs/outputs and labels separately for train and validation
#             train_inputs_cov.extend(train_inputs_cov_list)
#             train_outputs_stage1.extend(train_outputs_stage1_list)
#             train_outputs.extend(train_outputs_list)
#             train_labels_weights.extend(train_labels_weights_list)
#
#             val_inputs_cov.extend(val_inputs_cov_list)
#             val_outputs_stage1.extend(val_outputs_stage1_list)
#             val_outputs.extend(val_outputs_list)
#             val_labels_weights.extend(val_labels_weights_list)
#
#             data = {
#                 'train_inputs_cov': np.vstack(train_inputs_cov) if train_inputs_cov else np.array([]),
#                 'train_outputs_stage1': np.vstack(train_outputs_stage1) if train_outputs_stage1 else np.array([]),
#                 'train_outputs': np.vstack(train_outputs) if train_outputs else np.array([]),
#                 'train_labels_weights': np.vstack(train_labels_weights) if train_labels_weights else np.array([]),
#                 'val_inputs_cov': np.vstack(val_inputs_cov) if val_inputs_cov else np.array([]),
#                 'val_outputs_stage1': np.vstack(val_outputs_stage1) if val_outputs_stage1 else np.array([]),
#                 'val_outputs': np.vstack(val_outputs) if val_outputs else np.array([]),
#                 'val_labels_weights': np.vstack(val_labels_weights) if val_labels_weights else np.array([]),
#             }
#             savemat(os.path.join(run_folder, "data.mat"), data)
#             print("Final inputs, outputs, and labels saved to data.mat.")
#
#     return
#
#
#
#
#
#
#
#
#
#






import keyboard  # Import the keyboard module
import os
import time
import numpy as np
import torch
from scipy.io import savemat


def train_and_validate_stage2(model1, model2, train_loader_weights, val_loader_weights,
                              train_loader_cov, val_loader_cov,
                              train_loader_doa, val_loader_doa,
                              criterion, optimizer,
                              scheduler, device, epochs, first_epoch, save_path, save_flag):
    model1.eval()

    if save_flag:
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
    best_model_state_dict = None

    # Lists to store inputs, outputs, and labels for final evaluation
    train_inputs_cov = []
    train_outputs_stage1 = []
    train_outputs = []
    train_labels_weights = []
    val_inputs_cov = []
    val_outputs_stage1 = []
    val_outputs = []
    val_labels_weights = []

    def run_epoch(model1, model2, loader_weights, loader_cov, loader_doa, is_train, is_last_epoch=False):
        running_loss = 0.0
        running_mae = 0.0
        model2.train() if is_train else model2.eval()

        inputs_list = []
        outputs_stage1_list = []
        outputs_list = []
        labels_weights_list = []

        with torch.set_grad_enabled(is_train):
            for (labels_weights), (inputs_cov, labels_cov), (labels_doa) in zip(loader_weights, loader_cov, loader_doa):
                # Check if ESC key was pressed during training
                if is_train and keyboard.is_pressed('esc'):
                    raise KeyboardInterrupt

                inputs_cov, labels_cov = inputs_cov.to(device), labels_cov.to(device)
                labels_weights = labels_weights.to(device)
                labels_doa = labels_doa.to(device)

                outputs_stage1 = model1(inputs_cov)
                # outputs_stage1 = torch.tensor([])
                outputs = model2(outputs_stage1[:,0,:,:], labels_doa)

                if is_last_epoch:
                    inputs_list.append(inputs_cov.cpu().numpy())
                    outputs_stage1_list.append(outputs_stage1.cpu().numpy())
                    outputs_list.append(outputs.cpu().numpy())
                    labels_weights_list.append(labels_weights.cpu().numpy())

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

        return avg_loss, avg_mae, inputs_list, outputs_stage1_list, outputs_list, labels_weights_list

    try:
        for epoch in range(first_epoch, epochs):
            start_time = time.time()  # Start timing the epoch

            # Run training epoch
            epoch_train_loss, _, _, _, _, _ = run_epoch(model1, model2, train_loader_weights, train_loader_cov, train_loader_doa, is_train=True)
            train_losses.append(epoch_train_loss)
            print(f"Epoch {epoch + 1}, Train Loss: {epoch_train_loss}, Time: {time.time() - start_time:.2f} seconds")

            # Run validation epoch if it's the appropriate interval
            if (epoch + 1) % val_interval == 0:
                start_time = time.time()
                val_start_time = time.time()  # Start timing the validation loop
                epoch_val_loss, epoch_val_mae, _, _, _, _ = run_epoch(model1, model2, val_loader_weights, val_loader_cov, val_loader_doa, is_train=False)
                val_losses.append(epoch_val_loss)
                val_maes.append(epoch_val_mae)

                # Print validation loss and MAE
                print(f"Validation Loss: {epoch_val_loss}, MAE: {epoch_val_mae}, Validation Time: {time.time() - val_start_time:.2f} seconds")

                # Check if the current validation loss is the best
                if save_flag and epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_model_state_dict = model2.state_dict()
                    torch.save({
                        'model_state_dict': best_model_state_dict,
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
            print("Running final evaluation and saving inputs/outputs...")
            info = {
                'train_loss': np.array(train_losses),
                'val_loss': np.array(val_losses),
                'val_mae': np.array(val_maes),
            }
            savemat(os.path.join(run_folder, "train_info_stage2.mat"), info)

            _, _, train_inputs_cov_list, train_outputs_stage1_list, train_outputs_list, train_labels_weights_list = (
                run_epoch(model1, model2, train_loader_weights, train_loader_cov, train_loader_doa, is_train=False, is_last_epoch=True))

            model2.load_state_dict(best_model_state_dict)
            _, _, val_inputs_cov_list, val_outputs_stage1_list, val_outputs_list, val_labels_weights_list = (
                run_epoch(model1, model2, val_loader_weights, val_loader_cov, val_loader_doa, is_train=False, is_last_epoch=True))

            # Save all collected inputs/outputs and labels separately for train and validation
            train_inputs_cov.extend(train_inputs_cov_list)
            train_outputs_stage1.extend(train_outputs_stage1_list)
            train_outputs.extend(train_outputs_list)
            train_labels_weights.extend(train_labels_weights_list)

            val_inputs_cov.extend(val_inputs_cov_list)
            val_outputs_stage1.extend(val_outputs_stage1_list)
            val_outputs.extend(val_outputs_list)
            val_labels_weights.extend(val_labels_weights_list)

            data = {
                'train_inputs_cov': np.vstack(train_inputs_cov) if train_inputs_cov else np.array([]),
                'train_outputs_stage1': np.vstack(train_outputs_stage1) if train_outputs_stage1 else np.array([]),
                'train_outputs': np.vstack(train_outputs) if train_outputs else np.array([]),
                'train_labels_weights': np.vstack(train_labels_weights) if train_labels_weights else np.array([]),
                'val_inputs_cov': np.vstack(val_inputs_cov) if val_inputs_cov else np.array([]),
                'val_outputs_stage1': np.vstack(val_outputs_stage1) if val_outputs_stage1 else np.array([]),
                'val_outputs': np.vstack(val_outputs) if val_outputs else np.array([]),
                'val_labels_weights': np.vstack(val_labels_weights) if val_labels_weights else np.array([]),
            }
            savemat(os.path.join(run_folder, "data.mat"), data)
            print("Final inputs, outputs, and labels saved to data.mat.")

    return
