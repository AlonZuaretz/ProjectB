import torch
import keyboard  # Import the keyboard module


def train_and_validate_cov(model, train_loader, val_loader, criterion, optimizer, scheduler,
                           device, epochs, save_path, save_flag):
    model.train()

    # Lists to store loss metrics per epoch
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    try:
        for epoch in range(epochs):
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

            print(f'Epoch {epoch + 1}, Train Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}')

            # Check if the current validation loss is the best
            if save_flag and running_val_loss < best_val_loss:
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
                'train_loss': train_losses,
                'val_loss': val_losses,
            }
            torch.save(info, save_path + r"\train_info.pth")
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
            model2.train()  # Set model to training mode
            running_train_loss = 0.0

            # Training loop
            for (_, labels_weights), (inputs_cov, labels_cov), (labels_doa)\
                    in zip(train_loader_weights, train_loader_cov, train_loader_doa):
                # Check if ESC key was pressed
                if keyboard.is_pressed('esc'):
                    raise KeyboardInterrupt

                labels_weights = labels_weights.to(device)
                inputs_cov, labels_cov = inputs_cov.to(device), labels_cov.to(device)
                labels_doa = labels_doa.to(device)
                labels_stacked = torch.cat((labels_weights, labels_doa), dim=1)

                outputs_stage1 = model1(inputs_cov)
                inputs_stacked = torch.cat((inputs_cov, outputs_stage1), dim=1)
                outputs = model2(inputs_stacked)
                optimizer.zero_grad()
                loss = criterion(outputs, labels_stacked)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * labels_weights.size(0)

            scheduler.step()
            #             # Check gradients every epoch
            #             # for name, parameter in model2.named_parameters():
            #     if parameter.grad is not None:
            #         grad_norm = parameter.grad.norm()
            #         if grad_norm < 1e-6:
            #             print(f'Vanishing gradient detected in {name}: {grad_norm}')
            #         if grad_norm > 1e6:
            #             print(f'Exploding gradient detected in {name}: {grad_norm}')

            # Calculate average training loss for the epoch
            epoch_train_loss = running_train_loss / len(train_loader_weights.dataset)
            train_losses.append(epoch_train_loss)
            print(f'Epoch {epoch + 1}, Train Loss: {epoch_train_loss}')
            # Validation loop
            if (epoch + 1) % 2 == 0:
                model2.eval()  # Set model to evaluation mode
                running_val_loss = 0.0

                with torch.no_grad():
                    for (_, labels_weights), (inputs_cov, labels_cov), (labels_doa) \
                            in zip(val_loader_weights, val_loader_cov, val_loader_doa):

                        labels_weights = labels_weights.to(device)
                        inputs_cov, labels_cov = inputs_cov.to(device), labels_cov.to(device)
                        labels_doa = labels_doa.to(device)
                        labels_stacked = torch.cat((labels_weights, labels_doa), dim=1)

                        outputs_stage1 = model1(inputs_cov)
                        inputs_stacked = torch.cat((inputs_cov, outputs_stage1), dim=1)
                        outputs = model2(inputs_stacked)
                        loss = criterion(outputs, labels_stacked)

                        running_val_loss += loss.item() * labels_weights.size(0)

                # Calculate average validation loss for the epoch
                epoch_val_loss = running_val_loss / len(val_loader_weights.dataset)
                val_losses.append(epoch_val_loss)

                print(f'Validation Loss: {epoch_val_loss}')

                # Check if the current validation loss is the best
                if save_flag and running_val_loss < best_val_loss:
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
                'train_loss': train_losses,
                'val_loss': val_losses,
            }
            torch.save(info, save_path + r"\train_info_stage2.pth")
            print("Information saved.")

    return train_losses, val_losses
