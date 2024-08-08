import torch


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path, save_flag):
    model.train()

    # Lists to store loss metrics per epoch
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_train_loss = 0.0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

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
            torch.save(model.state_dict(), save_path + r"\checkpoint.pth")  # Save the model's state_dict

    if save_flag:
        info = {
            'train_loss': train_losses,
            'val_loss': val_losses,
        }
        torch.save(info, save_path + r"\train_info.pth")

    return train_losses, val_losses




