import torch
import time

def train_and_val_optuna(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    """
    A streamlined train and validation function for Optuna optimization.
    :param model: PyTorch model to be trained and validated.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param criterion: Loss function.
    :param optimizer: Optimizer for training.
    :param scheduler: Learning rate scheduler.
    :param device: Device to train on (CPU or GPU).
    :param epochs: Number of epochs to train.
    :return: Best validation loss achieved during training.
    """
    model.train()

    # Track best validation loss
    best_val_loss = float('inf')

    def run_epoch(loader, is_train):
        running_loss = 0.0
        model.train() if is_train else model.eval()

        with torch.set_grad_enabled(is_train):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)

                if is_train:
                    optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if is_train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

        avg_loss = running_loss / len(loader.dataset)
        return avg_loss

    for epoch in range(epochs):
        start_time = time.time()

        # Run training epoch
        train_loss = run_epoch(train_loader, is_train=True)

        # Run validation epoch
        val_loss = run_epoch(val_loader, is_train=False)

        # Update best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Step the scheduler
        scheduler.step()

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {time.time() - start_time:.2f} seconds")

    return best_val_loss
