import torch


def train_and_validate(model, train_loader, test_loader, criterion, optimizer, device, epochs=10, save_path='best_model.pth'):
    best_validation_loss = float('inf')  # Initialize the best validation loss to a large number

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            # Loss
            loss = criterion(output, target)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                validation_loss += loss.item() * data.size(0)

        validation_loss /= len(test_loader.dataset)

        print(f'Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}')

        # Check if the current validation loss is the best
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            print(f"Saving new best model with Validation Loss: {best_validation_loss:.4f}")
            torch.save(model.state_dict(), save_path)  # Save the model's state_dict


