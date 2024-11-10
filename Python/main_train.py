import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from data_helper import extract_data, create_dataloaders_cov
from neural_nets import Stage1Network
from train import train_and_validate
import optuna.visualization as ov


def objective(trial, data_path):
    # Hyperparameters to be optimized
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])

    # Data loading
    Xw, Yw, XR, YR, params = extract_data(data_path)
    train_loader, val_loader, idx_train, idx_test = create_dataloaders_cov(XR, YR, batch_size=batch_size)

    # Setup model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Stage1Network().to(device)
    criterion = nn.MSELoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses, val_losses = train_and_validate(model, train_loader, val_loader,
                                                  criterion, optimizer, device, epochs=100)

    # Objective value to minimize
    last_val_loss = val_losses[-1]
    return last_val_loss


if __name__ == "__main__":
    data_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV1_new\allSIR\dataForPython.mat"

    # Create a partial function that includes data_path
    objective = partial(objective, data_path=data_path)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("Best hyperparameters:", study.best_trial.params)

    # Visualization of the optimization
    ov.plot_optimization_history(study).show()
    ov.plot_slice(study).show()
    ov.plot_parallel_coordinate(study).show()
    ov.plot_contour(study).show()
