import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import os
import pandas as pd

from torch.optim.lr_scheduler import StepLR, ExponentialLR
from data.data_helper import extract_data, create_dataloaders
from opt_optuna.neural_nets_optuna import Stage1NetworkOptuna  # Use Optuna-compatible model
from opt_optuna.train_optuna import train_and_val_optuna


def objective(trial):
    # Define the hyperparameter space
    batch_size = trial.suggest_categorical("batch_size", [512, 1024])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    step_size = trial.suggest_int("step_size", 4, 10)
    gamma = trial.suggest_uniform("gamma", 0.1, 0.9)
    architecture_choice = trial.suggest_int("architecture_choice", 0, 4)

    # Data paths
    base_path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\dataV6"
    data_path = base_path + r"\dataForPython.mat"

    # Extract data and create dataloaders
    Xw, Yw, XR, XRd, YR, params = extract_data(data_path)
    cov_train_loader, cov_val_loader, _, _, _, _, _, _, _ = create_dataloaders(XR, XRd, YR, Yw, batch_size)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = Stage1NetworkOptuna(architecture_choice).double().to(device)

    # Set optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Set scheduler
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Loss function
    criterion = nn.MSELoss()

    # Training and validation using the new train function
    val_loss = train_and_val_optuna(
        model, cov_train_loader, cov_val_loader, criterion, optimizer, scheduler, device, epochs=50)

    # Save trial details and results to a dictionary
    trial_results = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "step_size": step_size,
        "gamma": gamma,
        "architecture_choice": architecture_choice,
        "val_loss": val_loss,
    }
    results_list.append(trial_results)  # Append the results for saving later

    return val_loss  # Minimize validation loss


if __name__ == "__main__":
    # Initialize a list to store trial results
    results_list = []

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    # Save the best trial
    print("Best trial:")
    print(study.best_trial)

    # Save best hyperparameters
    best_params = study.best_params
    print("Best parameters:", best_params)

    # Save results to a CSV file
    results_df = pd.DataFrame(results_list)
    save_path = r"C:\Users\alon.zuaretz\Documents\GitHub\ProjectB\Python\opt_optuna\optuna_results.csv"
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
