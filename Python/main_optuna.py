import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from data_helper import extract_data, create_dataloaders_cov
from neural_nets import Stage1Network
from train import train_and_validate_cov

# Load data once to avoid loading it in every iteration
path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4\dataForPython_train.mat"
save_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\ProjectB\dataV4\NN_results"
Xw, Yw, XR, YR, params = extract_data(path)

# Global parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Objective function for Optuna
def objective(trial):
    # Hyperparameters to optimize
    learning_rate = 5e-3
    step_size = 7
    gamma = 0.5

    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
    kernel_size = trial.suggest_categorical('kernel_size', [2, 3])
    activation = trial.suggest_categorical('activation', ['leaky_relu', 'relu'])
    optimizer_type = trial.suggest_categorical('optimizer_type', ['Adam', 'AdamW'])
    epochs = 70  # Fixed number of epochs for faster experimentation

    # Data loaders (created once per trial)
    train_loader_cov, val_loader_cov, _, _ = create_dataloaders_cov(XR, YR, pre_method=1, batch_size=batch_size)

    # Initialize model, optimizer, and scheduler
    model = Stage1Network(kernel_size=kernel_size, activation=activation).to(device)
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.MSELoss()

    # Training and validation loop
    train_losses, val_losses = train_and_validate_cov(model, train_loader_cov, val_loader_cov, criterion, optimizer,
                                                      scheduler,
                                                      device, epochs, save_path="", save_flag=False)
    # Print current trial information
    print(
        f"Trial {trial.number}: Learning Rate={learning_rate}, Batch Size={batch_size}, Step Size={step_size}, Gamma={gamma}, Kernel Size={kernel_size}, Activation={activation}, Optimizer Type={optimizer_type}")
    print(f"    Minimum Validation Loss: {min(val_losses)}")

    # Optuna will try to minimize the validation loss
    return min(val_losses)


def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Print best trial information
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Plot optimization history
    fig, ax = plt.subplots()
    ax.plot(study.trials_dataframe()["number"], study.trials_dataframe()["value"], marker='o')
    ax.set_title("Optimization History")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation Loss")
    ax.grid(True)
    plt.show()

    # Plot parameter importance
    try:
        from optuna.importance import get_param_importances
        importances = get_param_importances(study)
        fig, ax = plt.subplots()
        ax.barh(list(importances.keys()), list(importances.values()))
        ax.set_title("Hyperparameter Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Hyperparameter")
        plt.show()
    except ImportError:
        print("Optuna importance module is not available.")


if __name__ == "__main__":
    main()
