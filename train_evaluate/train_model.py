import torch
import matplotlib.pyplot as plt
import optuna
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

import models.model as model_funcs
import helper_functions.quant_lora as quant_funcs

def get_data_structure(batch_size=64, loader=False):
    # Download the training data from open datasets.
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    if loader:
        # Create data loaders:
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader

    return training_data, test_data

# Train model, use checkpoints, and store train loss and accuracy
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device, is_optuna=False, trial=None):
    # Move the model to the device
    model = model.to(device)

    # Prepare the loggers
    total_step = len(train_loader)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    # CIFAR10 mean and standard deviation:
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.201]

    # CIFAR100 mean and standard deviation:
    # mean =  [        0.5071,        0.4867,        0.4408    ]
    # std =  [        0.2675,        0.2565,        0.2761    ]

    normalize = transforms.Normalize(mean, std)

    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            images = normalize(images)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss.append(running_loss / total_step)
        train_acc.append(100 * correct / total)
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
              .format(epoch + 1, num_epochs, i + 1, total_step, running_loss / total_step, 100 * correct / total))
        # Validate the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            running_loss = 0.0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_loss.append(running_loss / len(val_loader))
            val_acc.append(100 * correct / total)
            print('Val Loss: {:.4f}, Val Accuracy: {:.2f}%'.format(running_loss / len(val_loader), 100 * correct / total))
        model.train()

        # At the end of the epoch, report the validation accuracy to Optuna
        if is_optuna:
            trial.report(val_acc[-1], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Save the model checkpoint
    if not is_optuna:
        torch.save(model.state_dict(), f'model_{num_epochs}epochs.ckpt')
    return train_loss, val_loss, train_acc, val_acc

# Plot the loss and accuracy
def plot_loss_acc(train_loss, val_loss, train_acc, val_acc):
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

def evaluate_test(model, criterion, test_loader, device):
    # calculate the model's accuracy on the test data:
    model.eval()
    correct = 0
    total = 0

    # CIFAR10 mean and standard deviation:
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.201]

    # CIFAR100 mean and standard deviation:
    # mean =  [        0.5071,        0.4867,        0.4408    ]
    # std =  [        0.2675,        0.2565,        0.2761    ]

    normalize = transforms.Normalize(mean, std)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # Move to device:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # Normalize the images batch:
            images = normalize(images)
            # Forward pass:
            outputs = model(images)
            # Evaluate the model:
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 10 == 0:
                print(f'For step {i} the accuracy is {100 * correct / total}%')

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

def optuna_objective(trial):
    # Define the hyperparameters space:
    lora_alpha = trial.suggest_categorical('lora_alpha', [24, 32, 40, 48])
    lora_rank = 2
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    lr = trial.suggest_float('lr', 1e-2, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    num_epochs = trial.suggest_int('num_epochs', 2, 7)

    # Define the model
    model = model_funcs.get_model()
    model = quant_funcs.quantize_lora(model, lora_rank, lora_alpha, quant_type='sparse')

    # Define the optimizer
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Define the loss function
    # TODO: change this!
    criterion = torch.nn.CrossEntropyLoss()

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the dataloaders
    train_data, _ = get_data_structure(batch_size=batch_size)
    # Split the train data into train and validation data
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Train the model
    train_loss, val_loss, train_acc, val_acc = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device)

    # Evaluate the model
    val_loss_res = val_loss[-1]

    return val_loss_res

def optuna_trials():
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name='cifar10-quant-lora', direction='maximize', sampler=sampler)
    study.optimize(optuna_objective, n_trials=10)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print(" Number of finished trials: ", len(study.trials))
    print(" Number of pruned trials: ", len(pruned_trials))
    print(" Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(" {}: {}".format(key, value))

    return study

if __name__ == '__main__':
    print("HERE!", optuna.__version__)