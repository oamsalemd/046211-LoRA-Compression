import sys
proj_path = "/home/ohada/DeepProject/ProjectPath"
if proj_path not in sys.path:
    sys.path.append(proj_path)

from tqdm import tqdm
import os
import torch
import matplotlib.pyplot as plt
import optuna
import pandas
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

import models.model as model_funcs
import helper_functions.quant_lora as quant_funcs


def get_cifar10_data(batch_size=64, loader=False):
    # Download the training data from open datasets.
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.201]

    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )

    if loader:
        # Create data loaders:
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader

    return training_data, test_data


def get_imagenet_data(batch_size=64, loader=False):
    data_path = '/home/ohada/DeepProject/archive/imagenet_validation'
    # Create a dataset from the ImageNet data:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    imagenet_data = datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )

    # split 'imagenet_data' to training and test data:
    train_size = int(0.8 * len(imagenet_data))
    test_size = len(imagenet_data) - train_size
    training_data, test_data = torch.utils.data.random_split(imagenet_data, [train_size, test_size])

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

    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
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
            print(
                'Val Loss: {:.4f}, Val Accuracy: {:.2f}%'.format(running_loss / len(val_loader), 100 * correct / total))
        model.train()

    if is_optuna:
        trial.report(val_loss[-1], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return train_loss, val_loss, train_acc, val_acc


# Plot the loss and accuracy
def plot_loss_acc(model, lora_rank, train_loss, val_loss, train_acc, val_acc):
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.xticks(range(len(train_loss)))
    plt.xlabel('epochs')
    plt.legend()
    # Save figure to file:
    plt.savefig(f'loss_{model}_r={lora_rank}.png')
    plt.figure()
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.xticks(range(len(train_loss)))
    plt.xlabel('epochs')
    plt.legend()
    # Save figure to file:
    plt.savefig(f'acc_{model}_r={lora_rank}.png')


def evaluate_test(model, test_loader, device):
    # calculate the model's accuracy on the test data:
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # Move to device:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # Forward pass:
            outputs = model(images)
            # Evaluate the model:
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # show progress using 'tqdm':
            # tqdm.write(f'For step {i} the accuracy is {100 * correct / total}%')
            # if i % 10 == 0:
            #     print(f'For step {i} the accuracy is {100 * correct / total}%')

    print(f'Accuracy of the network on {total} test images: {100 * correct / total}%')
    return 100 * correct / total


def optuna_objective(trial, dataset='imagenet', lora_rank=2):
    # Define the hyperparameters space:
    lora_alpha = trial.suggest_categorical('lora_alpha', [8, 16, 24, 32, 40, 48])
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    num_epochs = trial.suggest_int('num_epochs', 2, 10)

    # Define the model
    if dataset == 'imagenet':
        model = model_funcs.get_imagenet_model()
    else:
        model = model_funcs.get_cifar10_model()
    quant_funcs.quantize_lora(model, lora_rank, lora_alpha, quant_type='sparse')

    # Define the optimizer
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the dataloaders
    if dataset == 'imagenet':
        train_data, _ = get_imagenet_data(batch_size=batch_size)
    else:
        train_data, _ = get_cifar10_data(batch_size=batch_size)
    # Split the train data into train and validation data
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Train the model
    train_loss, val_loss, train_acc, val_acc = train_model(model, criterion, optimizer, train_loader, val_loader,
                                                           num_epochs, device, is_optuna=True, trial=trial)

    # Evaluate the model
    val_acc_res = val_acc[-1]

    return val_acc_res


def optuna_trials():
    studies = []
    # Store 'dataset', 'lora_rank', 'evaluation' data frame:
    evaluation = pandas.DataFrame(columns=['dataset', 'lora_rank', 'evaluation'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for dataset in ['imagenet', 'cifar10']:
        if dataset == 'cifar10':
            eval_model = model_funcs.get_cifar10_model()
            eval_res = evaluate_test(eval_model, get_cifar10_data(batch_size=64, loader=True)[1], device)
        else:
            eval_model = model_funcs.get_imagenet_model()
            eval_res = evaluate_test(eval_model, get_imagenet_data(batch_size=64, loader=True)[1], device)

        new_row = {'dataset': dataset, 'lora_rank': 'original', 'evaluation': eval_res}
        evaluation = pandas.concat([evaluation, pandas.DataFrame([new_row])], ignore_index=True)
        quant_funcs.quantize_linear_layers(eval_model, quant_type='sparse')
        if dataset == 'cifar10':
            eval_res = evaluate_test(eval_model, get_cifar10_data(batch_size=64, loader=True)[1], device)
        else:
            eval_res = evaluate_test(eval_model, get_imagenet_data(batch_size=64, loader=True)[1], device)
        new_row = {'dataset': dataset, 'lora_rank': 'quantized', 'evaluation': eval_res}
        evaluation = pandas.concat([evaluation, pandas.DataFrame([new_row])], ignore_index=True)

        lora_rank = []
        if dataset == 'cifar10':
            lora_rank = [2, 4, 6]
        elif dataset == 'imagenet':
            lora_rank = [2, 4, 8, 16, 32]
        for rank in lora_rank:
            sampler = optuna.samplers.TPESampler()
            study = optuna.create_study(study_name=f'{dataset}-quant-lora-r={rank}', direction='maximize',
                                        sampler=sampler)
            study.optimize(lambda trial: optuna_objective(trial, dataset, rank), n_trials=10)

            # pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            # complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

            optimizer = study.best_params['optimizer']
            lr = study.best_params['lr']
            batch_size = study.best_params['batch_size']
            num_epochs = study.best_params['num_epochs']
            lora_alpha = study.best_params['lora_alpha']

            if dataset == 'cifar10':
                model = model_funcs.get_cifar10_model()
                quant_funcs.quantize_lora(model, rank, lora_alpha, quant_type='sparse')
                train_data, test_data = get_cifar10_data(batch_size, loader=True)
            else:
                model = model_funcs.get_imagenet_model()
                quant_funcs.quantize_lora(model, rank, lora_alpha, quant_type='sparse')
                train_data, test_data = get_imagenet_data(batch_size, loader=True)

            if optimizer == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            train_loss, val_loss, train_acc, val_acc = train_model(model, torch.nn.CrossEntropyLoss(), optimizer, train_data, test_data, num_epochs, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), is_optuna=False)
            # Evaluate the model:
            eval_res = evaluate_test(model, test_data, device)
            # Save the trained model:
            torch.save(model.state_dict(), f'model_{dataset}_r={rank}_eval={eval_res}.ckpt')
            # Add an empty line to 'evaluation' data frame:
            new_row = {'dataset': dataset, 'lora_rank': rank, 'evaluation': eval_res}
            evaluation = pandas.concat([evaluation, pandas.DataFrame([new_row])], ignore_index=True)
            # Plot the loss and accuracy:
            plot_loss_acc(dataset, rank, train_loss, val_loss, train_acc, val_acc)

            # print("Study statistics: ")
            # print(" Number of finished trials: ", len(study.trials))
            # print(" Number of pruned trials: ", len(pruned_trials))
            # print(" Number of complete trials: ", len(complete_trials))
            print(f"Best trial ({dataset}), ({rank}):")
            trial = study.best_trial
            print(" Value: ", trial.value)
            print(" Params: ")
            for key, value in trial.params.items():
                print(" {}: {}".format(key, value))

            studies.append(study)
    # Save the evaluation data frame to file:
    evaluation.to_csv('evaluation.csv')
    # For study in studies, save the study to file:
    for study in studies:
        # Write HTML report:
        optuna.visualization.plot_optimization_history(study).write_html(f'{study.study_name}_optimization_history.html')

    return studies


if __name__ == '__main__':
    os.chdir('/home/ohada/DeepProject/ProjectPath/results')
    studies = optuna_trials()
    print(studies)
