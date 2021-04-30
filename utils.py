from timeit import default_timer as timer
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch
import pandas as pd


class CustomImageDataset(Dataset):
    """Custom image dataset for load images from dataframe
    Params:

        path: путь допапки с фотоками
        data: df путями и лэйблами
        transform: объект с transform functions
    
    Returns:

        image: готовое изображение
        label: тренировочные метки
    """

    def __init__(self, data, path, transform=None):
        self.path = path
        self.data = data.values
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extraction label and path ot image
        img_name, label = self.data[idx]

        # Load image
        img_path = os.path.join(self.path, img_name)
        image = cv2.imread(img_path)[:, :, [2, 1, 0]]

        # Transform if you need
        if self.transform:
            image = self.transform(image)

        return image, label


def train_loop(model,
               criterion,
               optimizer,
               train_loader,
               valid_loader,
               train_on_gpu,
               save_file_name,
               max_epochs_stop=3,
               n_epochs=20,
               print_every=1,
               path_to_save='.'):
    """Train function for PyTorch model
    Params:

        model: модель
        criterion: лосс
        optimizer: опитимизатор
        train_loader: train loader
        valid_loader: validation loader
        save_file_name (str): путь для сохранения лучшей модели
        max_epochs_stop (int): максимальное количество эпох без улучшения лоса до ранней остановки
        n_epochs (int): количество эпох
        print_every (int): частота эпох для печати статистики

    Returns:

        model: натренированная модель
        history (DataFrame): история тренировки
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        # print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), (os.path.join(path_to_save, save_file_name)))
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(os.path.join(path_to_save, save_file_name)))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])

    history.to_csv(os.path.join(path_to_save, 'history.csv'))
    return model, history


def check_result(model,
                 loader,
                 criterion,
                 on_gpu):
    """Evaluate function for PyTorch model
    Params:

        model: модель
        criterion: лосс
        train_loader: train loader
        on_gpu: переносить ли вычисления на GPU

    Returns:
        print final metrics
    """
    model.eval()
    check_loss = 0.0
    check_acc = 0.0

    for data, target in loader:
        # Tensors to gpu
        if on_gpu:
            data, target = data.cuda(), target.cuda()

        # Forward pass
        output = model(data)

        # Validation loss
        loss = criterion(output, target)
        check_loss += loss.item() * data.size(0)

        # Calculate accuracy
        _, pred = torch.max(output, dim=1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        accuracy = torch.mean(
            correct_tensor.type(torch.FloatTensor))
        check_acc += accuracy.item() * data.size(0)

    # Calculate average losses
    check_loss = check_loss / len(loader.dataset)

    # Calculate average accuracy
    check_acc = check_acc / len(loader.dataset)

    print(
        f'loss: {check_loss:.2f} and acc: {100 * check_acc:.2f}%'
    )
