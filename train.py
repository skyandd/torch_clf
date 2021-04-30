import os
import torch
import argparse

import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import models
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from utils import train_loop, CustomImageDataset, check_result

# Parse input params
parser = argparse.ArgumentParser(description='Load data')
parser.add_argument("-d", "--workdir",
                    required=True,
                    help='path to workdir')

parser.add_argument("-e", "--epoch", type=int, default=5,
                    help='N epoch for train model')

parser.add_argument("-b", "--batch", type=int, default=32,
                    help='Size batch for train')

parser.add_argument("-fn", "--filename", default='final_model',
                    help='Filename best model')

args = parser.parse_args()

# Save parsed params to variable
BATCH_SIZE = args.batch
N_CLASSES = 10
SAVE_FILE_NAME = args.filename
N_EPOCH = args.epoch
PATH = args.workdir

# Load data
data = pd.read_csv(os.path.join(PATH,  "data/imagewoof2-320/noisy_imagewoof.csv"))

# Cat labels to int
dict_label = {j: i for i, j in enumerate(set(data['noisy_labels_0']))}
data = data[['path'] + ['noisy_labels_0']]
data['noisy_labels_0'] = data['noisy_labels_0'].map(dict_label)

# Create datasets
train_path = [i for i in data['path'] if i.split('/')[0] == 'train']
val_path = [i for i in data['path'] if i.split('/')[0] == 'val']
train = data[data['path'].isin(train_path)]
test = data[data['path'].isin(val_path)]
train, valid, _, _ = train_test_split(train,
                                      train['noisy_labels_0'],
                                      test_size=0.1,
                                      stratify=train['noisy_labels_0'])

# Create data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_data = CustomImageDataset(train, os.path.join(PATH, "/data/imagewoof2-320"), data_transforms['train'])
valid_data = CustomImageDataset(valid, os.path.join(PATH, "data/imagewoof2-320"), data_transforms['val'])
test_data = CustomImageDataset(test, os.path.join(PATH, "data/imagewoof2-320"), data_transforms['test'])

# Create dataloader
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Freeze weights
model = models.mobilenet_v2(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Create model
# Add some layers to after last layer model
model.classifier[1] = nn.Sequential(
    nn.Linear(model.last_channel, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, N_CLASSES),
    nn.Softmax(dim=1))

# Whether to train on a gpu
train_on_gpu = torch.cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

# Number of gpus
if train_on_gpu:
    gpu_count = torch.cuda.device_count()
    # print(f'{gpu_count} gpus detected.')
    if gpu_count > 1:
        multi_gpu = True
    else:
        multi_gpu = False
if train_on_gpu:
    model = model.to('cuda')
if multi_gpu:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=10e-4)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

if __name__ == '__main__':
    # Running the model

    print('Starting Training with params: \nN_EPOCH = {},'
          ' \nBATCH_SIZE = {},'
          ' \nTRAIN_ON_GPU = {},'
          ' \nSAVE_FILE_NAME = {},'
          ' \nPATH_TO_SAVE = {}'.format(
        N_EPOCH, BATCH_SIZE,
        train_on_gpu, SAVE_FILE_NAME, PATH
    ))

    model, history = train_loop(
        model,
        criterion,
        optimizer_ft,
        train_loader,
        valid_loader,
        train_on_gpu,
        save_file_name=SAVE_FILE_NAME,
        max_epochs_stop=3,
        n_epochs=N_EPOCH,
        print_every=1,
        path_to_save=PATH
    )

    print("Starting check result on test data")
    # Check test result
    check_result(model, test_loader, criterion, train_on_gpu)
