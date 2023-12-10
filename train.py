import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset

from torch.utils.data import random_split
import torch.nn.init as init
import torchvision
import os
import cv2
import pandas as pd

label = {
    'PXL_20231006_105621514.TS': (1,1),
    'PXL_20231006_105839274.TS': (1,2),
    'PXL_20231006_105912563.TS': (1,3),
    'PXL_20231006_111220943.TS': (1,4),
    'PXL_20231006_111317388.TS': (1,5),

    'PXL_20231006_110008414.TS': (2,1),
    'PXL_20231006_110103438.TS': (2,2),
    'PXL_20231006_110136554.TS': (2,3),
    'PXL_20231006_111150785.TS': (2,4),
    'PXL_20231006_111403886.TS': (2,5),

    'PXL_20231006_110253143.TS': (3,1),
    'PXL_20231006_110454262.TS': (3,2),
    'PXL_20231006_110614512.TS': (3,3),
    'PXL_20231006_111117602.TS': (3,4),
    'PXL_20231006_111451715.TS': (3,5),

    'PXL_20231006_110837241.TS': (4,1),
    'PXL_20231006_110927004.TS': (4,2),
    'PXL_20231006_111012449.TS': (4,4),
    'PXL_20231006_111042814.TS': (4,5),
}

class imageDataset(Dataset):
    # Dictionary related postion to x,y coordinates.
    # x is along the long axis and y is along the short axis

    def __init__(self, folder_path, device=None, transform=None, resize=True):
        # data loading
        self.folder_path = folder_path
        self.transform = transform
        self.resize = resize
        self.image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.position = [label[file_name.split(".mp4")[0]] for file_name in self.image_files]
        self.device = device

    def __len__(self):
        return len(self.image_files)
        # dataset[0]

    def __getitem__(self, index):
        img_path = os.path.join(self.folder_path, self.image_files[index])
        img = cv2.imread(img_path)

        if self.resize:
            img = cv2.resize(img, (1080 // 10, 1920 // 10))
            img = img / 255.0  # Normalize pixel values

        if self.transform:
            img = torch.tensor(img,dtype=torch.float)
            img = img.permute(2, 1, 0)

            pos = torch.tensor(self.position[index],dtype=torch.float)
        
        if self.device:
            img = img.to(self.device)
            pos = pos.to(self.device)

        return img,pos


# define network
class Model(nn.Module):

    def __init__(self, n_outputs=2):
        super(Model, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=8, stride=3),  # First convolutional layer
            nn.ReLU(),  # SoftPlus, ReLu
            nn.MaxPool2d(2),  # Max pooling layer
            nn.Conv2d(3, 1, kernel_size=4, stride=1),  # Second convolutional layer
            nn.ReLU(),  # SoftPlus, ReLu
            nn.MaxPool2d(2),  # Max pooling layer
            # You can add a dropout layer here if needed
            nn.Flatten(),
            nn.Linear(in_features=98, out_features=98),  # Adjust the in_features based on the output size of the previous layer
            nn.ReLU(),
            nn.Linear(in_features=98, out_features=n_outputs),
       )

        # Apply weight initialization
        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)


    def forward(self, x):
        # Get features before the flattening operation
        features = self.seq[:5](x)  # Slices the sequence up to the flattening layer
        latent_layer = features

        # Continue with the rest of the network
        logits = self.seq[5:](features)
        return logits, latent_layer


def train(model, criterion, optimizer, train_loader, valid_loader, epochs, device, exp_folder):

    total_samples = len(dataset)
    total_batches = total_samples // train_loader.batch_size

    num_epochs = epochs

    # Logging
    train_loss = np.zeros(num_epochs)
    valid_loss = np.zeros(num_epochs)


    for epoch in range(num_epochs):
    # loop over all batches
        model.train()
        train_losses, train_lengths = 0, 0
        for i, (images, pos) in enumerate(train_loader):
            model.train() # To enable dropout
            optimizer.zero_grad()                            # Clear gradients for the next train
            output,_ = model(images)                           # Forward pass: Compute predicted y by passing x to the model
            batch_loss = criterion(output, pos)              # Compute loss
            train_losses += batch_loss
            batch_loss.backward()                            # Backward pass: Compute gradient of the loss with respect to model parameters
            optimizer.step()

            train_lengths += len(images)

        # Divide by the total accumulated batch sizes
        train_losses /= train_lengths
        train_loss[epoch] = train_losses.item()
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss : {train_losses.item()}")

        # Do the validaiton
        model.eval()
        val_losses, val_lengths = 0, 0
        for i, (images, pos) in enumerate(valid_loader):
            output,_ = model(images)
            val_losses += criterion(output, pos)
            val_lengths += len(images)

        # Divide by the total accumulated batch sizes
        val_losses /= val_lengths
        valid_loss[epoch] = val_losses.item()
        print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss : {val_losses.item()}")

    # save best model
    best_model_path = os.path.join(exp_folder, 'best_model.pt')
    torch.save(model.state_dict(), best_model_path)

    # Save the logs as CSV
    log_df = pd.DataFrame({
        'valid_loss': valid_loss,
        'train_loss': train_loss,
    })

    log_csv_path = os.path.join(exp_folder, 'training_log.csv')
    log_df.to_csv(log_csv_path, index=False)


#%% RUN
if __name__ == '__main__':
    exp_folder = os.getcwd() + '/experiments'
    os.makedirs(exp_folder, exist_ok=True) # create experiments folder if it doesn't exist
    # set path
    data_path = os.getcwd() + '/data'
    samples_folder = data_path + '/sampled_videos'
    print(samples_folder)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # init dataset
    dataset = imageDataset(samples_folder,transform=True,device=device)

    # split dataset randomly into tran and validation.
    generator1 = torch.Generator().manual_seed(42) # This is optional but setting it will make outcome reproducible
    train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2], generator=generator1)


    #dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)

    # initialize model
    model = Model().to(device)
    # Optimizer and Loss
    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) # Weight decay, L1, L2 regularization to reduce overfitting

    ### Static parameters
    epochs = 1

    print(f'Training with parameters \n{model}\n')

    train(model=model, 
          criterion=criterion, 
          optimizer=optimizer, 
          train_loader=train_loader,
          valid_loader=valid_loader, 
          epochs=epochs,
          device=device,
          exp_folder=exp_folder)
    