import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset

from matplotlib.colors import rgb_to_hsv
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

    def __init__(self, dropout_rate=0.25, n_outputs = 2):
        super(Model, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=8, stride=3),  # First convolutional layer
            nn.ReLU(),  # SoftPlus, ReLu
            nn.MaxPool2d(2),  # Max pooling layer
            # nn.Dropout(p=dropout_rate),  # Dropout layer

            nn.Conv2d(3, 1, kernel_size=4, stride=1),  # Second convolutional layer
            nn.ReLU(),  # SoftPlus, ReLu
            nn.MaxPool2d(2),  # Max pooling layer
            # nn.Dropout(p=dropout_rate),  # Dropout layer

            nn.Flatten(),
            nn.Linear(in_features=98, out_features=98),  # Adjust the in_features based on the output size of the previous layer
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Dropout layer

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

def evaluate(model, dataset, exp_folder):
    color_mapping = {

        (1, 1): "Pink",
        (2, 1): "Cyan",
        (3, 1): "Magenta",
        (4, 1): "Brown",


        (1, 2): "Lime",
        (2, 2): "Teal",
        (3, 2): "Indigo",
        (4, 2): "Violet",


        (1, 3): "Maroon",
        (2, 3): "Navy",
        (3, 3): "Aquamarine",
        (4, 3): "Turquoise",


        (1, 4): "Blue",
        (2, 4): "Green",
        (3, 4): "Yellow",
        (4, 4): "Purple",

        (4, 5): "Red",
        (3, 5): "Olive",
        (2, 5): "Gray",
        (1, 5): "Orange",
    }
    plt.figure(figsize=(10, 6))

    width = 5.3
    height = 4.17

    dx = width / 1000
    dy = height / 1000

    x = np.arange(0,width,dx)
    y = np.arange(0,height,dy)

    # plt.plot(x,y)

    for i in range(int(height) + 1):
        plt.plot(x, i * np.ones_like(x), color='grey',alpha=0.5)

    for i in range(int(width) + 1):
        plt.plot(i * np.ones_like(y),y, color='grey',alpha=0.5)

    plt.scatter(4,3, color='red',marker='+',linewidth=2)

    # Get model predictions:
    model.eval()
    pred_positions = []
    true_positions = []


    for image, pos in dataset:
        output,_ = model(image)
        color = color_mapping.get(tuple(pos.tolist()))
        pred_positions.append((output.cpu(),color))
        true_positions.append((pos.cpu(),color))
    for pos,color in pred_positions:
        # print(pos)
        plt.scatter(pos[0,1].detach().numpy(),pos[0,0].detach().numpy(),color=color)

    for pos,color in true_positions:
        # print(pos)
        plt.scatter(pos[1].detach().numpy(),pos[0].detach().numpy(),color=color,marker='x')

    plt.xlim(0,width)
    plt.ylim(0,height)
    # Set axis scale to be constant
    plt.axis('equal')
    plt.show()

    # save plots
    plot_path = os.path.join(exp_folder, 'prediction_plot.png')
    plt.savefig(plot_path)
    print(f'Plot saved to {plot_path}')

def show_latent_space(model, dataset, exp_folder):
        
    # Visualize the latent feature map
    img,pos = dataset[45]
    pred_pos,latent_layer = model(img)

    fig, axs = plt.subplots(1, 2, figsize=(8, 6))

    # Plot the latent feature map
    latent_layer_map = torch.rot90(latent_layer[0].flip(1))
    axs[0].imshow(latent_layer_map.cpu().detach().numpy(), cmap='viridis', aspect='auto')
    axs[0].set_title('Latent Layer')
    axs[0].set_xlabel('Feature Index')
    axs[0].set_ylabel('Value')
    axs[0].set_aspect('auto')  # Adjust aspect ratio to match the number of features

    # Plot the image
    image_array = img.permute(2, 1, 0)
    axs[1].imshow(image_array.cpu(), aspect='auto')
    axs[1].set_title('Image')
    axs[1].set_xlabel('Width')
    axs[1].set_ylabel('Height')

    plt.tight_layout()
    plt.show()

    # save plots
    plot_path = os.path.join(exp_folder, 'latent_plot.png')
    plt.savefig(plot_path)
    print(f'Plot saved to {plot_path}')

    # Assuming img is an RGB image with shape (3, 108, 192)
    image_array = img.permute(2, 1, 0).cpu().numpy()

    # Convert the RGB image to grayscale using the Viridis colormap
    hsv_image = rgb_to_hsv(image_array)
    gray_image = hsv_image[:, :, 2]

    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))

    # Plot the latent feature map
    latent_layer_map = torch.rot90(latent_layer[0].flip(1))
    axs[0].imshow(latent_layer_map.cpu().detach().numpy(), cmap='viridis', aspect='auto')
    axs[0].set_title('Latent Layer')
    axs[0].set_xlabel('Feature Index')
    axs[0].set_ylabel('Value')

    # Plot the grayscale image using the Viridis colormap
    axs[1].imshow(-gray_image, cmap='viridis', aspect='auto')
    axs[1].set_title('Grayscale Image')
    axs[1].set_xlabel('Width')
    axs[1].set_ylabel('Height')

    plt.tight_layout()
    plt.show()

    # save plots
    plot_path = os.path.join(exp_folder, 'latent_plot_2.png')
    plt.savefig(plot_path)
    print(f'Plot saved to {plot_path}')

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
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # initialize model
    model = Model().to(device)
    # Optimizer and Loss
    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) # Weight decay, L1, L2 regularization to reduce overfitting

    ### Static parameters
    epochs = 50

    print(f'Training with parameters \n{model}\n')

    train(model=model, 
          criterion=criterion, 
          optimizer=optimizer, 
          train_loader=train_loader,
          valid_loader=valid_loader, 
          epochs=epochs,
          device=device,
          exp_folder=exp_folder)
    
    evaluate(model=model, 
          dataset=dataset, 
          exp_folder=exp_folder)
    
    show_latent_space(model=model, 
          dataset=dataset, 
          exp_folder=exp_folder)
    
