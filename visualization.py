import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.colors import rgb_to_hsv
import numpy as np
from train import Model

import torch
# import Dataset
from torch.utils.data import Dataset
import cv2

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

# # get newest experiment id
# if args.exp_id is None:
#     exp_list = os.listdir('experiments')
#     exp_id = max(exp_list)
# else:
#     exp_id = args.exp_id

exp_folder = os.path.join(os.getcwd(), 'experiments')
print(f'Loading experiment from {exp_folder}')

log_df_path = os.path.join(exp_folder, 'training_log.csv')

try:
    log_df = pd.read_csv(log_df_path)
    print(f'Logging data has been loaded')
    logging_plots = True
except FileNotFoundError:
    print(f'No training log found at {log_df_path}.')
    # exit()
    logging_plots = False

# Plotting the losses
plt.figure(figsize=(10, 6))

valid_loss = log_df['valid_loss']
train_loss = log_df['train_loss']
num_epochs = len(valid_loss)

plt.plot(range(num_epochs), valid_loss, 'b', label='Validation loss')
plt.plot(range(num_epochs), train_loss, 'r', label='Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Total Loss')

plt.tight_layout()
plt.show()

# save plots
plot_path = os.path.join(exp_folder, 'loss_plot.png')
plt.savefig(plot_path)
print(f'Plot saved to {plot_path}')


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

# Load the best model (or a specific checkpoint)

model = Model()  # Replace with your actual model architecture
try:
    # get the best model
    best_model_path = os.path.join(exp_folder, 'best_model.pt')
    model.load_state_dict(torch.load(best_model_path))
    print(f'Best model has been loaded')

except FileNotFoundError or NameError:
    print(f'No best model found at best_model/best_model.pt.')


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
data_path = os.getcwd() + '/data'
samples_folder = data_path + '/sampled_videos'
dataset = imageDataset(samples_folder)


for image, pos in dataset:
    output,_ = model(image)
    color = color_mapping.get(tuple(pos.tolist()))
    pred_positions.append((output.detach().numpy(),color))
    true_positions.append((pos.detach().numpy(),color))
for pos,color in pred_positions:
    # print(pos)
    plt.scatter(pos[0,1],pos[0,0],color=color)

for pos,color in true_positions:
    # print(pos)
    plt.scatter(pos[1],pos[0],color=color,marker='x')

plt.xlim(0,width)
plt.ylim(0,height)
# Set axis scale to be constant
plt.axis('equal')
plt.show()

# save plots
plot_path = os.path.join(exp_folder, 'prediction_plot.png')
plt.savefig(plot_path)
print(f'Plot saved to {plot_path}')


# Visualize the latent feature map
img,pos = dataset[45]
pred_pos,latent_layer = model(img)

print(f"Latent_layer shape : {latent_layer.shape}")

print(f"Labbled pos {pos}. Prediction {pred_pos}")

fig, axs = plt.subplots(1, 2, figsize=(8, 6))

# Plot the latent feature map
latent_layer_map = torch.rot90(latent_layer[0].flip(1))
axs[0].imshow(latent_layer_map.detach().numpy(), cmap='viridis', aspect='auto')
axs[0].set_title('Latent Layer')
axs[0].set_xlabel('Feature Index')
axs[0].set_ylabel('Value')
axs[0].set_aspect('auto')  # Adjust aspect ratio to match the number of features

# Plot the image
image_array = img.permute(2, 1, 0)
axs[1].imshow(image_array, aspect='auto')
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
image_array = img.permute(2, 1, 0).numpy()

# Convert the RGB image to grayscale using the Viridis colormap
hsv_image = rgb_to_hsv(image_array)
gray_image = hsv_image[:, :, 2]

# Create a figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(8, 6))

# Plot the latent feature map
latent_layer_map = torch.rot90(latent_layer[0].flip(1))
axs[0].imshow(latent_layer_map.detach().numpy(), cmap='viridis', aspect='auto')
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