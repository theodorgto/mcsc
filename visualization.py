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
