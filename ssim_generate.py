import os
import torch
from skimage.io import imread
from piq import ssim, SSIMLoss

images = []
scores = []

# Directory path of frames for image similartiy comparison
dataset_path = r'F:\dataset_path'

for file in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, file)
    images.append(torch.tensor(imread(img_path)
                               ).permute(2, 0, 1).unsqueeze(0))

for i in range(len(images) - 1):
    scores.append(
        ssim(images[i].cuda(), images[i + 1].cuda(), data_range=255).item())

print(scores)
