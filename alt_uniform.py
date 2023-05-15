import os
import shutil
import torch
import numpy as np
import math

from mmaction.apis import init_recognizer, inference_recognizer


def move_frames(percentage, array_scores):
    array_scores = np.array(
        array_scores.split(',')).astype(float)

    no_of_sections = 12
    section_rank = np.argsort(array_scores)

    # Directory path for number of total frames for original frames
    dir_path = r'F:\directory_path'

    total_frames = len(os.listdir(dir_path))
    no_of_frames = math.ceil(total_frames * percentage / 100)

    frames = [i for i in range(1, total_frames + 1)]
    frames = np.array_split(frames, no_of_sections)

    no_of_frames = [i for i in range(1, no_of_frames + 1)]
    no_of_frames = np.array_split(no_of_frames, no_of_sections)

    ind = np.array([])

    for idx, rank in enumerate(section_rank):
        ind = np.concatenate((ind, np.random.choice(
            frames[rank], len(no_of_frames[idx]), replace=False)), axis=None)

    ind = np.sort(ind).astype(int)

    for i in range(1, no_of_frames + 1):
        # Specify the directory of original frames, F:\directory_path\img_{:05}.jpg
        src = r'F:\directory_path\img_{:05}.jpg'.format(ind[i - 1] + 1)

        # Specify the destination for selected frames, F:\destination_path\img_{:05}.jpg
        dst = r'F:\destination_path\img_{:05}.jpg'.format(i)

        shutil.copyfile(src, dst)


percentage = 20

config_file = 'config.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'checkpoint.pth'

# assign the desired device.
device = 'cuda:0'  # or 'cpu'
device = torch.device(device)

# build the model from a config file and a checkpoint file
model = init_recognizer(config_file, checkpoint_file, device)

# array_scores means the array of image similarity scores, which is generated through cv2_generate.py or ssim_generate.py
# e.g. array_scores = [0.9, 0.85, 0.7, 0.76 ...]
move_frames(percentage, array_scores)

# Specify the destination for selected frames, F:\destination_path\img_{:05}.jpg
video = r'F:\destination_path'
results = inference_recognizer(model, video)

# show the results
labels = open('tools/data/ucf101/label_map.txt').readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in results]

print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])

