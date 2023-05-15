import os
import shutil
import torch
import numpy as np
import math

from mmaction.apis import init_recognizer, inference_recognizer


def split_three(lst, ratio=[0.5, 0.35, 0.15]):
    import numpy as np

    first, second, third = ratio
    assert (np.sum(ratio) == 1.0)  # makes sure the splits make sense
    # note we only need to give the first 2 indices to split, the last one it returns the rest of the list or empty
    indicies_for_splitting = [
        int(len(lst) * first), int(len(lst) * (first+second))]
    no_of_frames_ratio = np.split(lst, indicies_for_splitting)
    return no_of_frames_ratio


def move_frames(percentage, array_scores):
    array_scores = np.array(
        array_scores).astype(float)

    no_of_frames = math.ceil((len(array_scores) + 1) * percentage / 100)

    no_of_sections = 3
    sections_array_scores = np.array_split(array_scores, no_of_sections)
    no_of_frames_ratio = split_three([i for i in range(0, no_of_frames)])

    sd_scores = []
    for array_scores in sections_array_scores:
        sd_scores.append(np.std(array_scores))

    sections_order = np.argpartition(
        sd_scores, -no_of_sections)[-no_of_sections:]

    frames = []
    ind = []

    for i in range(0, no_of_sections):
        frames.append(len(no_of_frames_ratio[i]))

    for idx, section in enumerate(sections_order):
        if frames[idx] != 0:
            temp = np.argpartition(sections_array_scores[section], frames[idx])[
                :frames[idx]]
            buffer = sum(len(x) for x in sections_array_scores[:section])
            ind = ind + [x + buffer for x in temp]

    ind = np.sort(ind)

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
