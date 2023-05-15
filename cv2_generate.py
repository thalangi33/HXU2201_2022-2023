import cv2
import os
import numpy as np

images = []
scores = []

# Directory path of frames for image similartiy comparison
dataset_path = r'F:\dataset_path'

for file in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, file)
    images.append(cv2.imread(img_path))

for i in range(len(images) - 1):
    res = cv2.absdiff(images[i], images[i + 1])
    res = res.astype(np.uint8)
    np.count_nonzero(res)
    percentage = (np.count_nonzero(res) * 100) / res.size
    scores.append(percentage)

print(scores)