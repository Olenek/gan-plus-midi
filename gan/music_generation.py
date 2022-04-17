import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.utils


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


Z_DIM = 100
NUM_CLASSES = 4
sample_size = 16  # % NUM_CLASSES = 0

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load('generator.pt')
model.eval()
noise = torch.randn(sample_size, Z_DIM, 1, 1).to(device)
labels = torch.Tensor(np.repeat(np.arange(0, NUM_CLASSES), sample_size / NUM_CLASSES)).type(torch.LongTensor).to(device)

directory = 'generated/'
os.makedirs(directory, exist_ok=True)

with torch.no_grad():
    fake = model(noise, labels)
    for i, image in enumerate(fake):
        torchvision.utils.save_image(image, f'generated/{labels[i]}-{i}_image.png')
