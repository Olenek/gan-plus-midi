import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.utils
from PIL import Image

from img2midi import image2midi
from utils import filter_image


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


torch.manual_seed(0)

Z_DIM = 100
NUM_CLASSES = 4
SAMPLE_SIZE = 32  # % NUM_CLASSES = 0

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = torch.load('saved/generator-l.pt')
generator.eval()
noise = torch.randn(SAMPLE_SIZE, Z_DIM, 1, 1).to(device)
labels = torch.Tensor(np.repeat(np.arange(0, NUM_CLASSES), SAMPLE_SIZE / NUM_CLASSES)).type(torch.LongTensor).to(device)

directory = 'generated-l/'
os.makedirs(directory, exist_ok=True)

# contrasts = [30, 50, 70]

with torch.no_grad():
    fake = generator(noise, labels)
    for i, image in enumerate(fake):
        subdir = os.path.join(directory, f'{labels[i].tolist()}')
        os.makedirs(subdir, exist_ok=True)
        img_path = os.path.join(subdir, f'{i % (SAMPLE_SIZE // NUM_CLASSES)}.png')
        torchvision.utils.save_image(image, img_path)

        with Image.open(img_path) as im:
            i = Image.fromarray(filter_image(im)).convert('RGB')
            i.save(img_path)

        image2midi(img_path)

# def test():
#     fs = FluidSynth()
#     fs.midi_to_audio('generated\\0\\0_30.mid',
#                      'out.wav')
#
# test()
