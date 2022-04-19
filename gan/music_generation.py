import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.utils
from img2midi import image2midi
import model
from midi2audio import FluidSynth


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
sample_size = 16  # % NUM_CLASSES = 0

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = torch.load('saved/generator.pt')
generator.eval()
noise = torch.randn(sample_size, Z_DIM, 1, 1).to(device)
labels = torch.Tensor(np.repeat(np.arange(0, NUM_CLASSES), sample_size / NUM_CLASSES)).type(torch.LongTensor).to(device)

directory = 'generated/'
os.makedirs(directory, exist_ok=True)

contrasts = [30, 50, 70, 80, 90]
fs = FluidSynth()

with torch.no_grad():
    fake = generator(noise, labels)
    for i, image in enumerate(fake):
        os.makedirs(f'generated/{labels[i]}', exist_ok=True)
        torchvision.utils.save_image(image, f'generated/{labels[i]}/{i%4}.png')
        for percentage in contrasts:
            image2midi(f'generated/{labels[i]}/{i%4}.png', contrast_percentage=percentage)
            # fs.midi_to_audio(f'generated/{labels[i]}/{i%4}_{percentage}.mid', f'generated/{labels[i]}/{i%4}_{percentage}.wav')


# def test():
#     fs = FluidSynth()
#     fs.midi_to_audio('generated\\0\\0_30.mid',
#                      'out.wav')
#
# test()
