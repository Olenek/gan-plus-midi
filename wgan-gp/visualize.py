import numpy as np
import torch
from torchviz import make_dot

from model import Generator, Discriminator
from settings import *

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES, IMG_SIZE)

labels = torch.Tensor(np.repeat(np.arange(0, NUM_CLASSES), SAMPLE_SIZE / NUM_CLASSES)).type(torch.LongTensor)
noise = torch.randn(SAMPLE_SIZE, Z_DIM, 1, 1)
y = gen(noise, labels)

make_dot(y.mean(), params=dict(gen.named_parameters())).render("attached", format="svg")
