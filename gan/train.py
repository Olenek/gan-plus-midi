"""
Training of WGAN-GP
"""
import os

import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator, Generator, initialize_weights
from utils import gradient_penalty

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
LEARNING_RATE_G = 1e-4
LEARNING_RATE_C = 1e-4
BATCH_SIZE = 32
IMG_SIZE = 128
CHANNELS_IMG = 1
NUM_CLASSES = 4
GEN_EMBEDDING = 128
Z_DIM = 128
NUM_EPOCHS = 1000
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
DATA_DIR = '../data_engineering/midi-images'

transforms = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)

dataset = datasets.ImageFolder(DATA_DIR, transform=transforms)
print(dataset.find_classes(DATA_DIR))
# comment mnist above and uncomment below for training on CelebA
# dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES, IMG_SIZE).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G)
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE_C)

# for tensorboard plotting
fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"../logs/fit128-4/real")
writer_fake = SummaryWriter(f"../logs/fit128-4/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device)

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise, labels)

            # critic_noise = 0.03*torch.randn(cur_batch_size, 1, IMG_SIZE, IMG_SIZE).to(device)

            # critic_real = critic(real + critic_noise, labels).reshape(-1)
            # critic_fake = critic(fake + critic_noise, labels).reshape(-1)
            # gp = gradient_penalty(critic, labels, real + critic_noise, fake + critic_noise, device=device)

            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            gp = gradient_penalty(critic, labels, real, fake, device=device)

            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx == len(loader)-2:
            print(

                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx+1}/{len(loader)} \
                  Loss C: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise, labels)
                # take out (up to) 16 examples
                img_grid_real = torchvision.utils.make_grid(real[:16], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:16], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1

os.makedirs('saved/', exist_ok=True)

torch.save(gen, 'saved/generator4.pt')
torch.save(critic, 'saved/critic4.pt')
