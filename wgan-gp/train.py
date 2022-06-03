import os

import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator, Generator, initialize_weights
from settings import *
from utils import gradient_penalty

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

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

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

generator = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING).to(device)
discriminator = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES, IMG_SIZE).to(device)
initialize_weights(generator)
initialize_weights(discriminator)

# initializate optimizer
opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(0.0, 0.9))
opt_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_C, betas=(0.0, 0.9))

# for tensorboard plotting
fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)

os.makedirs('logs/fit-l2/', exist_ok=True)
writer_real = SummaryWriter('logs/fit-l2/real')
writer_fake = SummaryWriter('logs/fit-l2/fake')
writer_loss = SummaryWriter('logs/fit-l2/loss')

generator.train()
discriminator.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device)
        # print(labels)

        # Train Critic: max E[disc(real)] - E[disc(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = generator(noise, labels)

            disc_real = discriminator(real, labels).reshape(-1)
            disc_fake = discriminator(fake, labels).reshape(-1)
            gp = gradient_penalty(discriminator, labels, real, fake, device=device)

            loss_disc = (
                    -(torch.mean(disc_real) - torch.mean(disc_fake)) + LAMBDA_GP * gp
            )
            discriminator.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

        # Train Generator: max E[disc(gen_fake)] <-> min -E[disc(gen_fake)]
        gen_fake = discriminator(fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        generator.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx == len(loader) - 2 and (epoch+1) % 20 == 0:
            print(

                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch {batch_idx + 1}/{len(loader)} \
                  Loss C: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = generator(fixed_noise, labels)
                # take out (up to) 8 examples
                img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True, nrow=4)
                img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True, nrow=4)

                writer_real.add_image("Real", img_grid_real, global_step=epoch)
                writer_fake.add_image("Fake", img_grid_fake, global_step=epoch)

                writer_loss.add_scalar("Generator", loss_gen, global_step=epoch)
                writer_loss.add_scalar("Discriminator", loss_disc, global_step=epoch)

os.makedirs('saved/', exist_ok=True)

torch.save(generator, 'saved/generator.pt')
torch.save(discriminator, 'saved/critic.pt')
