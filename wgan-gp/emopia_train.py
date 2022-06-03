import muspy
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator, Generator, initialize_weights
from settings import *
from utils import gradient_penalty

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


def emopia_transforms(music):
    t = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )

    music = muspy.adjust_resolution(music, target=4)
    mus_array = muspy.to_pianoroll_representation(music, encode_velocity=False, dtype=int) * 255
    length = mus_array.shape[0]
    to_trim = length - 128

    if to_trim < 0:
        x_tensor = t(np.concatenate((mus_array, np.zeros(-to_trim, 128)), axis=1).T)
    elif to_trim > 0 and to_trim % 2 == 0:
        x_tensor = t(mus_array[to_trim // 2:length - to_trim // 2].T)
    elif to_trim > 0:
        x_tensor = t(mus_array[to_trim // 2:length - to_trim // 2 - 1].T)
    else:
        x_tensor = t(mus_array.T)

    label = int(music.annotations[0].annotation['emo_class']) - 1  # subtracts one to get into 0:num_classes

    y_id = music.annotations[0].annotation['YouTube_ID']

    seg_id = music.annotations[0].annotation['seg_id']

    return x_tensor, label, y_id, seg_id


emopia = muspy.EMOPIADataset(EMOPIA_DIR)
emopia.convert()

dataset = emopia.to_pytorch_dataset(factory=emopia_transforms)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES, IMG_SIZE).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G)
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE_C)

# for tensorboard plotting
fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"../emopia_logs/fit128-3/real")
writer_fake = SummaryWriter(f"../emopia_logs/fit128-3/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels, y_id, seg_id) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = torch.as_tensor(labels)
        labels = labels.to(device)

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise, labels)

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
        if batch_idx == len(loader) - 2:
            print(

                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx + 1}/{len(loader)} \
                  Loss C: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise, labels)
                # take out (up to) 16 examples
                img_grid_real = torchvision.utils.make_grid(real[:16], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:16], normalize=True, value_range=(-1, 1))

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1

torch.save(gen, 'saved/generator_e3.pt')
torch.save(critic, 'saved/critic_e3.pt')
