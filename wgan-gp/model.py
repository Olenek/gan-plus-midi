"""
Discriminator and Generator implementation from DCGAN paper
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # input: N x (channels_img+1) x 128 x 128
            nn.Conv2d(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # shape: N x features_d x 64 x 64

            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            # shape: N x features_d*2 x 32 x 32

            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            # shape: N x features_d*4 x 16 x 16

            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # shape: N x features_d*8 x 8 x 8

            self._block(features_d * 8, features_d * 16, 4, 2, 1),
            # shape: N x features_d*16 x 4 x 4

            nn.Conv2d(in_channels=features_d * 16, out_channels=1, kernel_size=4, stride=1, padding=0)
            # shape: N x features_d*32 x 1 x 1
        )
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            LayerNorm(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            # Input: N x (channels_noise + embed_size) x 1 x 1
            self._block(channels_noise + embed_size, features_g * 16, 4, 1, 0),
            # shape: N x features_g*32 x 4 x 4

            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            # shape: N x features_g*16 x 8 x 8

            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 32x32
            # shape: N x features_g*16 x 16 x 16

            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 64x64
            # shape: N x features_g*16 x 32 x 32

            self._block(features_g * 2, features_g * 1, 4, 2, 1),  # img: 64x64
            # shape: N x features_g*16 x 64 x 64

            nn.ConvTranspose2d(
                features_g, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 128 x 128
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        # latent z: N x noise_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.net(x)


class LayerNorm(torch.nn.Module):
    """An implementation of Layer normalization that does not require size
    information. Copied from https://github.com/pytorch/pytorch/issues/1959."""

    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.n_features = n_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.Tensor(n_features).uniform_())
            self.beta = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 128, 128
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    labels = torch.randint(0, 4, (N,))
    disc = Discriminator(in_channels, 16, num_classes=4, img_size=H)
    assert disc(x, labels).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8, num_classes=4, img_size=H, embed_size=100)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z, labels).shape == (N, in_channels, H, W), "Generator test failed"

# test()
