import numpy as np
import torch
from PIL import ImageOps


def gradient_penalty(critic, labels, real, fake, device="cpu"):
    batch_size, c, H, W = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, labels)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    _gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return _gradient_penalty


def filter_image(image):
    _img = ImageOps.grayscale(image)
    arr = np.asarray(_img)
    threshold = np.max(arr) * 0.2

    good_pixels = arr >= threshold

    return np.where(good_pixels, 255, 0)
