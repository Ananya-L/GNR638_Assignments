import torch
import torchvision.transforms.functional as F
import random

def gaussian_noise(img, severity=0.1):
    noise = torch.randn_like(img) * severity
    return torch.clamp(img + noise, 0, 1)

def brightness_shift(img, severity=0.3):
    factor = 1 + random.uniform(-severity, severity)
    return torch.clamp(img * factor, 0, 1)

def motion_blur(img):
    kernel = torch.tensor([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [1,1,1,1,1],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ], dtype=torch.float32) / 5.0

    kernel = kernel.expand(img.shape[0],1,5,5)

    img = img.unsqueeze(0)

    return torch.nn.functional.conv2d(
        img,
        kernel,
        padding=2,
        groups=img.shape[1]
    ).squeeze(0)