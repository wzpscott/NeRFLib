import torch
import numpy as np
import matplotlib.pyplot as plt


def img2mse(x, y):
    return torch.mean((x - y) ** 2)


def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def to8b(x):
    return (255*np.clip(x, 0, 1)).astype(np.uint8)


def plot(imgs, m, n, save_dir=''):
    for i, img in enumerate(imgs):
        plt.subplot(m, n, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(save_dir)
