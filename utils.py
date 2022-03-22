import torch
import numpy as np
import matplotlib.pyplot as plt
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def plot(imgs, m, n, save_dir=''):
    for i, img in enumerate(imgs):
        plt.subplot(m,n,i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(save_dir)
