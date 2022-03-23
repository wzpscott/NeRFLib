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

def batchify(fn, batchify_args:dict, other_args:dict, chunk_size, total_size):
    res = []
    for i in range(0, total_size, chunk_size):
        tmp_batchify_args = {k:v[i:i+chunk_size] for k,v in batchify_args.items()}
        tmp_res = fn(**tmp_batchify_args, **other_args)
        if i==0: # first loop, initialize res 
            for j in range(len(tmp_res)):
                res.append(tmp_res[j])
        else:
            for j in range(len(tmp_res)):
                res[j] = torch.cat([res[j], tmp_res[j]], dim=0)

    return res

def decode_and_render(decoder, renderer, x_encoded, dir_encoded, z_vals, dirs):
    output = decoder(x_encoded, dir_encoded)
    rgb_map, weights, depth_map = renderer.render(output, z_vals, dirs)
    return rgb_map, weights, depth_map