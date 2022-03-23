import imageio
import matplotlib.pyplot as plt

import torch 
from sampler import UniformSampler
from loader import BlenderLoader
from encoder import FrequencyEncoder
from decoder import NeRF
from renderer import VolumeRenderer
from utils import *

from tqdm import tqdm

def decode_and_render(decoder, renderer, x_encoded, dir_encoded, z_vals, dirs):
    output = decoder(x_encoded, dir_encoded)
    rgb_map, depth_map = renderer.render(output, z_vals, dirs)
    return rgb_map, depth_map

NUM_ITERS = 20000
BATCH_SIZE = 1024
NUM_SAMPLES = 128
CHUNK_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
L_x, L_dir = 10, 4
lrate = 5 * 1e-4
lrate_decay = 500

imgs_path = '/home/wzpscott/NeuralRendering/NeRFLib/data/nerf_synthetic/lego/train'
poses_path = '/home/wzpscott/NeuralRendering/NeRFLib/data/nerf_synthetic/lego/transforms_train.json'

loader = BlenderLoader(imgs_path, poses_path)
sampler = UniformSampler(2, 6)
encoder = FrequencyEncoder()
decoder = NeRF(x_dim=2*L_x*3+3, dir_dim=2*L_dir*3+3).to(DEVICE)
renderer = VolumeRenderer()

grad_vars = list(decoder.parameters())
optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))

for iter in tqdm(range(NUM_ITERS)):
    y, origins, dirs = loader.load_batch(BATCH_SIZE)
    y = y[...,:3] # rgba to rgb

    # move useful tensors to gpu
    y = y.to(DEVICE)
    origins = origins.to(DEVICE)
    dirs = dirs.to(DEVICE)

    x, z_vals = sampler.sample(origins, dirs, NUM_SAMPLES)
    x_encoded = encoder.encode(x, L_x)
    dir_encoded = encoder.encode(dirs, L_dir)
    dir_encoded = dir_encoded.unsqueeze(1).expand(BATCH_SIZE, NUM_SAMPLES, 2*L_dir*3+3)

    # rgb_map, depth_map = decode_and_render(decoder, renderer, x_encoded, dir_encoded, z_vals, dirs)
    rgb_map = torch.cat(
            [decode_and_render(decoder, renderer, \
                x_encoded[i:i+CHUNK_SIZE], dir_encoded[i:i+CHUNK_SIZE], \
                z_vals[i:i+CHUNK_SIZE], dirs[i:i+CHUNK_SIZE])[0] 
            for i in range(0, x.shape[0], CHUNK_SIZE)], dim=0)
            
    optimizer.zero_grad()
    img_loss = img2mse(rgb_map, y)
    loss = img_loss
    psnr = mse2psnr(img_loss.cpu())
    loss.backward()
    optimizer.step()

    ###  NOTE: IMPORTANT! update learning rate   ###
    decay_rate = 0.1
    decay_steps = lrate_decay * 1000
    new_lrate = lrate * (decay_rate ** (iter / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate

    if (iter+1)%50 == 0:
        tqdm.write(f'Iter: {iter+1} Loss: {loss.item()} PSNR: {psnr.item()}')

    if (iter+1) % 2000 == 0:
        y, origins, dirs = loader.load_image()
        y = y[...,:3] # rgba to rgb
        syn_img = []

        for j in range(0, y.shape[0], BATCH_SIZE):
            y_batch = y[j:j+BATCH_SIZE]
            origins_batch = origins[j:j+BATCH_SIZE]
            dirs_batch = dirs[j:j+BATCH_SIZE]

            # move useful tensors to gpu
            origins_batch = origins_batch.to(DEVICE)
            dirs_batch = dirs_batch.to(DEVICE)

            x, z_vals = sampler.sample(origins_batch, dirs_batch, NUM_SAMPLES)
            x_encoded = encoder.encode(x, L_x)
            dir_encoded = encoder.encode(dirs_batch, L_dir)
            dir_encoded = dir_encoded.unsqueeze(1).expand(y_batch.shape[0], NUM_SAMPLES, 2*L_dir*3+3)

            with torch.no_grad():
                CHUNK_SIZE = 128
                rgb_map = torch.cat(
                        [decode_and_render(decoder, renderer, \
                            x_encoded[i:i+CHUNK_SIZE], dir_encoded[i:i+CHUNK_SIZE], \
                            z_vals[i:i+CHUNK_SIZE], dirs_batch[i:i+CHUNK_SIZE])[0] 
                        for i in range(0, x.shape[0], CHUNK_SIZE)], dim=0)
            syn_img.append(rgb_map.cpu())
        syn_img = torch.cat(syn_img, dim=0).numpy()
        syn_img = syn_img.reshape(loader.H, loader.W, 3)
        syn_img = to8b(syn_img)
        gt_img = to8b(y.reshape(loader.H, loader.W, 3).numpy())

        plot([gt_img, syn_img], 1, 2, f'/home/wzpscott/NeuralRendering/NeRFLib/test/iter={iter+1}.png')