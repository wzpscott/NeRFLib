import imageio
import matplotlib.pyplot as plt

import torch 
from sampler import ImportanceSampler, UniformSampler
from loader import BlenderLoader
from encoder import FrequencyEncoder
from decoder import NeRFDecoder
from renderer import VolumeRenderer
from utils import *

from tqdm import tqdm

DEBUG = True
NUM_ITERS = 10000
BATCH_SIZE = 1024
NUM_SAMPLES_UNIFORM = 64
NUM_SAMPLES_IMPORTANCE = 64
CHUNK_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
L_x, L_dir = 10, 4
lrate = 5 * 1e-4
lrate_decay = 500
decay_rate = 0.1
decay_steps = lrate_decay * 1000

imgs_path = '/home/wzpscott/NeuralRendering/NeRFLib/data/nerf_synthetic/lego/train'
poses_path = '/home/wzpscott/NeuralRendering/NeRFLib/data/nerf_synthetic/lego/transforms_train.json'
if DEBUG:
    imgs_path = '/home/wzpscott/NeuralRendering/NeRFLib/data/nerf_synthetic/toy_lego/train'
    poses_path = '/home/wzpscott/NeuralRendering/NeRFLib/data/nerf_synthetic/toy_lego/transforms_train.json'
loader = BlenderLoader(imgs_path, poses_path, debug=DEBUG)

uniform_sampler = UniformSampler(2, 6, NUM_SAMPLES_UNIFORM)
importance_sampler = ImportanceSampler(2, 6, NUM_SAMPLES_IMPORTANCE)
encoder = FrequencyEncoder()
decoder = NeRFDecoder(x_dim=2*L_x*3+3, dir_dim=2*L_dir*3+3).to(DEVICE)
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

    x, z_vals = uniform_sampler.sample(origins, dirs)
    x_encoded = encoder.encode(x, L_x)
    dir_encoded = encoder.encode(dirs, L_dir)
    dir_encoded = dir_encoded.unsqueeze(1).expand(BATCH_SIZE, NUM_SAMPLES_UNIFORM, 2*L_dir*3+3)

    # batchify version of:
    # rgb_map, weights, depth_map = decode_and_render(decoder, renderer, x_encoded, dir_encoded, z_vals, dirs)
    rgb_map, weights, depth_map = batchify(
        fn=decode_and_render,
        batchify_args={
            'x_encoded':x_encoded, 'dir_encoded':dir_encoded, 'z_vals':z_vals, 'dirs':dirs
        },
        other_args={
            'decoder':decoder, 'renderer':renderer,
        },
        chunk_size=CHUNK_SIZE,
        total_size=x_encoded.shape[0]
    )

    if iter > 1000:
        x, z_vals = importance_sampler.sample(origins, dirs, z_vals, weights, include_uniform_samples=True)
        
        x_encoded = encoder.encode(x, L_x)
        dir_encoded = encoder.encode(dirs, L_dir)
        dir_encoded = dir_encoded.unsqueeze(1).expand(BATCH_SIZE, NUM_SAMPLES_UNIFORM+NUM_SAMPLES_IMPORTANCE, 2*L_dir*3+3)

        # batchify version of:
        # rgb_map, weights, depth_map = decode_and_render(decoder, renderer, x_encoded, dir_encoded, z_vals, dirs)
        rgb_map, weights, depth_map = batchify(
            fn=decode_and_render,
            batchify_args={
                'x_encoded':x_encoded, 'dir_encoded':dir_encoded, 'z_vals':z_vals, 'dirs':dirs
            },
            other_args={
                'decoder':decoder, 'renderer':renderer,
            },
            chunk_size=CHUNK_SIZE,
            total_size=x_encoded.shape[0]
        )

    img_loss = img2mse(rgb_map, y)
    loss = img_loss
    psnr = mse2psnr(img_loss.cpu())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ###  NOTE: IMPORTANT! update learning rate   ###
    new_lrate = lrate * (decay_rate ** (iter / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate

    if (iter+1)%50 == 0:

        tqdm.write(f'Iter: {iter+1} Loss: {loss.item()} PSNR: {psnr.item()} z_var:{torch.var(z_vals).item()}')

    if (iter+1) % 500 == 0:
        y, origins, dirs = loader.load_image()
        y = y[...,:3] # rgba to rgb
        syn_rgb_img = []
        syn_depth_img = []

        for j in range(0, y.shape[0], BATCH_SIZE):
            y_batch = y[j:j+BATCH_SIZE]
            origins_batch = origins[j:j+BATCH_SIZE]
            dirs_batch = dirs[j:j+BATCH_SIZE]

            # move useful tensors to gpu
            origins_batch = origins_batch.to(DEVICE)
            dirs_batch = dirs_batch.to(DEVICE)

            x, z_vals = uniform_sampler.sample(origins_batch, dirs_batch)
            x_encoded = encoder.encode(x, L_x)
            dir_encoded = encoder.encode(dirs_batch, L_dir)
            dir_encoded = dir_encoded.unsqueeze(1).expand(y_batch.shape[0], NUM_SAMPLES_UNIFORM, 2*L_dir*3+3)

            with torch.no_grad():
                # print(x_encoded.device, dir_encoded.device, z_vals.device)
                rgb_map, weights, depth_map = batchify(
                    fn=decode_and_render,
                    batchify_args={
                        'x_encoded':x_encoded, 'dir_encoded':dir_encoded, 'z_vals':z_vals, 'dirs':dirs_batch
                    },
                    other_args={
                        'decoder':decoder, 'renderer':renderer,
                    },
                    chunk_size=CHUNK_SIZE,
                    total_size=x_encoded.shape[0]
                )
            syn_rgb_img.append(rgb_map.cpu())
            syn_depth_img.append(depth_map.cpu())
        syn_rgb_img = torch.cat(syn_rgb_img, dim=0).numpy()
        syn_rgb_img = syn_rgb_img.reshape(loader.H, loader.W, 3)
        syn_rgb_img = to8b(syn_rgb_img)

        syn_depth_img = torch.cat(syn_depth_img, dim=0).numpy()
        syn_depth_img = syn_depth_img.reshape(loader.H, loader.W)
        syn_depth_img = to8b(syn_depth_img)

        gt_img = to8b(y.reshape(loader.H, loader.W, 3).numpy())

        plot([gt_img, syn_rgb_img, syn_depth_img], 1, 3, f'/home/wzpscott/NeuralRendering/NeRFLib/test/iter={iter+1}.png')