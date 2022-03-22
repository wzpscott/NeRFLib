import os
import os.path as osp
import imageio 
import json
import numpy as np
import torch
import torch.nn.functional as F

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

class BlenderLoader():
    def __init__(self, imgs_path, poses_path):
        imgs = []
        poses = []
        with open(poses_path, 'r') as fp:
            poses_raw = json.load(fp)

        for pose_raw in poses_raw['frames']:
            pose = np.array(pose_raw["transform_matrix"])
            poses.append(pose)

        for img_path in os.listdir(imgs_path):
            img = imageio.imread(osp.join(imgs_path, img_path))
            img = (np.array(img) / 255.).astype(np.float32)
            imgs.append(img)

        imgs = np.stack(imgs, 0)
        poses = np.stack(poses, 0)   
        imgs = torch.FloatTensor(imgs)
        poses = torch.FloatTensor(poses)

        H, W = imgs[0].shape[:2]
        camera_angle_x = float(poses_raw['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        K = torch.Tensor([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

        origins = []
        dirs = []
        for pose in poses:
            rays_o, rays_d = get_rays(H, W, K, pose)
            origins.append(rays_o)
            dirs.append(rays_d)

        origins = torch.stack(origins,0)
        dirs = torch.stack(dirs,0)
        dirs = F.normalize(dirs, p=2, dim=-1) # united vectors as directions
    
        self.colors = imgs.reshape(-1, 4) # [num rays, 4] rgba color
        self.origins = origins.reshape(-1, 3) # [num rays, 3] (x0,y0,z0) origins of rays
        self.dirs = dirs.reshape(-1, 3) # [num rays, 3] directions of rays(united vectors)
        
        self.H = H
        self.W = W
        self.num_rays = self.colors.shape[0]
    def load_batch(self, batch_size):
        sampled_idxs = np.random.randint(0, self.num_rays, batch_size)
        return self.colors[sampled_idxs], self.origins[sampled_idxs], self.dirs[sampled_idxs]
    def load_image(self, idx=0):
        H, W = self.H, self.W
        sampled_idxs = range(idx*H*W,(idx+1)*H*W)
        return self.colors[sampled_idxs], self.origins[sampled_idxs], self.dirs[sampled_idxs]
if __name__ == '__main__':
    imgs_path = '/home/wzpscott/NeuralRendering/data/nerf_synthetic/lego/train'
    poses_path = '/home/wzpscott/NeuralRendering/data/nerf_synthetic/lego/transforms_train.json'
    loader = BlenderLoader(imgs_path, poses_path)
    _, _, dirs = loader.load_batch(10)
    


