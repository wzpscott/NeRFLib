import torch
import torch.nn.functional as F

class VolumeRenderer():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    @staticmethod
    def alpha_activate_fn(alpha, dists):
        return 1.-torch.exp(-F.relu(alpha)*dists)

    def render(self, raw, z_vals, rays_d):
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat(
            [dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(self.device)], 
            dim=-1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        alpha, rgb = raw[...,-1], raw[...,:3]
        alpha = self.alpha_activate_fn(alpha, dists)
        rgb = torch.sigmoid(rgb)

        # print(alpha.shape, torch.cumprod(
            # torch.cat([torch.ones((alpha.shape[0], 1)).to(self.device), 1.-alpha + 1e-10], -1), -1)[:, :-1].shape)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(self.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        return rgb_map, depth_map
        
        