import numpy as np
import torch


class UniformSampler():
    def __init__(self, near, far):
        self.near = near
        self.far = far
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def sample(self, origins, dirs, num_samples, random=False):
        t_vals = torch.linspace(0., 1., steps=num_samples).to(self.device)
        z_vals = self.near * (1.-t_vals) + self.far * (t_vals)
        z_vals = z_vals.unsqueeze(0)
        z_vals = z_vals.expand([origins.shape[0], num_samples])

        if random:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            z_vals = lower + (upper - lower) * t_rand

        pts = origins[..., None, :] + dirs[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples, 3]
        return pts, z_vals
