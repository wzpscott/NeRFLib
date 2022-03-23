import numpy as np
import torch


class UniformSampler():
    def __init__(self, near, far, num_samples):
        self.near = near
        self.far = far
        self.num_samples = num_samples
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def sample(self, origins, dirs, random=False):
        t_vals = torch.linspace(0., 1., steps=self.num_samples).to(self.device)
        z_vals = self.near * (1.-t_vals) + self.far * (t_vals)
        z_vals = z_vals.unsqueeze(0)
        z_vals = z_vals.expand([origins.shape[0], self.num_samples])

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

class ImportanceSampler():
    def __init__(self, near, far, num_samples):
        self.near = near
        self.far = far
        self.num_samples = num_samples
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def sample(self, origins, dirs, z_vals, weights, include_uniform_samples=False):
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = self.sample_pdf(z_vals_mid, weights[...,1:-1], self.num_samples)
        z_samples = z_samples.detach()
        if include_uniform_samples:
            z_samples, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        pts = origins[...,None,:] + dirs[...,None,:] * z_samples[...,:,None]
        return pts, z_samples

    def sample_pdf(self, bins, weights, N_samples):
        # Get pdf
        weights = weights + 1e-5 # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(self.device)

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

        return samples