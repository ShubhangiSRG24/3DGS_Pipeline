#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

def compute_isolation_loss(gaussians, k=5, opacity_threshold=0.1):
    """
    Compute isolation loss: L_iso = mean(alpha * knn_mean_distance)
    Penalizes opaque isolated points to encourage spatial coherence.
    
    Args:
        gaussians: GaussianModel instance
        k: Number of nearest neighbors for KNN computation
        opacity_threshold: Minimum opacity to be considered for isolation loss
    
    Returns:
        isolation_loss: Scalar isolation loss value
    """
    from utils.knn_utils import batched_knn_distances
    
    xyz = gaussians.get_xyz
    opacity = gaussians.get_opacity.squeeze()
    device = xyz.device
    
    # Filter points above opacity threshold
    opaque_mask = opacity > opacity_threshold
    if opaque_mask.sum() <= k:
        # Not enough opaque points for meaningful KNN
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    opaque_xyz = xyz[opaque_mask]
    opaque_alpha = opacity[opaque_mask]
    
    # Use batched KNN with caching for efficiency
    use_sampling = opaque_xyz.shape[0] > 50000
    knn_distances, _, valid_mask = batched_knn_distances(
        opaque_xyz, k, chunk_size=4096, use_sampling=use_sampling, sample_ratio=0.5
    )
    
    # Filter opacities for valid points when using sampling
    if use_sampling:
        opaque_alpha = opaque_alpha[valid_mask[opaque_mask]]
    
    # Compute mean distance to k nearest neighbors for each point
    knn_mean_distances = knn_distances.mean(dim=1)
    
    # Isolation loss: weight by opacity (more opaque = higher penalty)
    isolation_loss = (opaque_alpha * knn_mean_distances).mean()
    
    return isolation_loss

def compute_sh_smoothness_loss(gaussians, k=8, dc_weight=1.0, ac_weight=0.1):
    """
    Compute SH smoothness loss: L_sh = TV(DC) + small * TV(AC)
    Uses local neighborhood L2 regularization to prevent spiky colors.
    
    Args:
        gaussians: GaussianModel instance
        k: Number of nearest neighbors for local smoothness
        dc_weight: Weight for DC component smoothness
        ac_weight: Weight for AC component smoothness (much smaller)
    
    Returns:
        sh_smoothness_loss: Scalar smoothness loss value
    """
    from utils.knn_utils import batched_knn_distances
    
    xyz = gaussians.get_xyz
    features_dc = gaussians._features_dc  # [N, 1, 3] for RGB
    features_rest = gaussians._features_rest  # [N, (sh_degree+1)^2-1, 3] if exists
    device = xyz.device
    
    if xyz.shape[0] <= k:
        # Not enough points for meaningful neighborhood
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Use batched KNN with caching for efficiency
    use_sampling = xyz.shape[0] > 50000
    knn_distances, knn_indices, valid_mask = batched_knn_distances(
        xyz, k, chunk_size=4096, use_sampling=use_sampling, sample_ratio=0.7
    )
    
    # If using sampling, filter features to valid points
    if use_sampling:
        features_dc = features_dc[valid_mask]
        if features_rest is not None:
            features_rest = features_rest[valid_mask]
    
    # Compute DC smoothness loss
    dc_loss = torch.tensor(0.0, device=device, requires_grad=True)
    if dc_weight > 0:
        # Get DC colors for each point [N, 3]
        dc_colors = features_dc.squeeze(1)  # Remove middle dimension
        
        # Get neighbor DC colors [N, k, 3]
        neighbor_dc = dc_colors[knn_indices]
        
        # Compute L2 differences with neighbors
        dc_center = dc_colors.unsqueeze(1)  # [N, 1, 3]
        dc_diffs = torch.norm(neighbor_dc - dc_center, dim=2)  # [N, k]
        
        # Weight by inverse distance to give closer neighbors more influence
        distance_weights = 1.0 / (knn_distances + 1e-6)  # [N, k]
        distance_weights = distance_weights / distance_weights.sum(dim=1, keepdim=True)
        
        # Weighted smoothness loss
        dc_loss = (dc_diffs * distance_weights).sum(dim=1).mean() * dc_weight
    
    # Compute AC smoothness loss (if AC components exist)
    ac_loss = torch.tensor(0.0, device=device, requires_grad=True)
    if ac_weight > 0 and features_rest is not None and features_rest.shape[1] > 0:
        # Flatten AC features for easier processing [N, num_ac_features * 3]
        ac_colors = features_rest.view(features_rest.shape[0], -1)
        
        # Get neighbor AC colors [N, k, num_features]
        neighbor_ac = ac_colors[knn_indices]
        
        # Compute L2 differences with neighbors
        ac_center = ac_colors.unsqueeze(1)  # [N, 1, num_features]
        ac_diffs = torch.norm(neighbor_ac - ac_center, dim=2)  # [N, k]
        
        # Weight by inverse distance (same weights as DC)
        distance_weights = 1.0 / (knn_distances + 1e-6)
        distance_weights = distance_weights / distance_weights.sum(dim=1, keepdim=True)
        
        # Weighted AC smoothness loss (much smaller weight)
        ac_loss = (ac_diffs * distance_weights).sum(dim=1).mean() * ac_weight
    
    # Total SH smoothness loss
    sh_smoothness_loss = dc_loss + ac_loss
    
    return sh_smoothness_loss

def compute_stochastic_isolation_loss(gaussians, k=5, opacity_threshold=0.1, 
                                    sampling_ratio=0.1, ramp_factor=1.0):
    """
    Compute isolation loss on a random subset of points for efficiency.
    
    Args:
        gaussians: GaussianModel instance
        k: Number of nearest neighbors for KNN computation
        opacity_threshold: Minimum opacity to be considered for isolation loss
        sampling_ratio: Fraction of points to sample (0.05-0.15)
        ramp_factor: Multiplicative factor for progressive ramping (0.1-1.0)
        
    Returns:
        isolation_loss: Scalar isolation loss value
    """
    from utils.knn_utils import batched_knn_distances
    
    xyz = gaussians.get_xyz
    opacity = gaussians.get_opacity.squeeze()
    device = xyz.device
    n_total = xyz.shape[0]
    
    # Stochastic sampling: select random subset of points
    n_sample = max(100, int(n_total * sampling_ratio))  # At least 100 points
    if n_sample >= n_total:
        # Use all points if sample size exceeds total
        sample_indices = torch.arange(n_total, device=device)
    else:
        sample_indices = torch.randperm(n_total, device=device)[:n_sample]
    
    sampled_xyz = xyz[sample_indices]
    sampled_opacity = opacity[sample_indices]
    
    # Filter points above opacity threshold from sample
    opaque_mask = sampled_opacity > opacity_threshold
    if opaque_mask.sum() <= k:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    opaque_xyz = sampled_xyz[opaque_mask]
    opaque_alpha = sampled_opacity[opaque_mask]
    
    # Use batched KNN with caching for efficiency
    use_sampling = opaque_xyz.shape[0] > 10000  # Lower threshold for subset
    knn_distances, _, valid_mask = batched_knn_distances(
        opaque_xyz, k, chunk_size=2048, use_sampling=use_sampling, sample_ratio=0.7
    )
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute mean distances for valid points
    mean_distances = torch.mean(knn_distances[valid_mask], dim=1)
    valid_alphas = opaque_alpha[valid_mask]
    
    # Isolation loss: opacity-weighted mean distance
    isolation_loss = torch.mean(valid_alphas * mean_distances)
    
    # Scale by sampling ratio to approximate full loss and apply ramping
    scaling_factor = (1.0 / sampling_ratio) * ramp_factor
    
    return isolation_loss * scaling_factor

def compute_stochastic_sh_smoothness_loss(gaussians, k=8, dc_weight=1.0, ac_weight=0.1,
                                        sampling_ratio=0.1, ramp_factor=1.0):
    """
    Compute SH smoothness loss on a random subset of points for efficiency.
    
    Args:
        gaussians: GaussianModel instance
        k: Number of nearest neighbors for SH comparison
        dc_weight: Weight for DC component smoothness
        ac_weight: Weight for AC component smoothness  
        sampling_ratio: Fraction of points to sample
        ramp_factor: Progressive ramping factor
        
    Returns:
        sh_smoothness_loss: Scalar SH smoothness loss value
    """
    from utils.knn_utils import batched_knn_distances
    
    xyz = gaussians.get_xyz
    features = gaussians._features_dc
    features_rest = gaussians._features_rest
    device = xyz.device
    n_total = xyz.shape[0]
    
    # Stochastic sampling
    n_sample = max(100, int(n_total * sampling_ratio))
    if n_sample >= n_total:
        sample_indices = torch.arange(n_total, device=device)
    else:
        sample_indices = torch.randperm(n_total, device=device)[:n_sample]
    
    sampled_xyz = xyz[sample_indices]
    sampled_dc = features[sample_indices]  # [N_sample, 1, 3]
    sampled_rest = features_rest[sample_indices] if features_rest.shape[0] > 0 else None
    
    if sampled_xyz.shape[0] <= k:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute KNN for sampled points
    knn_distances, knn_indices, valid_mask = batched_knn_distances(
        sampled_xyz, k, chunk_size=2048, use_sampling=False
    )
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # DC component smoothness - simplified for efficiency
    dc_loss = torch.tensor(0.0, device=device, requires_grad=True)
    if dc_weight > 0:
        valid_indices = torch.where(valid_mask)[0][:1000]  # Limit for efficiency
        
        if len(valid_indices) > 0:
            # Simplified DC difference computation
            batch_dc_diffs = []
            for idx in valid_indices[:min(len(valid_indices), 100)]:  # Further limit
                center_dc = sampled_dc[idx]  # [1, 3]
                neighbor_indices = knn_indices[idx]
                valid_neighbors = neighbor_indices[neighbor_indices < n_sample]
                
                if len(valid_neighbors) > 1:
                    neighbor_dc = sampled_dc[valid_neighbors]  # [K_valid, 1, 3]
                    dc_diff = torch.mean(torch.norm(center_dc - neighbor_dc, dim=2))
                    batch_dc_diffs.append(dc_diff)
            
            if batch_dc_diffs:
                dc_loss = torch.mean(torch.stack(batch_dc_diffs))
    
    # Skip AC component for efficiency
    ac_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    total_loss = dc_weight * dc_loss + ac_weight * ac_loss
    scaling_factor = (1.0 / sampling_ratio) * ramp_factor
    
    return total_loss * scaling_factor
