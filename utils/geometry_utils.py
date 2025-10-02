#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import torch
import torch.nn.functional as F

def project_gaussians_to_image(xyz, viewpoint):
    """
    Project 3D Gaussian positions to 2D image coordinates.
    
    Args:
        xyz: [N, 3] 3D positions
        viewpoint: Camera viewpoint
    
    Returns:
        pixels: [N, 2] pixel coordinates (x, y)
        depths: [N] depth values
        valid: [N] boolean mask for valid projections
    """
    device = xyz.device
    
    # Transform to camera space
    world2cam = viewpoint.world_view_transform.transpose(0, 1)  # [4, 4]
    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=device)], dim=1)  # [N, 4]
    xyz_cam = (world2cam @ xyz_homo.transpose(0, 1)).transpose(0, 1)  # [N, 4]
    
    # Extract camera space coordinates
    cam_pos = xyz_cam[:, :3]  # [N, 3]
    depths = cam_pos[:, 2]    # [N] - Z coordinate is depth
    
    # Project to screen space
    proj_matrix = viewpoint.projection_matrix.transpose(0, 1)  # [4, 4]
    screen_pos = (proj_matrix @ xyz_cam.transpose(0, 1)).transpose(0, 1)  # [N, 4]
    
    # Perspective divide
    screen_pos = screen_pos / (screen_pos[:, 3:4] + 1e-8)  # [N, 4]
    
    # Convert to pixel coordinates
    # Screen coordinates are in [-1, 1], convert to [0, W-1] and [0, H-1]
    pixels_x = (screen_pos[:, 0] + 1.0) * 0.5 * (viewpoint.image_width - 1)
    pixels_y = (screen_pos[:, 1] + 1.0) * 0.5 * (viewpoint.image_height - 1)
    pixels = torch.stack([pixels_x, pixels_y], dim=1)  # [N, 2]
    
    # Check valid projections (positive depth, within image bounds)
    valid_depth = depths > 0.1  # Minimum depth threshold
    valid_x = (pixels_x >= 0) & (pixels_x < viewpoint.image_width)
    valid_y = (pixels_y >= 0) & (pixels_y < viewpoint.image_height)
    valid = valid_depth & valid_x & valid_y
    
    return pixels, depths, valid

def sample_image_at_pixels(image, pixels):
    """
    Sample image colors at specified pixel locations using bilinear interpolation.
    
    Args:
        image: [3, H, W] image tensor
        pixels: [N, 2] pixel coordinates (x, y)
    
    Returns:
        colors: [N, 3] sampled RGB colors
    """
    device = image.device
    C, H, W = image.shape
    
    # Normalize coordinates to [-1, 1] for grid_sample
    x_norm = (pixels[:, 0] / (W - 1)) * 2.0 - 1.0
    y_norm = (pixels[:, 1] / (H - 1)) * 2.0 - 1.0
    
    # Grid sample expects [N, H, W, 2] format
    grid = torch.stack([x_norm, y_norm], dim=1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
    
    # Sample using bilinear interpolation
    image_batch = image.unsqueeze(0)  # [1, 3, H, W]
    sampled = F.grid_sample(
        image_batch, grid, mode='bilinear', padding_mode='border', align_corners=True
    )
    
    # Extract colors [1, 3, 1, N] -> [N, 3]
    colors = sampled.squeeze(0).squeeze(1).transpose(0, 1)
    
    return colors

def get_phase_schedule(iteration, max_iterations):
    """
    Compute training phase schedule based on iteration fraction.
    
    Phases:
    - Phase 0 (0-20%): Startup phase with aggressive densification
    - Phase 1 (20-40%): Early training with moderate floater filtering
    - Phase 2 (40-70%): Main training with balanced operations
    - Phase 3 (70-85%): Late training with conservative operations
    - Phase 4 (85-100%): Finalization with minimal changes
    
    Args:
        iteration: Current iteration
        max_iterations: Total training iterations
        
    Returns:
        phase: Current training phase (0-4)
        phase_progress: Progress within current phase (0.0-1.0)
        total_progress: Overall training progress (0.0-1.0)
    """
    total_progress = min(iteration / max_iterations, 1.0)
    
    # Define phase boundaries
    phase_boundaries = [0.0, 0.2, 0.4, 0.7, 0.85, 1.0]
    
    # Find current phase
    phase = 0
    for i in range(len(phase_boundaries) - 1):
        if total_progress >= phase_boundaries[i] and total_progress < phase_boundaries[i + 1]:
            phase = i
            break
    else:
        phase = len(phase_boundaries) - 2  # Last phase
    
    # Compute progress within current phase
    phase_start = phase_boundaries[phase]
    phase_end = phase_boundaries[phase + 1]
    phase_progress = (total_progress - phase_start) / (phase_end - phase_start)
    phase_progress = max(0.0, min(1.0, phase_progress))
    
    return phase, phase_progress, total_progress