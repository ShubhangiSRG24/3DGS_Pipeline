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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, importance_scores = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Early debug logging
    if not hasattr(pc, '_early_debug_logged'):
        print(f"EARLY DEBUG: pc.get_xyz.shape: {pc.get_xyz.shape}")
        print(f"EARLY DEBUG: pc._xyz.shape: {pc._xyz.shape}")
        print(f"EARLY DEBUG: pc._features_dc.shape: {pc._features_dc.shape}")
        print(f"EARLY DEBUG: pc._features_rest.shape: {pc._features_rest.shape}")
        pc._early_debug_logged = True
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    try:
        xyz_tensor = pc.get_xyz
        print(f"DEBUG: xyz_tensor shape: {xyz_tensor.shape}")
        screenspace_points = torch.zeros_like(xyz_tensor, dtype=xyz_tensor.dtype, requires_grad=True, device="cuda") + 0
        print(f"DEBUG: screenspace_points shape: {screenspace_points.shape}")
    except Exception as e:
        print(f"ERROR in screenspace_points creation: {e}")
        print(f"DEBUG: pc._xyz.shape: {pc._xyz.shape}")
        print(f"DEBUG: pc._features_dc.shape: {pc._features_dc.shape}")
        print(f"DEBUG: pc._features_rest.shape: {pc._features_rest.shape}")
        raise e
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

        
# --- NOVELTY: APPLY LEARNED IMPORTANCE MASK ---
    if importance_scores is not None:
        opacity = opacity * importance_scores
    # --- END NOVELTY ---


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Debug tensor shapes before passing to rasterizer
    if not hasattr(pc, '_rasterizer_debug_logged'):
        print(f"RASTERIZER DEBUG: means3D shape: {means3D.shape}")
        print(f"RASTERIZER DEBUG: means2D shape: {means2D.shape}")
        print(f"RASTERIZER DEBUG: opacity shape: {opacity.shape}")
        if scales is not None:
            print(f"RASTERIZER DEBUG: scales shape: {scales.shape}")
        if rotations is not None:
            print(f"RASTERIZER DEBUG: rotations shape: {rotations.shape}")
        print(f"RASTERIZER DEBUG: shs shape: {shs.shape if shs is not None else 'None'}")
        pc._rasterizer_debug_logged = True

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
