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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from .knn_cache import KNNCache
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch.nn.functional as F

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self._exposure = None  # Default exposure, will be properly initialized in create_from_pcd
        self.spatial_lr_scale = 0
        
        # Feature vector computation
        self._feature_vectors = None
        self._last_refine_iteration = -1
        self._refine_interval = 100  # Will be set from training args
        self._visibility_count = torch.empty(0)
        self._grad_ema_pos = torch.empty(0)
        self._grad_ema_scale = torch.empty(0) 
        self._grad_ema_rot = torch.empty(0)
        
        # Intelligent kNN caching system
        self._knn_cache = KNNCache(max_age=3, change_threshold=0.05, point_count_threshold=0.1)
        self._recent_camera_positions = []  # Track camera positions for cache invalidation
        
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        scaling = self.scaling_activation(self._scaling)
        # Ensure scaling size matches other tensors
        features_size = min(self._features_dc.shape[0], self._features_rest.shape[0])
        if scaling.shape[0] != features_size:
            scaling = scaling[:features_size]
        return scaling

    @property
    def get_rotation(self):
        rotation = self.rotation_activation(self._rotation)
        # Ensure rotation size matches other tensors
        features_size = min(self._features_dc.shape[0], self._features_rest.shape[0])
        if rotation.shape[0] != features_size:
            rotation = rotation[:features_size]
        return rotation
    
    @property
    def get_xyz(self):
        xyz = self._xyz
        # Ensure xyz has the correct shape [num_points, 3] and matches feature tensor sizes
        features_size = min(self._features_dc.shape[0], self._features_rest.shape[0])
        if xyz.shape[0] != features_size:
            if not hasattr(self, '_xyz_size_warning_logged'):
                print(f"WARNING: XYZ size mismatch - XYZ: {xyz.shape[0]}, Features: {features_size}, truncating XYZ")
                self._xyz_size_warning_logged = True
            xyz = xyz[:features_size]
        return xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest

        # Ensure matching number of Gaussians
        if features_dc.shape[0] != features_rest.shape[0]:
            min_gaussians = min(features_dc.shape[0], features_rest.shape[0])
            if not hasattr(self, '_size_warning_logged'):
                print(f"WARNING: Feature size mismatch - DC: {features_dc.shape[0]}, REST: {features_rest.shape[0]}, using {min_gaussians}")
                self._size_warning_logged = True
            features_dc = features_dc[:min_gaussians]
            features_rest = features_rest[:min_gaussians]

        # Debug: print the actual error case
        if not hasattr(self, '_debug_shapes_logged'):
            print(f"DETAILED DEBUG: DC shape: {features_dc.shape}, REST shape: {features_rest.shape}")
            self._debug_shapes_logged = True

        # Handle the standard spherical harmonics case
        # DC: [N, 1, 3] and REST: [N, 15, 3] should concatenate to [N, 16, 3]
        if (len(features_dc.shape) == 3 and len(features_rest.shape) == 3):
            # Check if all dimensions except the middle one match
            if (features_dc.shape[0] == features_rest.shape[0] and
                features_dc.shape[2] == features_rest.shape[2]):
                # This is the standard case: concatenate along the SH coefficient dimension
                return torch.cat((features_dc, features_rest), dim=1)
            else:
                print(f"Shape mismatch in SH tensors: DC {features_dc.shape}, REST {features_rest.shape}")

        # If we get here, we need to handle non-standard shapes
        # But we must maintain the 3D structure for the renderer

        # Fallback: ensure both tensors have compatible shapes
        if len(features_dc.shape) == 3 and len(features_rest.shape) == 3:
            # Force them to have the same last dimension by truncating
            min_last_dim = min(features_dc.shape[2], features_rest.shape[2])
            features_dc = features_dc[:, :, :min_last_dim]
            features_rest = features_rest[:, :, :min_last_dim]
            return torch.cat((features_dc, features_rest), dim=1)

        # Last resort: flatten and then reshape back to 3D
        features_dc_flat = features_dc.view(features_dc.shape[0], -1)
        features_rest_flat = features_rest.view(features_rest.shape[0], -1)
        combined_flat = torch.cat((features_dc_flat, features_rest_flat), dim=1)

        # Reshape back to 3D format [N, features, 3] for the renderer
        # Assume 3 color channels (RGB)
        num_features = combined_flat.shape[1] // 3
        return combined_flat.view(features_dc.shape[0], num_features, 3)
    
    @property
    def get_features_dc(self):
        features_dc = self._features_dc
        # Ensure DC features have consistent size with REST features
        min_gaussians = min(features_dc.shape[0], self._features_rest.shape[0])
        if features_dc.shape[0] != min_gaussians:
            features_dc = features_dc[:min_gaussians]
        return features_dc

    @property
    def get_features_rest(self):
        features_rest = self._features_rest
        # Ensure REST features have consistent size with DC features
        min_gaussians = min(features_rest.shape[0], self._features_dc.shape[0])
        if features_rest.shape[0] != min_gaussians:
            features_rest = features_rest[:min_gaussians]
        return features_rest
    
    @property
    def get_opacity(self):
        opacity = self.opacity_activation(self._opacity)
        # Ensure opacity size matches other tensors
        features_size = min(self._features_dc.shape[0], self._features_rest.shape[0])
        if opacity.shape[0] != features_size:
            opacity = opacity[:features_size]
        return opacity
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        
        # Initialize feature computation variables
        self._visibility_count = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._grad_ema_pos = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._grad_ema_scale = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self._grad_ema_rot = torch.zeros((self.get_xyz.shape[0], 4), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._refine_interval = training_args.densification_interval

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        if self._exposure is not None:
            self.exposure_optimizer = torch.optim.Adam([self._exposure])
        else:
            self.exposure_optimizer = None

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if optimizable_tensors and "opacity" in optimizable_tensors:
            self._opacity = optimizable_tensors["opacity"]
        else:
            print("[ERROR] Failed to replace opacity tensor in optimizer")
            # Fallback: directly set the tensor (but this won't update optimizer state)
            self._opacity = nn.Parameter(opacities_new.requires_grad_(True))

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None:
                    stored_state = {}
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                stored_state["step"] = torch.tensor(0.0)

                if group['params'][0] in self.optimizer.state:
                    del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                if group['params'][0] in self.optimizer.state:
                    del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                # Initialize empty state with step key for new parameters
                self.optimizer.state[group['params'][0]] = {
                    "exp_avg": torch.zeros_like(group["params"][0]),
                    "exp_avg_sq": torch.zeros_like(group["params"][0]),
                    "step": torch.tensor(0.0)
                }
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]
        
        # Update feature tracking variables
        self._visibility_count = self._visibility_count[valid_points_mask]
        self._grad_ema_pos = self._grad_ema_pos[valid_points_mask]
        self._grad_ema_scale = self._grad_ema_scale[valid_points_mask]
        self._grad_ema_rot = self._grad_ema_rot[valid_points_mask]
        self._feature_vectors = None  # Invalidate cached features
        
        # Invalidate kNN cache after point pruning (major geometry change)
        self._knn_cache.clear()
        print(f"[KNN_CACHE] Cache cleared after pruning {mask.sum().item():,} points")
        
        # Sync auxiliary tensors for shape safety
        self.sync_auxiliary_tensors(f"prune_points(pruned={mask.sum().item()})")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                if group['params'][0] in self.optimizer.state:
                    del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                # Initialize state for new parameters
                self.optimizer.state[group['params'][0]] = {
                    "exp_avg": torch.zeros_like(group["params"][0]),
                    "exp_avg_sq": torch.zeros_like(group["params"][0]),
                    "step": torch.tensor(0.0)
                }
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Ensure tmp_radii synchronization
        if self.tmp_radii is not None:
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
            # Validate final shape
            if self.tmp_radii.shape[0] != self.get_xyz.shape[0]:
                print(f"Warning: tmp_radii shape {self.tmp_radii.shape[0]} != gaussian count {self.get_xyz.shape[0]} after densification")
                # Fix mismatch
                if self.tmp_radii.shape[0] < self.get_xyz.shape[0]:
                    padding = torch.zeros(self.get_xyz.shape[0] - self.tmp_radii.shape[0], device=self.tmp_radii.device, dtype=self.tmp_radii.dtype)
                    self.tmp_radii = torch.cat([self.tmp_radii, padding])
                else:
                    self.tmp_radii = self.tmp_radii[:self.get_xyz.shape[0]]
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # Extend feature tracking variables for new points
        n_new_points = new_xyz.shape[0]
        new_visibility = torch.zeros(n_new_points, device="cuda")
        new_grad_ema_pos = torch.zeros((n_new_points, 3), device="cuda")
        new_grad_ema_scale = torch.zeros((n_new_points, 3), device="cuda")
        new_grad_ema_rot = torch.zeros((n_new_points, 4), device="cuda")
        
        self._visibility_count = torch.cat((self._visibility_count, new_visibility))
        self._grad_ema_pos = torch.cat((self._grad_ema_pos, new_grad_ema_pos))
        self._grad_ema_scale = torch.cat((self._grad_ema_scale, new_grad_ema_scale))
        self._grad_ema_rot = torch.cat((self._grad_ema_rot, new_grad_ema_rot))
        self._feature_vectors = None  # Invalidate cached features
        
        # Invalidate kNN cache after point densification (major geometry change)
        self._knn_cache.clear()
        print(f"[KNN_CACHE] Cache cleared after adding {n_new_points:,} points via densification")
        
        # Sync auxiliary tensors for shape safety
        self.sync_auxiliary_tensors(f"densification_postfix(added={n_new_points})")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Ensure radii tensor matches current gaussian count
        if radii.shape[0] != self.get_xyz.shape[0]:
            print(f"Warning: radii shape {radii.shape[0]} doesn't match gaussian count {self.get_xyz.shape[0]}")
            # Pad or truncate radii to match current gaussian count
            if radii.shape[0] < self.get_xyz.shape[0]:
                padding = torch.zeros(self.get_xyz.shape[0] - radii.shape[0], device=radii.device, dtype=radii.dtype)
                radii = torch.cat([radii, padding])
            else:
                radii = radii[:self.get_xyz.shape[0]]
        
        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
        # Update visibility count
        self._visibility_count[update_filter] += 1
        
        # Update gradient EMAs with decay factor 0.9
        decay = 0.9
        if self._xyz.grad is not None:
            self._grad_ema_pos[update_filter] = decay * self._grad_ema_pos[update_filter] + (1 - decay) * torch.abs(self._xyz.grad[update_filter])
        if self._scaling.grad is not None:
            self._grad_ema_scale[update_filter] = decay * self._grad_ema_scale[update_filter] + (1 - decay) * torch.abs(self._scaling.grad[update_filter])
        if self._rotation.grad is not None:
            self._grad_ema_rot[update_filter] = decay * self._grad_ema_rot[update_filter] + (1 - decay) * torch.abs(self._rotation.grad[update_filter])

    def sync_auxiliary_tensors(self, operation="unknown"):
        """
        Shape-safety helper: Re-sizes/realigns all auxiliary tensors after densify/prune/merge operations.
        
        Ensures all auxiliary tensors match the current point count. If a tensor has incorrect size,
        it is rebuilt with zeros and valid indices are copied when possible.
        
        Args:
            operation: String describing the operation that triggered this sync (for logging)
        """
        current_n_points = self.get_xyz.shape[0]
        device = self.get_xyz.device
        
        print(f"[SYNC] Synchronizing auxiliary tensors after {operation} - target size: {current_n_points:,}")
        
        # 1. EMA Buffers (gradient tracking)
        expected_shapes = {
            'xyz_gradient_accum': (current_n_points, 1),
            'denom': (current_n_points, 1),
            '_grad_ema_pos': (current_n_points, 3),
            '_grad_ema_scale': (current_n_points, 3),
            '_grad_ema_rot': (current_n_points, 4)
        }
        
        for attr_name, expected_shape in expected_shapes.items():
            if hasattr(self, attr_name):
                current_tensor = getattr(self, attr_name)
                
                if current_tensor.shape != expected_shape:
                    print(f"  [SYNC] Resizing {attr_name}: {current_tensor.shape} -> {expected_shape}")
                    
                    # Create new tensor with zeros
                    new_tensor = torch.zeros(expected_shape, device=device, dtype=current_tensor.dtype)
                    
                    # Copy valid indices if old tensor has data
                    if current_tensor.numel() > 0 and current_tensor.shape[0] > 0:
                        copy_size = min(current_tensor.shape[0], expected_shape[0])
                        if copy_size > 0:
                            try:
                                new_tensor[:copy_size] = current_tensor[:copy_size]
                                print(f"    [SYNC] Copied {copy_size} entries from old {attr_name}")
                            except Exception as e:
                                print(f"    [SYNC] Could not copy {attr_name}: {e}")
                    
                    setattr(self, attr_name, new_tensor)
        
        # 2. 1D Auxiliary tensors
        scalar_tensors = {
            'max_radii2D': (current_n_points,),
            '_visibility_count': (current_n_points,)
        }
        
        for attr_name, expected_shape in scalar_tensors.items():
            if hasattr(self, attr_name):
                current_tensor = getattr(self, attr_name)
                
                if current_tensor.shape != expected_shape:
                    print(f"  [SYNC] Resizing {attr_name}: {current_tensor.shape} -> {expected_shape}")
                    
                    # Create new tensor with zeros
                    new_tensor = torch.zeros(expected_shape, device=device, dtype=current_tensor.dtype)
                    
                    # Copy valid indices if old tensor has data
                    if current_tensor.numel() > 0:
                        copy_size = min(current_tensor.shape[0], expected_shape[0])
                        if copy_size > 0:
                            try:
                                new_tensor[:copy_size] = current_tensor[:copy_size]
                                print(f"    [SYNC] Copied {copy_size} entries from old {attr_name}")
                            except Exception as e:
                                print(f"    [SYNC] Could not copy {attr_name}: {e}")
                    
                    setattr(self, attr_name, new_tensor)
        
        # 3. Reset cached feature vectors (they need recomputation with new point count)
        if hasattr(self, '_feature_vectors') and self._feature_vectors is not None:
            if self._feature_vectors.shape[0] != current_n_points:
                print(f"  [SYNC] Clearing cached feature vectors (shape mismatch: {self._feature_vectors.shape[0]} != {current_n_points})")
                self._feature_vectors = None
                self._last_refine_iteration = -1  # Force recomputation
        
        # 4. Validate all tensors have correct shapes
        validation_errors = []
        
        for attr_name, expected_shape in expected_shapes.items():
            if hasattr(self, attr_name):
                actual_shape = getattr(self, attr_name).shape
                if actual_shape != expected_shape:
                    validation_errors.append(f"{attr_name}: {actual_shape} != {expected_shape}")
        
        for attr_name, expected_shape in scalar_tensors.items():
            if hasattr(self, attr_name):
                actual_shape = getattr(self, attr_name).shape
                if actual_shape != expected_shape:
                    validation_errors.append(f"{attr_name}: {actual_shape} != {expected_shape}")
        
        if validation_errors:
            print(f"  [SYNC] ❌ Validation failed after sync:")
            for error in validation_errors:
                print(f"    - {error}")
            raise RuntimeError(f"Auxiliary tensor synchronization failed: {validation_errors}")
        else:
            print(f"  [SYNC] ✅ All auxiliary tensors synchronized successfully")
    
    def rebuild_boolean_mask(self, mask, target_size, name="unknown_mask"):
        """
        Shape-safety helper: Rebuild boolean mask with zero-padding if length != current point count.
        
        Args:
            mask: Input boolean mask tensor
            target_size: Target size (should match current point count)
            name: Name for logging purposes
            
        Returns:
            Properly sized boolean mask
        """
        if mask is None:
            return torch.zeros(target_size, dtype=torch.bool, device=self.get_xyz.device)
        
        if len(mask) == target_size:
            return mask  # Already correct size
        
        print(f"  [SYNC] Rebuilding {name}: {len(mask)} -> {target_size}")
        
        # Create new mask with zeros
        new_mask = torch.zeros(target_size, dtype=torch.bool, device=self.get_xyz.device)
        
        # Copy valid indices
        if len(mask) > 0:
            copy_size = min(len(mask), target_size)
            new_mask[:copy_size] = mask[:copy_size]
            print(f"    [SYNC] Copied {copy_size} mask entries, {new_mask.sum()} are True")
        
        return new_mask

    def compute_feature_vectors(self, iteration, force_update=False):
        """
        Compute per-Gaussian feature vectors containing:
        - Opacity α
        - Gradient EMA (position, scale, rotation)
        - kNN mean distance (batched top-k on positions)
        - DC/AC SH ratio
        - Number of views seen (visibility count)
        - Color variance (DC residual proxy)
        - Max 2D radii
        """
        # Only update if we're at a refine cycle or forced
        if not force_update and (iteration - self._last_refine_iteration) % self._refine_interval != 0:
            return self._feature_vectors
            
        self._last_refine_iteration = iteration
        n_gaussians = self.get_xyz.shape[0]
        
        # CUDA error prevention: Validate tensor state before computation
        try:
            # Check for basic tensor corruption
            if torch.isnan(self.get_xyz).any() or torch.isinf(self.get_xyz).any():
                print(f"[ERROR] NaN/Inf detected in xyz tensor at iteration {iteration}")
                return None
                
            if torch.isnan(self.get_opacity).any() or torch.isinf(self.get_opacity).any():
                print(f"[ERROR] NaN/Inf detected in opacity tensor at iteration {iteration}")
                return None
                
        except Exception as e:
            print(f"[ERROR] CUDA validation failed in compute_feature_vectors at iter {iteration}: {e}")
            return None
        
        with torch.no_grad():
            # Validate tensor shapes before computation - following trainFloaters.py pattern
            if (self._features_dc.shape[0] != n_gaussians or 
                self.get_opacity.shape[0] != n_gaussians or
                self._grad_ema_pos.shape[0] != n_gaussians):
                print(f"[ERROR] Tensor shape mismatch detected in compute_feature_vectors")
                print(f"  xyz: {n_gaussians}, features_dc: {self._features_dc.shape[0]}")
                print(f"  opacity: {self.get_opacity.shape[0]}, grad_ema_pos: {self._grad_ema_pos.shape[0]}")
                return None
            
            features = []
            
            # 1. Opacity α (1 dim) - with CUDA validation
            try:
                opacity = self.get_opacity.squeeze(-1)  # [N]
                if torch.isnan(opacity).any() or torch.isinf(opacity).any():
                    print(f"[ERROR] NaN/Inf in opacity tensor")
                    return None
                features.append(opacity)
            except RuntimeError as e:
                print(f"[ERROR] CUDA error in opacity computation: {e}")
                return None
            
            # 2. Gradient EMA for position, scale, rotation (3 dims) - with bounds checking
            try:
                pos_grad_mag = torch.norm(self._grad_ema_pos, dim=-1)  # [N]
                scale_grad_mag = torch.norm(self._grad_ema_scale, dim=-1)  # [N]  
                rot_grad_mag = torch.norm(self._grad_ema_rot, dim=-1)  # [N]
                
                # Validate gradient magnitudes
                for grad_name, grad_mag in [("pos", pos_grad_mag), ("scale", scale_grad_mag), ("rot", rot_grad_mag)]:
                    if torch.isnan(grad_mag).any() or torch.isinf(grad_mag).any():
                        print(f"[ERROR] NaN/Inf in {grad_name} gradient magnitude")
                        return None
                        
                features.extend([pos_grad_mag, scale_grad_mag, rot_grad_mag])
            except RuntimeError as e:
                print(f"[ERROR] CUDA error in gradient computation: {e}")
                return None
            
            # 3. kNN mean distance (1 dim) - with intelligent caching system
            try:
                k = min(4, n_gaussians - 1)
                if k <= 0:
                    knn_distances = torch.zeros(n_gaussians, device="cuda")
                else:
                    positions = self.get_xyz.clone().detach()  # Use clone to avoid memory issues
                    
                    # Try to get cached kNN results
                    cached_distances, _ = self._knn_cache.get(
                        xyz=positions, 
                        k=k,
                        visibility_count=self._visibility_count if len(self._visibility_count) > 0 else None,
                        camera_positions=self._recent_camera_positions
                    )
                    
                    if cached_distances is not None:
                        # Cache hit! Use cached results
                        knn_distances = cached_distances
                        print(f"[KNN_CACHE] Cache hit! Reusing kNN for {n_gaussians:,} points, k={k}")
                    else:
                        # Cache miss - compute kNN with optimized batching
                        print(f"[KNN_CACHE] Cache miss - computing kNN for {n_gaussians:,} points, k={k}")
                        
                        # Use smaller batch size for memory safety
                        batch_size = min(1000, n_gaussians // 4) if n_gaussians > 4000 else min(500, n_gaussians)
                        if batch_size < 1:
                            batch_size = 1
                            
                        knn_distances = torch.zeros(n_gaussians, device="cuda")
                        all_knn_indices = torch.zeros((n_gaussians, k), device="cuda", dtype=torch.long)
                        
                        for i in range(0, n_gaussians, batch_size):
                            end_idx = min(i + batch_size, n_gaussians)
                            batch_pos = positions[i:end_idx]
                            
                            try:
                                # Validate batch position tensor
                                if torch.isnan(batch_pos).any() or torch.isinf(batch_pos).any():
                                    print(f"[ERROR] NaN/Inf in batch positions at batch {i}")
                                    continue
                                    
                                # Compute distances to all other points with error handling
                                dists = torch.cdist(batch_pos, positions)  # [batch_size, N]
                                
                                # Validate distance tensor
                                if torch.isnan(dists).any() or torch.isinf(dists).any():
                                    print(f"[ERROR] NaN/Inf in distance computation at batch {i}")
                                    continue
                                
                                # Get k nearest neighbors (excluding self)
                                knn_dists, knn_indices = torch.topk(dists, k + 1, largest=False, dim=-1)  # +1 to exclude self
                                knn_distances[i:end_idx] = knn_dists[:, 1:k+1].mean(dim=-1)  # Exclude self (distance=0)
                                all_knn_indices[i:end_idx] = knn_indices[:, 1:k+1]  # Store indices for caching
                                
                            except RuntimeError as e:
                                print(f"[ERROR] CUDA error computing distances for batch {i}: {e}")
                                continue
                        
                        # Update cache with new results
                        self._knn_cache.update(
                            xyz=positions,
                            k=k, 
                            distances=knn_distances,
                            indices=all_knn_indices,
                            visibility_count=self._visibility_count if len(self._visibility_count) > 0 else None,
                            camera_positions=self._recent_camera_positions
                        )
                        
                features.append(knn_distances)
                
            except RuntimeError as e:
                print(f"[ERROR] CUDA error in kNN distance computation: {e}")
                return None
            
            # 4. DC/AC SH ratio (1 dim) - with validation and error handling
            try:
                dc_magnitude = torch.norm(self._features_dc.squeeze(-1), dim=-1)  # [N]
                
                # Validate DC magnitude
                if torch.isnan(dc_magnitude).any() or torch.isinf(dc_magnitude).any():
                    print(f"[ERROR] NaN/Inf in DC magnitude computation")
                    return None
                
                if self._features_rest.shape[-1] > 0:
                    ac_magnitude = torch.norm(self._features_rest.view(n_gaussians, -1), dim=-1)  # [N]
                    
                    # Validate AC magnitude
                    if torch.isnan(ac_magnitude).any() or torch.isinf(ac_magnitude).any():
                        print(f"[ERROR] NaN/Inf in AC magnitude computation")
                        return None
                    
                    # Add small epsilon to avoid division by zero
                    sh_ratio = dc_magnitude / (ac_magnitude + 1e-6)
                else:
                    sh_ratio = torch.ones_like(dc_magnitude)  # All DC if no AC coefficients
                    
                # Final validation of SH ratio
                if torch.isnan(sh_ratio).any() or torch.isinf(sh_ratio).any():
                    print(f"[ERROR] NaN/Inf in SH ratio computation")
                    return None
                    
                features.append(sh_ratio)
                
            except RuntimeError as e:
                print(f"[ERROR] CUDA error in SH ratio computation: {e}")
                return None
            
            # 5. Visibility count (normalized, 1 dim) - with error handling
            try:
                max_count = self._visibility_count.max()
                normalized_visibility = self._visibility_count / (max_count + 1e-6)
                
                # Validate normalized visibility
                if torch.isnan(normalized_visibility).any() or torch.isinf(normalized_visibility).any():
                    print(f"[ERROR] NaN/Inf in normalized visibility computation")
                    return None
                    
                features.append(normalized_visibility)
            except RuntimeError as e:
                print(f"[ERROR] CUDA error in visibility computation: {e}")
                return None
            
            # 6. Color variance proxy via DC residual (1 dim) - with error handling
            try:
                dc_colors = self._features_dc.squeeze(-1)  # [N, 3]
                
                # Validate DC colors tensor
                if torch.isnan(dc_colors).any() or torch.isinf(dc_colors).any():
                    print(f"[ERROR] NaN/Inf in DC colors tensor")
                    return None
                
                # Simple variance approximation using difference from local mean
                color_variance = torch.var(dc_colors, dim=0, keepdim=True).expand(n_gaussians, 3)
                color_var_magnitude = torch.norm(color_variance, dim=-1)  # [N]
                
                # Validate color variance magnitude
                if torch.isnan(color_var_magnitude).any() or torch.isinf(color_var_magnitude).any():
                    print(f"[ERROR] NaN/Inf in color variance magnitude")
                    return None
                    
                features.append(color_var_magnitude)
            except RuntimeError as e:
                print(f"[ERROR] CUDA error in color variance computation: {e}")
                return None
            
            # 7. Max 2D radii (1 dim, log-scale for better distribution) - with error handling
            try:
                radii_log = torch.log(self.max_radii2D + 1.0)  # +1 to handle zero radii
                
                # Validate radii log
                if torch.isnan(radii_log).any() or torch.isinf(radii_log).any():
                    print(f"[ERROR] NaN/Inf in radii log computation")
                    return None
                    
                features.append(radii_log)
            except RuntimeError as e:
                print(f"[ERROR] CUDA error in radii computation: {e}")
                return None
            
            # Stack all features: [N, total_dims] - with final validation
            try:
                # Total: 1 + 3 + 1 + 1 + 1 + 1 + 1 = 9 dimensions
                self._feature_vectors = torch.stack(features, dim=-1)
                
                # Final validation of feature vectors
                if torch.isnan(self._feature_vectors).any() or torch.isinf(self._feature_vectors).any():
                    print(f"[ERROR] NaN/Inf in final feature vectors")
                    return None
                    
            except RuntimeError as e:
                print(f"[ERROR] CUDA error in feature stacking: {e}")
                return None
            
        return self._feature_vectors
    
    def update_camera_position(self, camera_position):
        """Update recent camera positions for kNN cache invalidation."""
        if camera_position is not None:
            self._recent_camera_positions.append(camera_position)
            # Keep only recent positions (last 20) to avoid unbounded growth
            if len(self._recent_camera_positions) > 20:
                self._recent_camera_positions = self._recent_camera_positions[-20:]
    
    def get_knn_cache_stats(self):
        """Get kNN cache performance statistics."""
        return self._knn_cache.get_performance_stats()
        
    def print_knn_cache_stats(self):
        """Print kNN cache performance statistics."""
        self._knn_cache.print_performance_stats()
    
    def get_feature_vectors(self, iteration=None, force_update=False):
        """Get cached feature vectors or compute if needed"""
        if self._feature_vectors is None or force_update:
            return self.compute_feature_vectors(iteration or 0, force_update=True)
        return self._feature_vectors
    
    def compute_quantile_thresholds(self, features=None, percentiles=None):
        """
        Compute quantile thresholds for feature-based filtering.
        
        Args:
            features: Feature vectors [N, 9] (uses cached if None)
            percentiles: Dict of percentile values for each feature
            
        Returns:
            Dict of quantile thresholds for each feature
        """
        if features is None:
            features = self._feature_vectors
            if features is None:
                raise ValueError("No feature vectors available. Call compute_feature_vectors() first.")
        
        if percentiles is None:
            percentiles = {
                'alpha_low': 5.0,         # Low opacity threshold (5th percentile) - spec: α 5%
                'alpha_high': 95.0,       # High opacity threshold (95th percentile)
                'knn_high': 95.0,         # Isolated points (95th percentile of kNN distance) - spec: kNN 95%
                'pos_grad_high': 90.0,    # High position gradient (90th percentile) - spec: grad-EMA 90%
                'scale_grad_high': 90.0,  # High scale gradient (90th percentile) - spec: grad-EMA 90%
                'rot_grad_high': 90.0,    # High rotation gradient (90th percentile) - spec: grad-EMA 90%
                'visibility_low': 10.0,   # Low visibility (10th percentile)
                'radii_high': 95.0        # Large radii (95th percentile)
            }
        
        # Feature indices in the feature vector:
        # [0]: Opacity α, [1]: Position grad EMA, [2]: Scale grad EMA, [3]: Rotation grad EMA
        # [4]: kNN mean distance, [5]: DC/AC SH ratio, [6]: Visibility count, [7]: Color variance, [8]: Log max 2D radii
        
        thresholds = {}
        with torch.no_grad():
            # Alpha thresholds
            thresholds['alpha_low'] = torch.quantile(features[:, 0], percentiles['alpha_low'] / 100.0)
            thresholds['alpha_high'] = torch.quantile(features[:, 0], percentiles['alpha_high'] / 100.0)
            
            # Gradient thresholds
            thresholds['pos_grad_high'] = torch.quantile(features[:, 1], percentiles['pos_grad_high'] / 100.0)
            thresholds['scale_grad_high'] = torch.quantile(features[:, 2], percentiles['scale_grad_high'] / 100.0)
            thresholds['rot_grad_high'] = torch.quantile(features[:, 3], percentiles['rot_grad_high'] / 100.0)
            
            # Spatial isolation threshold
            thresholds['knn_high'] = torch.quantile(features[:, 4], percentiles['knn_high'] / 100.0)
            
            # Visibility threshold
            thresholds['visibility_low'] = torch.quantile(features[:, 6], percentiles['visibility_low'] / 100.0)
            
            # Radii threshold
            thresholds['radii_high'] = torch.quantile(features[:, 8], percentiles['radii_high'] / 100.0)
        
        return thresholds
    
    def generate_condition_masks(self, features=None, thresholds=None, percentiles=None):
        """
        Generate boolean masks for different floater conditions.
        
        Args:
            features: Feature vectors [N, 9]
            thresholds: Pre-computed thresholds (computed if None)
            percentiles: Percentile values for threshold computation
            
        Returns:
            Dict of boolean masks for different conditions
        """
        if features is None:
            features = self._feature_vectors
            if features is None:
                raise ValueError("No feature vectors available. Call compute_feature_vectors() first.")
        
        if thresholds is None:
            thresholds = self.compute_quantile_thresholds(features, percentiles)
        
        masks = {}
        with torch.no_grad():
            # Low opacity mask
            masks['low_alpha'] = features[:, 0] < thresholds['alpha_low']
            
            # High opacity mask  
            masks['high_alpha'] = features[:, 0] > thresholds['alpha_high']
            
            # Isolated points (high kNN distance)
            masks['isolated'] = features[:, 4] > thresholds['knn_high']
            
            # High gradient conditions
            masks['high_pos_grad'] = features[:, 1] > thresholds['pos_grad_high']
            masks['high_scale_grad'] = features[:, 2] > thresholds['scale_grad_high'] 
            masks['high_rot_grad'] = features[:, 3] > thresholds['rot_grad_high']
            
            # Combined high gradient mask
            masks['high_grad'] = (masks['high_pos_grad'] | 
                                masks['high_scale_grad'] | 
                                masks['high_rot_grad'])
            
            # Low visibility mask
            masks['low_visibility'] = features[:, 6] < thresholds['visibility_low']
            
            # Large radii mask
            masks['large_radii'] = features[:, 8] > thresholds['radii_high']
        
        return masks, thresholds
    
    def detect_floater_candidates(self, features=None, percentiles=None, logic_mode='default'):
        """
        Detect floater candidates using quantile-based thresholding with fast pre-filtering.
        
        Args:
            features: Feature vectors [N, 9]
            percentiles: Custom percentile thresholds
            logic_mode: Detection logic ('default', 'strict', 'loose', 'custom')
            
        Returns:
            floater_mask: Boolean mask of floater candidates [N]
            masks: Individual condition masks
            thresholds: Computed threshold values
            stats: Detection statistics
        """
        n_total = self.get_xyz.shape[0]
        
        # ═══ FAST PRE-FILTERING: Apply cheap tests first ═══
        print(f"[FLOATER] Starting fast pre-filtering on {n_total} points")
        
        with torch.no_grad():
            # Cheap test 1: Very low opacity (almost transparent)
            opacity = self.get_opacity.squeeze(-1)  # [N]
            very_low_opacity = opacity < 0.005  # Much stricter than typical 0.01
            
            # Cheap test 2: High position gradient instability (if available)
            high_instability = torch.zeros_like(very_low_opacity, dtype=torch.bool)
            if hasattr(self, '_grad_ema_pos') and self._grad_ema_pos is not None:
                pos_grad_norm = torch.norm(self._grad_ema_pos, dim=-1)
                # Use 95th percentile as threshold for high instability
                if len(pos_grad_norm) > 0:
                    instability_threshold = torch.quantile(pos_grad_norm, 0.95)
                    high_instability = pos_grad_norm > instability_threshold
            
            # Cheap test 3: Very low visibility count (rarely rendered)
            low_visibility = torch.zeros_like(very_low_opacity, dtype=torch.bool)
            if hasattr(self, '_visibility_count') and self._visibility_count is not None:
                # Points visible in <10% of typical visibility
                if self._visibility_count.max() > 0:
                    visibility_threshold = self._visibility_count.max() * 0.1
                    low_visibility = self._visibility_count < visibility_threshold
            
            # Combine cheap tests - candidate must pass at least one cheap test
            cheap_floater_candidates = very_low_opacity | high_instability | low_visibility
            
            n_cheap_candidates = cheap_floater_candidates.sum().item()
            n_filtered_out = n_total - n_cheap_candidates
            
            print(f"[FLOATER] Fast pre-filtering results:")
            print(f"  Very low opacity: {very_low_opacity.sum().item():,}")
            print(f"  High instability: {high_instability.sum().item():,}")  
            print(f"  Low visibility: {low_visibility.sum().item():,}")
            print(f"  Cheap candidates: {n_cheap_candidates:,} ({n_cheap_candidates/n_total:.1%})")
            print(f"  Filtered out: {n_filtered_out:,} ({n_filtered_out/n_total:.1%})")
        
        # If no cheap candidates found, return early (major speedup)
        if n_cheap_candidates == 0:
            print(f"[FLOATER] No cheap candidates found - skipping expensive feature computation")
            empty_mask = torch.zeros(n_total, dtype=torch.bool, device=self.get_xyz.device)
            return empty_mask, {}, {}, {
                'total_gaussians': n_total,
                'floater_candidates': 0,
                'floater_ratio': 0.0,
                'low_alpha_count': 0,
                'isolated_count': 0,
                'high_grad_count': 0,
                'low_visibility_count': 0,
                'cheap_filter_speedup': n_filtered_out
            }
        
        # ═══ LIMIT DETAILED EXAMINATION: Cap expensive analysis to top candidates ═══
        # Apply hard limits: max 10% of scene or 100k points, whichever is smaller
        max_detailed_candidates = min(
            int(n_total * 0.1),  # 10% of total points
            100_000,             # Hard cap at 100k points
            n_cheap_candidates   # Can't exceed cheap candidates
        )
        
        detailed_candidates_mask = cheap_floater_candidates.clone()
        final_candidate_indices = torch.where(cheap_floater_candidates)[0]
        
        if n_cheap_candidates > max_detailed_candidates:
            print(f"[FLOATER] Limiting detailed analysis: {n_cheap_candidates:,} → {max_detailed_candidates:,} candidates")
            
            # Prioritize candidates by combining cheap test scores
            with torch.no_grad():
                # Create priority scores for ranking
                priority_scores = torch.zeros(n_cheap_candidates, device=self.get_xyz.device)
                candidate_opacities = opacity[cheap_floater_candidates]
                
                # Priority 1: Lower opacity = higher priority (more likely floater)
                opacity_priority = 1.0 - candidate_opacities  # Invert: lower opacity = higher score
                priority_scores += opacity_priority * 2.0  # Weight opacity highly
                
                # Priority 2: Higher instability = higher priority  
                if high_instability.sum() > 0:
                    candidate_instability = high_instability[cheap_floater_candidates].float()
                    priority_scores += candidate_instability * 1.5
                
                # Priority 3: Lower visibility = higher priority
                if low_visibility.sum() > 0:
                    candidate_low_vis = low_visibility[cheap_floater_candidates].float()
                    priority_scores += candidate_low_vis * 1.0
                
                # Select top candidates by priority
                _, top_indices = torch.topk(priority_scores, max_detailed_candidates, largest=True)
                selected_candidate_indices = final_candidate_indices[top_indices]
                
                # Update masks to reflect limited selection
                detailed_candidates_mask.fill_(False)
                detailed_candidates_mask[selected_candidate_indices] = True
                final_candidate_indices = selected_candidate_indices
                
                print(f"[FLOATER] Selected top {max_detailed_candidates:,} candidates by priority scoring")
        
        n_detailed_candidates = len(final_candidate_indices)
        
        # ═══ EXPENSIVE FEATURE COMPUTATION: Only on limited detailed candidates ═══
        if features is None:
            print(f"[FLOATER] Computing expensive features for {n_detailed_candidates:,} detailed candidates")
            
            # Only compute features for the limited set of detailed candidates
            features = self._feature_vectors
            if features is None:
                raise ValueError("No feature vectors available. Call compute_feature_vectors() first.")
                
            # Filter features to only include detailed candidates
            if features.shape[0] == n_total:
                features = features[detailed_candidates_mask]
            else:
                # Features were computed for full set - filter them appropriately
                if features.shape[0] == n_cheap_candidates:
                    # Features match cheap candidates - need to sub-select  
                    cheap_indices = torch.where(cheap_floater_candidates)[0]
                    detailed_in_cheap = torch.isin(cheap_indices, final_candidate_indices)
                    features = features[detailed_in_cheap]
                else:
                    features = features[detailed_candidates_mask] if features.shape[0] == n_total else features
        
        # Generate condition masks (only for cheap candidates now)
        masks, thresholds = self.generate_condition_masks(features, None, percentiles)
        
        # Apply detection logic (only on the cheap candidate subset)
        with torch.no_grad():
            if logic_mode == 'default':
                # Original logic: (low_alpha OR isolated) AND high_grad
                subset_floater_mask = (masks['low_alpha'] | masks['isolated']) & masks['high_grad']
                
            elif logic_mode == 'strict':
                # Stricter conditions: low_alpha AND isolated AND high_grad
                subset_floater_mask = masks['low_alpha'] & masks['isolated'] & masks['high_grad']
                
            elif logic_mode == 'loose':
                # Looser conditions: low_alpha OR isolated OR (high_grad AND low_visibility)
                subset_floater_mask = (masks['low_alpha'] | 
                                     masks['isolated'] | 
                                     (masks['high_grad'] & masks['low_visibility']))
                                      
            elif logic_mode == 'comprehensive':
                # More comprehensive: include large radii and low visibility
                spatial_issues = masks['isolated'] | masks['large_radii'] 
                appearance_issues = masks['low_alpha'] | masks['low_visibility']
                subset_floater_mask = (spatial_issues | appearance_issues) & masks['high_grad']
                
            else:
                raise ValueError(f"Unknown logic_mode: {logic_mode}")
        
        # ═══ EXPAND BACK TO FULL SIZE: Map subset results to full mask ═══
        # Create full-size floater mask and populate only the detailed candidate positions
        floater_mask = torch.zeros(n_total, dtype=torch.bool, device=self.get_xyz.device)
        
        if features.shape[0] == n_detailed_candidates:
            # Features were computed only for detailed candidates
            floater_mask[final_candidate_indices] = subset_floater_mask
        elif features.shape[0] == n_cheap_candidates:
            # Features were computed for all cheap candidates, but we only analyzed a subset
            if n_detailed_candidates < n_cheap_candidates:
                # Map back through the cheap->detailed selection
                cheap_indices = torch.where(cheap_floater_candidates)[0]
                detailed_in_cheap = torch.isin(cheap_indices, final_candidate_indices)
                expanded_subset_mask = torch.zeros(n_cheap_candidates, dtype=torch.bool, device=self.get_xyz.device)
                expanded_subset_mask[detailed_in_cheap] = subset_floater_mask
                floater_mask[cheap_floater_candidates] = expanded_subset_mask
            else:
                # All cheap candidates were analyzed
                floater_mask[cheap_floater_candidates] = subset_floater_mask
        else:
            # Features were computed for full set
            floater_mask = subset_floater_mask
        
        # Compute statistics (for full dataset)
        n_floaters = floater_mask.sum().item()
        
        efficiency_msg = f"analyzed {n_detailed_candidates:,}/{n_total:,} ({n_detailed_candidates/n_total:.1%})"
        print(f"[FLOATER] Final results: {n_floaters:,} floaters found, {efficiency_msg}")
        
        stats = {
            'total_gaussians': n_total,
            'floater_candidates': n_floaters,
            'floater_ratio': n_floaters / n_total if n_total > 0 else 0.0,
            'low_alpha_count': masks['low_alpha'].sum().item(),
            'isolated_count': masks['isolated'].sum().item(), 
            'high_grad_count': masks['high_grad'].sum().item(),
            'low_visibility_count': masks['low_visibility'].sum().item(),
            'logic_mode': logic_mode,
            'cheap_filter_speedup': n_filtered_out,
            'cheap_candidates': n_cheap_candidates,
            'detailed_candidates': n_detailed_candidates,
            'analysis_efficiency': n_detailed_candidates / n_total if n_total > 0 else 0.0,
            'max_detailed_limit': max_detailed_candidates
        }
        
        return floater_mask, masks, thresholds, stats
