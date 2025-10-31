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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
from PIL import Image

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image_path, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.resolution = resolution
        self.train_test_exp = train_test_exp
        self.is_test_dataset = is_test_dataset
        self.is_test_view = is_test_view

        # Cache for lazy loading
        self._original_image = None
        self._alpha_mask = None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # Load image once to get dimensions, but don't keep it in memory
        temp_image = Image.open(image_path)
        resized_image_rgb = PILtoTorch(temp_image, resolution)
        self.image_width = resized_image_rgb.shape[2]
        self.image_height = resized_image_rgb.shape[1]
        temp_image.close()
        del resized_image_rgb

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def width(self):
        return self.image_width

    @property
    def height(self):
        return self.image_height

    @property
    def original_image(self):
        if self._original_image is None:
            self._load_image()
        return self._original_image

    @property
    def alpha_mask(self):
        if self._alpha_mask is None:
            self._load_image()
        return self._alpha_mask

    def _load_image(self):
        """Load image from disk and cache it"""
        image = Image.open(self.image_path)
        resized_image_rgb = PILtoTorch(image, self.resolution)
        gt_image = resized_image_rgb[:3, ...]

        if resized_image_rgb.shape[0] == 4:
            self._alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else:
            self._alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if self.train_test_exp and self.is_test_view:
            if self.is_test_dataset:
                self._alpha_mask[..., :self._alpha_mask.shape[-1] // 2] = 0
            else:
                self._alpha_mask[..., self._alpha_mask.shape[-1] // 2:] = 0

        self._original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        image.close()
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

myMiniCam3 = MiniCam
