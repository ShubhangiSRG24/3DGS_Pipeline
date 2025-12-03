import csv
import math
import os
import sys
import time
from argparse import ArgumentParser, Namespace
from random import randint, seed

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Initialize LPIPS metric
lpips_fn = lpips.LPIPS(net="alex").cuda()

# ----------------------------------------------------------------------------------
# NOVELTY: Monocular Depth Estimator with Uncertainty
# ----------------------------------------------------------------------------------


class DepthEstimator:
    def __init__(self, cache_dir):
        self.device = "cuda"
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Depth cache directory: {self.cache_dir}")

        print("Loading MiDaS DPT_Large depth estimation model from PyTorch Hub...")
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform
        print("Depth estimation model loaded.")

    def get_cache_path(self, image_name, suffix):
        return os.path.join(self.cache_dir, os.path.basename(image_name) + f"_{suffix}.pt")

    @torch.no_grad()
    def _estimate(self, img_numpy):
        transformed = self.transform(img_numpy).to(self.device)
        prediction = self.model(transformed)
        return F.interpolate(
            prediction.unsqueeze(1), size=img_numpy.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()

    def precompute_depths_with_uncertainty(self, cameras):
        print("Pre-computing depth maps and uncertainty for all cameras...")
        for cam in tqdm(cameras, desc="Pre-computing Depth"):
            depth_cache_path = self.get_cache_path(cam.image_name, "depth")
            uncert_cache_path = self.get_cache_path(cam.image_name, "uncertainty")

            if os.path.exists(depth_cache_path) and os.path.exists(uncert_cache_path):
                continue

            from torchvision.transforms.functional import hflip, to_pil_image

            pil_image = to_pil_image(cam.original_image.cpu())
            img_numpy = np.array(pil_image)
            img_numpy_flipped = np.array(hflip(pil_image))

            depth1 = self._estimate(img_numpy)
            depth2_flipped = self._estimate(img_numpy_flipped)
            depth2 = torch.fliplr(depth2_flipped)

            mean_depth = torch.mean(torch.stack([depth1, depth2]), dim=0)
            uncertainty = torch.var(torch.stack([depth1, depth2]), dim=0)

            mean_depth = (mean_depth - mean_depth.min()) / (mean_depth.max() - mean_depth.min() + 1e-6)
            uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-6)

            torch.save(mean_depth.cpu(), depth_cache_path)
            torch.save(uncertainty.cpu(), uncert_cache_path)

    def load_depth_and_uncertainty(self, camera):
        depth_path = self.get_cache_path(camera.image_name, "depth")
        uncert_path = self.get_cache_path(camera.image_name, "uncertainty")
        if os.path.exists(depth_path) and os.path.exists(uncert_path):
            depth = torch.load(depth_path).to(self.device)
            uncertainty = torch.load(uncert_path).to(self.device)
            return depth, uncertainty
        return None, None


# ----------------------------------------------------------------------------------
# Enhanced GaussianModel with Differentiable Pruning and Event Counters
# ----------------------------------------------------------------------------------


class RefinedGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)
        self.training_args = None
        self._importance = None
        self._pending_new_importance = None
        self.scene_extent = None
        self.event_counters = {
            "Added(Densify)_Iter": 0,
            "Pruned(Densify)_Iter": 0,
            "Pruned(Final)_Iter": 0,
            "Added(Densify)_Cum": 0,
            "Pruned(Densify)_Cum": 0,
            "Pruned(Final)_Cum": 0,
        }

    def reset_iter_counters(self):
        for key in self.event_counters:
            if key.endswith("_Iter"):
                self.event_counters[key] = 0

    def training_setup(self, training_args):
        self.training_args = training_args
        super().training_setup(training_args)

        num_points = self.get_xyz.shape[0]
        if self._importance is None or self._importance.shape[0] != num_points:
            self._importance = nn.Parameter(torch.ones(num_points, 1, device="cuda") * 2.0)

        found = False
        for group in self.optimizer.param_groups:
            if group["name"] == "importance":
                group["params"] = [self._importance]
                found = True
        if not found:
            self.optimizer.add_param_group(
                {"params": [self._importance], "lr": training_args.feature_lr * 0.1, "name": "importance"}
            )

    @property
    def get_importance(self):
        return self._importance

    def register_scene_extent(self, extent):
        self.scene_extent = extent

    def densify_and_add(self, properties_dict):
        num_added = properties_dict["xyz"].shape[0]
        self.event_counters["Added(Densify)_Iter"] += num_added
        self.event_counters["Added(Densify)_Cum"] += num_added

        super().densify_and_add(properties_dict)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        num_new = int(selected_pts_mask.sum().item())
        if num_new > 0:
            self._pending_new_importance = self._importance[selected_pts_mask].detach().clone()

        super().densify_and_clone(grads, grad_threshold, scene_extent)

        if num_new > 0:
            self.event_counters["Added(Densify)_Iter"] += num_new
            self.event_counters["Added(Densify)_Cum"] += num_new

        self._pending_new_importance = None

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        base_new = int(selected_pts_mask.sum().item())
        num_new = base_new * N
        if num_new > 0:
            pending_importance = self._importance[selected_pts_mask].detach().clone()
            self._pending_new_importance = pending_importance.repeat(N, 1)

        super().densify_and_split(grads, grad_threshold, scene_extent, N=N)

        if num_new > 0:
            self.event_counters["Added(Densify)_Iter"] += num_new
            self.event_counters["Added(Densify)_Cum"] += num_new

        self._pending_new_importance = None

    def densification_postfix(
        self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii
    ):
        if self._pending_new_importance is not None:
            new_importance = self._pending_new_importance.to(new_xyz.device)
        else:
            new_importance = torch.ones(new_xyz.shape[0], 1, device=new_xyz.device) * 2.0

        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "importance": new_importance,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._importance = optimizable_tensors["importance"]
        self._pending_new_importance = None

        if self.tmp_radii is not None:
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
            if self.tmp_radii.shape[0] != self.get_xyz.shape[0]:
                print(
                    f"Warning: tmp_radii shape {self.tmp_radii.shape[0]} != gaussian count {self.get_xyz.shape[0]} after densification"
                )
                if self.tmp_radii.shape[0] < self.get_xyz.shape[0]:
                    padding = torch.zeros(
                        self.get_xyz.shape[0] - self.tmp_radii.shape[0],
                        device=self.tmp_radii.device,
                        dtype=self.tmp_radii.dtype,
                    )
                    self.tmp_radii = torch.cat([self.tmp_radii, padding])
                else:
                    self.tmp_radii = self.tmp_radii[: self.get_xyz.shape[0]]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        n_new_points = new_xyz.shape[0]
        if n_new_points > 0:
            new_visibility = torch.zeros(n_new_points, device="cuda")
            new_grad_ema_pos = torch.zeros((n_new_points, 3), device="cuda")
            new_grad_ema_scale = torch.zeros((n_new_points, 3), device="cuda")
            new_grad_ema_rot = torch.zeros((n_new_points, 4), device="cuda")

            self._visibility_count = torch.cat((self._visibility_count, new_visibility))
            self._grad_ema_pos = torch.cat((self._grad_ema_pos, new_grad_ema_pos))
            self._grad_ema_scale = torch.cat((self._grad_ema_scale, new_grad_ema_scale))
            self._grad_ema_rot = torch.cat((self._grad_ema_rot, new_grad_ema_rot))

        self._feature_vectors = None
        self._knn_cache.clear()
        print(f"[KNN_CACHE] Cache cleared after adding {n_new_points:,} points via densification")

        self.sync_auxiliary_tensors(f"densification_postfix(added={n_new_points})")

    def prune_points(self, mask, source="densify"):
        num_pruned = mask.sum().item()
        if source == "densify":
            self.event_counters["Pruned(Densify)_Iter"] += num_pruned
            self.event_counters["Pruned(Densify)_Cum"] += num_pruned
        elif source == "final":
            self.event_counters["Pruned(Final)_Iter"] += num_pruned
            self.event_counters["Pruned(Final)_Cum"] += num_pruned

        super().prune_points(mask)

        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if group.get("name") == "importance" and len(group["params"]) == 1:
                    self._importance = group["params"][0]
                    break

    def floater_cleanup(
        self,
        visibility_threshold,
        grad_threshold,
        opacity_threshold,
        importance_threshold,
        max_fraction,
        distance_factor=1.5,
    ):
        """Aggressively prune low-confidence Gaussians (floaters)."""
        if self._xyz.shape[0] == 0:
            return

        device = self._xyz.device
        with torch.no_grad():
            visibility = self._visibility_count
            importance = torch.sigmoid(self.get_importance).squeeze(-1)
            opacity = self.get_opacity.squeeze(-1)
            grad_norm = torch.norm(self._grad_ema_pos, dim=1)
            scaling = self.get_scaling.max(dim=1).values

            candidate_mask = visibility <= visibility_threshold
            if opacity_threshold > 0:
                candidate_mask &= opacity <= opacity_threshold
            if importance_threshold > 0:
                candidate_mask &= importance <= importance_threshold
            if grad_threshold > 0:
                candidate_mask &= grad_norm <= grad_threshold

            if self.scene_extent is not None and self.scene_extent > 0:
                distances = torch.norm(self._xyz - self._xyz.mean(dim=0, keepdim=True), dim=1)
                far_mask = distances >= (self.scene_extent * distance_factor)
                candidate_mask |= far_mask & (importance <= importance_threshold)

            # Suppress oversized Gaussians that rarely contribute
            size_mask = scaling >= (self.percent_dense * (self.scene_extent if self.scene_extent else 1.0) * 0.75)
            candidate_mask |= size_mask & (importance <= importance_threshold * 1.2)

            candidate_indices = torch.nonzero(candidate_mask).squeeze(-1)
            if candidate_indices.numel() == 0:
                return

            max_fraction = max_fraction if max_fraction > 0 else 0.05
            prune_cap = max(1, int(self.get_xyz.shape[0] * max_fraction))
            if candidate_indices.numel() > prune_cap:
                scores = (
                    0.45 * importance[candidate_indices]
                    + 0.35 * opacity[candidate_indices]
                    + 0.20 * grad_norm[candidate_indices]
                )
                _, order = torch.topk(scores, k=prune_cap, largest=False)
                candidate_indices = candidate_indices[order]

            prune_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=device)
            prune_mask[candidate_indices] = True
            self.prune_points(prune_mask, source="densify")

    def cap_gaussian_count(self, max_gaussians):
        if max_gaussians <= 0 or self.get_xyz.shape[0] <= max_gaussians:
            return
        with torch.no_grad():
            importance = torch.sigmoid(self.get_importance).squeeze(-1)
            excess = self.get_xyz.shape[0] - max_gaussians
            prune_indices = torch.topk(importance, k=excess, largest=False).indices
            prune_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
            prune_mask[prune_indices] = True
            self.prune_points(prune_mask, source="densify")

    def final_prune(self, threshold=0.05):
        with torch.no_grad():
            prune_mask = (torch.sigmoid(self.get_importance) < threshold).squeeze()
            if prune_mask.sum().item() > 0:
                self.prune_points(prune_mask, source="final")
                print(f"\n[FINAL PRUNE] Removed {prune_mask.sum().item()} Gaussians with low importance.")


# ----------------------------------------------------------------------------------
# Main Training and Evaluation Functions
# ----------------------------------------------------------------------------------


def validation_loop(iteration, scene, gaussians, pipe, background, depth_estimator):
    print("\n[ITER {}] Starting validation loop...".format(iteration))

    # Set deterministic evaluation for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    psnr_list, ssim_list, lpips_list, l1_list = [], [], [], []
    depth_abs_rel_list, depth_delta1_list = [], []
    total_render_time = 0.0

    for viewpoint_cam in tqdm(scene.getTestCameras(), desc="Validation"):
        gt_image = viewpoint_cam.original_image.cuda()

        start_time = time.time()
        with torch.no_grad():
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, background, importance_scores=torch.sigmoid(gaussians.get_importance)
            )
        total_render_time += time.time() - start_time

        image = torch.clamp(render_pkg["render"], 0.0, 1.0)

        mse = F.mse_loss(image, gt_image).item()
        psnr_list.append(10 * math.log10(1.0 / (mse + 1e-9)))
        ssim_list.append(ssim(image, gt_image).item())
        lpips_list.append(lpips_fn(image.unsqueeze(0) * 2 - 1, gt_image.unsqueeze(0) * 2 - 1).item())
        l1_list.append(l1_loss(image, gt_image).item())

        gt_depth, _ = depth_estimator.load_depth_and_uncertainty(viewpoint_cam)
        if gt_depth is not None:
            rendered_depth = render_pkg["depth"].squeeze(0)
            if gt_depth.shape != rendered_depth.shape:
                gt_depth = F.interpolate(
                    gt_depth.unsqueeze(0).unsqueeze(0), size=rendered_depth.shape, mode="bilinear", align_corners=False
                ).squeeze()

            valid_mask = (gt_depth > 1e-6) & (rendered_depth > 1e-6)
            if valid_mask.sum() > 0:
                gt_valid, rendered_valid = gt_depth[valid_mask], rendered_depth[valid_mask]

                # Robust scale/shift alignment for evaluation
                t_gt, t_rendered = torch.median(gt_valid), torch.median(rendered_valid)
                s_gt = torch.median(torch.abs(gt_valid - t_gt))
                s_rendered = torch.median(torch.abs(rendered_valid - t_rendered))
                rendered_aligned = (rendered_valid - t_rendered) * (s_gt / (s_rendered + 1e-6)) + t_gt

                abs_rel = torch.mean(torch.abs(gt_valid - rendered_aligned) / gt_valid).item()
                depth_abs_rel_list.append(abs_rel)
                thresh = torch.max((gt_valid / rendered_aligned), (rendered_aligned / gt_valid))
                delta1 = (thresh < 1.25).float().mean().item()
                depth_delta1_list.append(delta1)

    # Restore default settings
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    metrics = {
        "Val/PSNR": np.mean(psnr_list),
        "Val/SSIM": np.mean(ssim_list),
        "Val/LPIPS": np.mean(lpips_list),
        "Val/L1": np.mean(l1_list),
        "Val/DepthAbsRel": np.mean(depth_abs_rel_list) if depth_abs_rel_list else 0,
        "Val/DepthDelta1.25": np.mean(depth_delta1_list) if depth_delta1_list else 0,
        "RenderFPS": len(scene.getTestCameras()) / total_render_time if total_render_time > 0 else 0,
    }

    print(
        f"[ITER {iteration}] Validation: PSNR {metrics['Val/PSNR']:.2f}, LPIPS {metrics['Val/LPIPS']:.4f}, Depth Abs-Rel {metrics['Val/DepthAbsRel']:.4f}"
    )
    return metrics


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # Set seeds for reproducibility
    seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    first_iter = 0
    tb_writer, csv_writer, csv_file = prepare_output_and_logger(dataset)
    gaussians = RefinedGaussianModel(dataset.sh_degree)

    dataset.test_iterations = testing_iterations
    scene = Scene(dataset, gaussians)
    gaussians.register_scene_extent(scene.cameras_extent)
    gaussians.training_setup(opt)

    depth_estimator = DepthEstimator(cache_dir=os.path.join(scene.model_path, "depth_cache"))
    depth_estimator.precompute_depths_with_uncertainty(scene.getTrainCameras() + scene.getTestCameras())

    num_cameras = len(scene.getTrainCameras())
    cam_uid_to_idx = {cam.uid: i for i, cam in enumerate(scene.getTrainCameras())}
    depth_alignment_params = nn.Embedding(num_cameras, 2).cuda()
    torch.nn.init.normal_(depth_alignment_params.weight, mean=0.0, std=1e-4)
    depth_align_optimizer = torch.optim.Adam(depth_alignment_params.parameters(), lr=1e-3)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    progress_bar = tqdm(range(first_iter, opt.iterations + 1), desc="Training progress")

    for iteration in range(first_iter + 1, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        gaussians.optimizer.zero_grad(set_to_none=True)
        depth_align_optimizer.zero_grad(set_to_none=True)

        viewpoint_cam = scene.getTrainCameras().copy().pop(randint(0, len(scene.getTrainCameras()) - 1))

        render_pkg = render(
            viewpoint_cam, gaussians, pipe, background, importance_scores=torch.sigmoid(gaussians.get_importance)
        )
        image, rendered_depth, visibility_filter = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["visibility_filter"],
        )

        gt_image = viewpoint_cam.original_image.cuda()

        l1_val = l1_loss(image, gt_image)
        d_ssim = 1.0 - ssim(image, gt_image)
        color_loss = (1.0 - opt.lambda_dssim) * l1_val + opt.lambda_dssim * d_ssim

        depth_loss = 0.0
        gt_depth, uncertainty = depth_estimator.load_depth_and_uncertainty(viewpoint_cam)

        if gt_depth is not None:
            rendered_depth_no_channel = rendered_depth.squeeze(0)
            if gt_depth.shape != rendered_depth_no_channel.shape:
                gt_depth = F.interpolate(
                    gt_depth.unsqueeze(0).unsqueeze(0),
                    size=rendered_depth_no_channel.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            cam_idx = cam_uid_to_idx[viewpoint_cam.uid]
            align_params = depth_alignment_params(torch.tensor([cam_idx], device="cuda"))
            scale, shift = align_params[0, 0], align_params[0, 1]
            rendered_depth_aligned = rendered_depth_no_channel * torch.exp(scale) + shift

            weights = torch.exp(-uncertainty / opt.uncertainty_tau)
            weights = weights / (weights.mean() + 1e-6)

            valid_depth_mask = (rendered_depth_no_channel > 0) & (gt_depth > 0)
            if valid_depth_mask.sum() > 0:
                depth_loss = (
                    weights[valid_depth_mask]
                    * F.huber_loss(
                        rendered_depth_aligned[valid_depth_mask], gt_depth[valid_depth_mask], reduction="none"
                    )
                ).mean()

        importance_loss = opt.lambda_importance * torch.sigmoid(gaussians.get_importance).mean()
        lpips_loss_val = opt.lambda_lpips * lpips_fn(image.unsqueeze(0) * 2 - 1, gt_image.unsqueeze(0) * 2 - 1).mean()

        total_loss = color_loss + opt.lambda_depth * depth_loss + importance_loss + lpips_loss_val
        total_loss.backward()

        gaussians.optimizer.step()
        depth_align_optimizer.step()

        iter_end.record()
        torch.cuda.synchronize()
        elapsed = iter_start.elapsed_time(iter_end)

        with torch.no_grad():
            if iteration < opt.refinement_start_iter:
                if iteration < opt.densify_until_iter:
                    viewspace_point_tensor, radii = render_pkg["viewspace_points"], render_pkg["radii"]
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, 20, radii)
                    if iteration % opt.opacity_reset_interval == 0:
                        gaussians.reset_opacity()
            else:
                if iteration % opt.cleanup_interval == 0:
                    gaussians.final_prune()

            if iteration >= opt.floater_cleanup_start and iteration % opt.floater_cleanup_interval == 0:
                gaussians.floater_cleanup(
                    visibility_threshold=opt.floater_visibility_threshold,
                    grad_threshold=opt.floater_grad_threshold,
                    opacity_threshold=opt.floater_opacity_threshold,
                    importance_threshold=opt.floater_importance_threshold,
                    max_fraction=opt.floater_cleanup_max_ratio,
                    distance_factor=opt.floater_distance_factor,
                )
                gaussians.cap_gaussian_count(opt.max_gaussians)

        progress_bar.update(1)

        if iteration in saving_iterations:
            scene.save(iteration)
        eval_due = False
        if iteration in testing_iterations:
            eval_due = True
        elif opt.eval_interval > 0 and iteration % opt.eval_interval == 0:
            eval_due = True

        if eval_due:  # Time to run validation/logging
            log_row = validation_loop(iteration, scene, gaussians, pipe, background, depth_estimator)

            model_size_mb = 0.0
            if iteration in saving_iterations:
                ply_path = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
                if os.path.exists(ply_path):
                    model_size_mb = os.path.getsize(ply_path) / (1024 * 1024)

            total_pruned_iter = gaussians.event_counters.get("Pruned(Densify)_Iter", 0) + gaussians.event_counters.get(
                "Pruned(Final)_Iter", 0
            )
            total_pruned_cum = gaussians.event_counters.get("Pruned(Densify)_Cum", 0) + gaussians.event_counters.get(
                "Pruned(Final)_Cum", 0
            )

            log_row.update(
                {
                    "Iter": iteration,
                    "Train/L1": l1_val.item(),
                    "Train/TotalLoss": total_loss.item(),
                    "Train/SSIM": 1.0 - d_ssim.item(),
                    "IterTime(ms)": elapsed,
                    **gaussians.event_counters,
                    "Pruned(Total)_Iter": total_pruned_iter,
                    "Pruned(Total)_Cum": total_pruned_cum,
                    "TotalGaussians": gaussians.get_xyz.shape[0],
                    "Model_Size(MB)": f"{model_size_mb:.2f}" if model_size_mb > 0 else "N/A",
                }
            )
            csv_writer.writerow({k: log_row.get(k, "N/A") for k in csv_writer.fieldnames})
            csv_file.flush()
            gaussians.reset_iter_counters()

        if iteration in checkpoint_iterations:
            torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")

    progress_bar.close()
    csv_file.close()
    print("\nTraining complete.")


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = "./output"
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    csv_path = os.path.join(args.model_path, "evaluation_log.csv")
    csv_file = open(csv_path, "w", newline="")

    header = [
        "Iter",
        "Val/PSNR",
        "Val/SSIM",
        "Val/LPIPS",
        "Val/L1",
        "Val/DepthAbsRel",
        "Val/DepthDelta1.25",
        "Train/L1",
        "Train/TotalLoss",
        "Train/SSIM",
        "RenderFPS",
        "IterTime(ms)",
        "Added(Densify)_Iter",
        "Pruned(Densify)_Iter",
        "Pruned(Final)_Iter",
        "Added(Densify)_Cum",
        "Pruned(Densify)_Cum",
        "Pruned(Final)_Cum",
        "Pruned(Total)_Iter",
        "Pruned(Total)_Cum",
        "TotalGaussians",
        "Model_Size(MB)",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=header)
    csv_writer.writeheader()

    tb_writer = SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None
    return tb_writer, csv_writer, csv_file


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--lambda_depth", type=float, default=0.1, help="Weight for the depth loss term.")
    parser.add_argument("--lambda_lpips", type=float, default=0.1, help="Weight for the LPIPS regularization term.")
    parser.add_argument(
        "--lambda_importance", type=float, default=0.01, help="Weight for the differentiable pruning regularization."
    )
    parser.add_argument("--uncertainty_tau", type=float, default=0.1, help="Temperature for uncertainty weighting.")
    parser.add_argument(
        "--cleanup_interval", type=int, default=100, help="Interval for cleanup/regularization in Stage 2."
    )
    parser.add_argument("--floater_cleanup_start", type=int, default=2000, help="Iteration to start floater cleanup.")
    parser.add_argument(
        "--floater_cleanup_interval", type=int, default=400, help="Interval between floater cleanup passes."
    )
    parser.add_argument(
        "--floater_visibility_threshold",
        type=int,
        default=2,
        help="Minimum visibility count before a Gaussian is considered a floater.",
    )
    parser.add_argument(
        "--floater_grad_threshold", type=float, default=5e-4, help="Maximum gradient EMA for floater candidates."
    )
    parser.add_argument(
        "--floater_opacity_threshold", type=float, default=0.04, help="Maximum opacity for floater candidates."
    )
    parser.add_argument(
        "--floater_importance_threshold",
        type=float,
        default=0.35,
        help="Maximum learned importance score for floater candidates (after sigmoid).",
    )
    parser.add_argument(
        "--floater_cleanup_max_ratio",
        type=float,
        default=0.05,
        help="Maximum fraction of Gaussians that can be removed in a single floater cleanup.",
    )
    parser.add_argument(
        "--floater_distance_factor",
        type=float,
        default=1.4,
        help="Distance factor relative to scene extent for floater detection.",
    )
    parser.add_argument(
        "--max_gaussians", type=int, default=180000, help="Upper bound on active Gaussian count (0 to disable)."
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=200,
        help="Run validation and log metrics every N iterations (0 to disable periodic eval).",
    )

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset_params = lp.extract(args)
    opt_params = op.extract(args)

    opt_params.lambda_depth = args.lambda_depth
    opt_params.lambda_lpips = args.lambda_lpips
    opt_params.lambda_importance = args.lambda_importance
    opt_params.uncertainty_tau = args.uncertainty_tau
    opt_params.cleanup_interval = args.cleanup_interval
    opt_params.floater_cleanup_start = args.floater_cleanup_start
    opt_params.floater_cleanup_interval = args.floater_cleanup_interval
    opt_params.floater_visibility_threshold = args.floater_visibility_threshold
    opt_params.floater_grad_threshold = args.floater_grad_threshold
    opt_params.floater_opacity_threshold = args.floater_opacity_threshold
    opt_params.floater_importance_threshold = args.floater_importance_threshold
    opt_params.floater_cleanup_max_ratio = args.floater_cleanup_max_ratio
    opt_params.floater_distance_factor = args.floater_distance_factor
    opt_params.max_gaussians = args.max_gaussians
    opt_params.eval_interval = args.eval_interval

    training(
        dataset_params,
        opt_params,
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )
