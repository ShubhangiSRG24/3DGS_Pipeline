# === trainFloaters_new.py (conflict-safe CLI + adaptive caps) ===
import os, sys, math, csv, time, glob, uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from random import randint, seed
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.knn_utils import batched_knn_distances
from arguments import ModelParams, PipelineParams, OptimizationParams

import lpips
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# ----------------------- small utility -----------------------
def _ensure_batch_dim(t: torch.Tensor) -> torch.Tensor:
    return t if t.dim() == 4 else t.unsqueeze(0)

def _append_if_finite(lst, v):
    if math.isfinite(v): lst.append(float(v))

def safe_add_argument(parser: ArgumentParser, *flags, **kwargs):
    """Add an argparse option only if none of its flags already exist."""
    existing = parser._option_string_actions  # type: ignore[attr-defined]
    if any(f in existing for f in flags):
        return
    parser.add_argument(*flags, **kwargs)

# ----------------------- LPIPS -----------------------
lpips_fn = lpips.LPIPS(net='alex')
if torch.cuda.is_available():
    lpips_fn = lpips_fn.cuda()
lpips_fn.eval()

# ----------------------- Depth Estimator -----------------------
class DepthEstimator:
    def __init__(self, cache_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.available = True
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.dpt_transform
        except Exception as exc:
            print(f"[Depth] MiDaS unavailable: {exc} (depth loss disabled)")
            self.available, self.model, self.transform = False, None, None

    def get_cache_path(self, image_name, suffix):
        return os.path.join(self.cache_dir, os.path.basename(image_name) + f"_{suffix}.pt")

    @torch.no_grad()
    def _estimate(self, img_numpy):
        transformed = self.transform(img_numpy).to(self.device)
        pred = self.model(transformed)
        return F.interpolate(pred.unsqueeze(1), size=img_numpy.shape[:2],
                             mode="bicubic", align_corners=False).squeeze()

    def precompute_depths_with_uncertainty(self, cameras):
        if not self.available: return
        from torchvision.transforms.functional import to_pil_image, hflip
        for cam in tqdm(cameras, desc="[Depth] Precompute"):
            dpth_p = self.get_cache_path(cam.image_name, "depth")
            unc_p = self.get_cache_path(cam.image_name, "uncertainty")
            if os.path.exists(dpth_p) and os.path.exists(unc_p): continue
            pil = to_pil_image(cam.original_image.cpu())
            img = np.array(pil)
            img_f = np.array(hflip(pil))
            d1 = self._estimate(img)
            d2 = torch.fliplr(self._estimate(img_f))
            mean = torch.mean(torch.stack([d1, d2]), dim=0)
            var  = torch.var(torch.stack([d1, d2]), dim=0)
            mean = (mean - mean.min()) / (mean.max() - mean.min() + 1e-6)
            var  = (var  - var.min())  / (var.max()  - var.min()  + 1e-6)
            torch.save(mean.cpu(), dpth_p)
            torch.save(var.cpu(),  unc_p)

    def load_depth_and_uncertainty(self, camera):
        if not self.available: return None, None
        dpth_p = self.get_cache_path(camera.image_name, "depth")
        unc_p  = self.get_cache_path(camera.image_name, "uncertainty")
        if os.path.exists(dpth_p) and os.path.exists(unc_p):
            return torch.load(dpth_p).to(self.device), torch.load(unc_p).to(self.device)
        return None, None

# ----------------------- Refined Gaussian Model -----------------------
class RefinedGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)
        self.tmp_radii = None
        self.training_args = None
        self._importance = None
        self._pending_new_importance = None
        self.scene_extent = None
        self.event_counters = {
            'Added(Densify)_Iter': 0, 'Pruned(Densify)_Iter': 0, 'Pruned(Final)_Iter': 0,
            'Added(Densify)_Cum': 0,  'Pruned(Densify)_Cum': 0,  'Pruned(Final)_Cum': 0,
        }
        self._visibility_count = torch.empty(0)
        self._grad_ema_pos = torch.empty(0)
        self._grad_ema_scale = torch.empty(0)
        self._grad_ema_rot = torch.empty(0)
        self._feature_vectors = None
        from scene.knn_cache import KNNCache
        self._knn_cache = KNNCache()

    def reset_iter_counters(self):
        for k in list(self.event_counters.keys()):
            if k.endswith("_Iter"): self.event_counters[k] = 0

    def training_setup(self, training_args):
        self.training_args = training_args
        super().training_setup(training_args)
        n = self.get_xyz.shape[0]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self._importance is None or self._importance.shape[0] != n:
            self._importance = nn.Parameter(torch.ones(n, 1, device=dev) * 2.0)
        if self._visibility_count.numel() == 0 or self._visibility_count.shape[0] != n:
            self._visibility_count = torch.zeros(n, device=dev)
            self._grad_ema_pos   = torch.zeros((n, 3), device=dev)
            self._grad_ema_scale = torch.zeros((n, 3), device=dev)
            self._grad_ema_rot   = torch.zeros((n, 4), device=dev)
        found = False
        for g in self.optimizer.param_groups:
            if g.get("name") == "importance":
                g["params"] = [self._importance]; found = True
        if not found:
            self.optimizer.add_param_group({'params': [self._importance],
                                            'lr': training_args.feature_lr * 0.1,
                                            'name': "importance"})

    @property
    def get_importance(self):
        return self._importance

    def register_scene_extent(self, extent): self.scene_extent = extent

    # --------- densify helpers with optional per-iter limits (0 = unlimited) ---------
    def _limit_selected_points(self, selected_mask, scores, max_points, reason="densify"):
        if max_points <= 0: return selected_mask
        cnt = int(selected_mask.sum().item())
        if cnt <= max_points: return selected_mask
        idx = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0: return selected_mask
        k = min(max_points, idx.numel())
        if scores is None:
            chosen = idx[torch.randperm(idx.numel(), device=idx.device)[:k]]
        else:
            pruned_scores = torch.nan_to_num(scores[idx], nan=0.0, posinf=0.0, neginf=0.0)
            _, order = torch.topk(pruned_scores, k=k, largest=True)
            chosen = idx[order]
        mask = torch.zeros_like(selected_mask)
        mask[chosen] = True
        if self.training_args is not None and getattr(self.training_args, "log_densify_clamps", False):
            print(f"[densify] limited {reason}: kept {k} of {cnt} candidates")
        return mask

    def densify_and_add(self, props):
        num_added = props['xyz'].shape[0]
        self.event_counters['Added(Densify)_Iter'] += num_added
        self.event_counters['Added(Densify)_Cum'] += num_added
        super().densify_and_add(props)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        grad_norm = torch.norm(grads, dim=-1)
        scaling_ok = torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        candidate = torch.logical_and(grad_norm >= grad_threshold, scaling_ok)
        sel = candidate
        if self.training_args is not None:
            cap = int(getattr(self.training_args, "max_clone_points_per_iter", 0) or 0)
            if cap > 0:
                sel = self._limit_selected_points(candidate, grad_norm, cap, reason="clone")
        disallowed = candidate & ~sel
        if disallowed.any():
            grads = grads.clone()
            idx = torch.nonzero(disallowed, as_tuple=False).squeeze(-1)
            if idx.numel() > 0: grads[idx[idx < grads.shape[0]]] = 0.0
        num_new = int(sel.sum().item())
        if num_new > 0:
            self._pending_new_importance = self._importance[sel].detach().clone()
        super().densify_and_clone(grads, grad_threshold, scene_extent)
        if num_new > 0:
            self.event_counters['Added(Densify)_Iter'] += num_new
            self.event_counters['Added(Densify)_Cum'] += num_new
        self._pending_new_importance = None

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n0 = self.get_xyz.shape[0]
        pad = torch.zeros((n0), device=self.get_xyz.device)
        pad[:grads.shape[0]] = grads.squeeze()
        scale_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        candidate = torch.logical_and(pad >= grad_threshold, scale_mask)
        sel = candidate
        if self.training_args is not None:
            cap = int(getattr(self.training_args, "max_split_points_per_iter", 0) or 0)
            if cap > 0:
                seeds = cap // max(N, 1)
                if seeds == 0:
                    sel = torch.zeros_like(candidate, dtype=torch.bool)
                else:
                    sel = self._limit_selected_points(candidate, pad, seeds, reason=f"split (N={N})")
        disallowed = candidate & ~sel
        if disallowed.any():
            grads = grads.clone()
            idx = torch.nonzero(disallowed, as_tuple=False).squeeze(-1)
            if idx.numel() > 0: grads[idx[idx < grads.shape[0]]] = 0.0
        base = int(sel.sum().item())
        if base > 0:
            self._pending_new_importance = self._importance[sel].detach().clone().repeat(N,1)
        super().densify_and_split(grads, grad_threshold, scene_extent, N=N)
        if base * N > 0:
            self.event_counters['Added(Densify)_Iter'] += base * N
            self.event_counters['Added(Densify)_Cum'] += base * N
        self._pending_new_importance = None

    def densification_postfix(self, new_xyz, f_dc, f_rest, opac, scaling, rot, new_tmp_radii):
        new_importance = self._pending_new_importance.to(new_xyz.device) if self._pending_new_importance is not None \
                         else torch.ones(new_xyz.shape[0], 1, device=new_xyz.device) * 2.0
        d = {"xyz": new_xyz, "f_dc": f_dc, "f_rest": f_rest, "opacity": opac,
             "scaling": scaling, "rotation": rot, "importance": new_importance}
        optim = self.cat_tensors_to_optimizer(d)
        self._xyz, self._features_dc, self._features_rest = optim["xyz"], optim["f_dc"], optim["f_rest"]
        self._opacity, self._scaling, self._rotation = optim["opacity"], optim["scaling"], optim["rotation"]
        self._importance = optim["importance"]
        self._pending_new_importance = None
        if self.tmp_radii is None:
            self.tmp_radii = new_tmp_radii
        else:
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
            if self.tmp_radii.shape[0] != self.get_xyz.shape[0]:
                need = self.get_xyz.shape[0] - self.tmp_radii.shape[0]
                if need > 0:
                    self.tmp_radii = torch.cat([self.tmp_radii, torch.zeros(need, device=self.tmp_radii.device)])
                else:
                    self.tmp_radii = self.tmp_radii[:self.get_xyz.shape[0]]
        dev = self._xyz.device
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=dev)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=dev)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=dev)
        n_new = new_xyz.shape[0]
        if n_new > 0:
            self._visibility_count = torch.cat((self._visibility_count, torch.zeros(n_new, device=dev)))
            self._grad_ema_pos   = torch.cat((self._grad_ema_pos,   torch.zeros((n_new,3), device=dev)))
            self._grad_ema_scale = torch.cat((self._grad_ema_scale, torch.zeros((n_new,3), device=dev)))
            self._grad_ema_rot   = torch.cat((self._grad_ema_rot,   torch.zeros((n_new,4), device=dev)))
        self._feature_vectors = None
        self._knn_cache.clear()
        self.sync_auxiliary_tensors("densification_postfix")

    def _identify_high_frequency_regions(self, grads):
        n = self.get_xyz.shape[0]
        if n == 0:
            d = self.get_xyz.device if self.get_xyz.numel() else "cuda"
            return torch.zeros(0, dtype=torch.bool, device=d), torch.zeros(0, device=d)
        dev = self.get_xyz.device
        grad_ema = self._grad_ema_pos if self._grad_ema_pos.shape[0] == n else torch.zeros((n,3), device=dev)
        grad_norm = torch.norm(grad_ema, dim=1)
        sh_energy = torch.zeros(n, device=dev)
        if self._features_rest is not None and self._features_rest.numel() > 0:
            sh_energy = torch.norm(self._features_rest.reshape(n, -1), dim=1)
        min_scale = self.get_scaling.min(dim=1).values
        grad_norm_norm = grad_norm / (torch.max(grad_norm) + 1e-6)
        sh_norm = sh_energy / (torch.max(sh_energy) + 1e-6)
        target_min_scale = getattr(self.training_args, "high_freq_min_scale", 0.02)
        scale_score = torch.clamp((target_min_scale - min_scale) / (target_min_scale + 1e-6), 0.0, 1.0)
        score = 0.5*sh_norm + 0.3*grad_norm_norm + 0.2*scale_score
        hf_mask = (sh_norm >= getattr(self.training_args, "high_freq_sh_threshold", 0.2)) \
                  | (grad_norm_norm >= getattr(self.training_args, "high_freq_grad_threshold", 0.35)) \
                  | (scale_score > 0.0)
        return hf_mask, score

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        hf_mask, hf_score = self._identify_high_frequency_regions(grads)
        boosted = grads.clone()
        b = getattr(self.training_args, "high_freq_grad_boost", 1.5)
        if b > 1.0 and hf_mask.any(): boosted[hf_mask] = boosted[hf_mask] * b
        split_scale = getattr(self.training_args, "high_freq_split_scale", 0.85)
        split_factor = max(2, int(getattr(self.training_args, "high_freq_split_factor", 2)))
        split_th = max_grad * split_scale if split_scale > 0 else max_grad
        self.tmp_radii = radii
        self.densify_and_clone(boosted, max_grad, extent)
        self.densify_and_split(boosted, split_th, extent, N=split_factor)

        cur = self.get_xyz.shape[0]
        if hf_mask.shape[0] != cur:
            m = torch.zeros(cur, dtype=torch.bool, device=hf_mask.device); m[:hf_mask.shape[0]] = hf_mask; hf_mask = m
        if hf_score.shape[0] != cur:
            s = torch.zeros(cur, device=hf_score.device); s[:hf_score.shape[0]] = hf_score; hf_score = s

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_vs = self.max_radii2D > max_screen_size
            big_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_vs | big_ws
        if hf_mask.any():
            keep_th = getattr(self.training_args, "high_freq_keep_score", 0.25)
            prune_mask = prune_mask & ~hf_mask
            prune_mask = prune_mask & ~(hf_score >= keep_th)
        self.prune_points(prune_mask)
        self.tmp_radii = None
        torch.cuda.empty_cache()

    def prune_points(self, mask, source='densify'):
        num = mask.sum().item()
        if source == 'densify':
            self.event_counters['Pruned(Densify)_Iter'] += num
            self.event_counters['Pruned(Densify)_Cum'] += num
        else:
            self.event_counters['Pruned(Final)_Iter'] += num
            self.event_counters['Pruned(Final)_Cum'] += num
        if self._visibility_count.numel() > 0: self._visibility_count = self._visibility_count[~mask]
        if self._grad_ema_pos.numel() > 0:   self._grad_ema_pos   = self._grad_ema_pos[~mask]
        if self._grad_ema_scale.numel() > 0: self._grad_ema_scale = self._grad_ema_scale[~mask]
        if self._grad_ema_rot.numel() > 0:   self._grad_ema_rot   = self._grad_ema_rot[~mask]
        if self.tmp_radii is None:
            self.tmp_radii = torch.zeros(mask.shape[0], device=self.get_xyz.device)
        elif self.tmp_radii.shape[0] != mask.shape[0]:
            need = mask.shape[0] - self.tmp_radii.shape[0]
            if need > 0:
                self.tmp_radii = torch.cat([self.tmp_radii, torch.zeros(need, device=self.tmp_radii.device)])
            else:
                self.tmp_radii = self.tmp_radii[:mask.shape[0]]
        super().prune_points(mask)
        if self.optimizer is not None:
            for g in self.optimizer.param_groups:
                if g.get("name") == "importance" and len(g["params"]) == 1:
                    self._importance = g["params"][0]; break

    def floater_cleanup(self,
                        visibility_threshold,
                        grad_threshold,
                        opacity_threshold,
                        importance_threshold,
                        max_fraction,
                        distance_factor=1.5,
                        knn_k=6,
                        isolation_scale=0.05,
                        detail_scale=0.02,
                        color_std_threshold=0.05,
                        sh_energy_threshold=0.1,
                        sample_ratio=0.3):
        """Prune isolated low-importance Gaussians while keeping high-frequency detail."""
        num_points = self._xyz.shape[0]
        if num_points == 0:
            return

        device = self._xyz.device
        knn_k = int(max(2, min(knn_k, num_points - 1))) if num_points > 2 else 2

        with torch.no_grad():
            visibility = self._visibility_count
            importance = torch.sigmoid(self.get_importance).squeeze(-1)
            opacity = self.get_opacity.squeeze(-1)
            grad_norm = torch.norm(self._grad_ema_pos, dim=1)
            scaling = self.get_scaling
            max_scale = scaling.max(dim=1).values
            min_scale = scaling.min(dim=1).values

            if self._features_rest is not None and self._features_rest.numel() > 0:
                sh_flat = self._features_rest.reshape(self._features_rest.shape[0], -1)
                sh_energy = torch.norm(sh_flat, dim=1)
            else:
                sh_energy = torch.zeros(num_points, device=device)

            base_mask = (visibility <= visibility_threshold)
            if opacity_threshold > 0:
                base_mask &= (opacity <= opacity_threshold)
            if importance_threshold > 0:
                base_mask &= (importance <= importance_threshold)
            if grad_threshold > 0:
                base_mask &= (grad_norm <= grad_threshold)

            use_sampling = num_points > 120_000
            xyz_detached = self._xyz.detach()
            knn_distances, knn_indices, valid_mask = batched_knn_distances(
                xyz_detached,
                knn_k,
                chunk_size=4096,
                use_sampling=use_sampling,
                sample_ratio=sample_ratio
            )
            sampled_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

            mean_knn = torch.zeros(num_points, device=device)
            color_variation = torch.zeros(num_points, device=device)
            if sampled_indices.numel() > 0:
                mean_knn_values = knn_distances.mean(dim=1)
                mean_knn[sampled_indices] = mean_knn_values

                if self._features_dc.numel() > 0:
                    dc_colors = self._features_dc.squeeze(1)
                    neighbor_colors = dc_colors[knn_indices]
                    center_colors = dc_colors[sampled_indices].unsqueeze(1)
                    color_variation[sampled_indices] = torch.norm(neighbor_colors - center_colors, dim=2).mean(dim=1)

            scene_extent = self.scene_extent if self.scene_extent and self.scene_extent > 0 else float(max_scale.mean().item() + 1e-3)
            isolation_threshold = max(isolation_scale * scene_extent, 1e-4)
            detail_scale_threshold = max(detail_scale * scene_extent, 1e-5)

            isolation_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
            if sampled_indices.numel() > 0:
                isolation_mask[sampled_indices] = mean_knn[sampled_indices] >= isolation_threshold

            far_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
            if self.scene_extent is not None and self.scene_extent > 0:
                center = self._xyz.mean(dim=0, keepdim=True)
                distances = torch.norm(self._xyz - center, dim=1)
                far_mask = distances >= (self.scene_extent * distance_factor)

            fine_scale_mask = min_scale <= detail_scale_threshold
            detail_protect_mask = fine_scale_mask.clone()
            if color_std_threshold > 0 and sampled_indices.numel() > 0:
                detail_protect_mask[sampled_indices] |= (color_variation[sampled_indices] >= color_std_threshold)
            if sh_energy_threshold > 0:
                detail_protect_mask |= (sh_energy >= sh_energy_threshold)

            base_mask &= ~detail_protect_mask

            floater_mask = base_mask & isolation_mask
            far_candidate_mask = base_mask & far_mask & isolation_mask
            large_mask = base_mask & (max_scale >= detail_scale_threshold * 4.0) & isolation_mask
            if color_std_threshold > 0 and sampled_indices.numel() > 0:
                low_variance_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
                low_variance_mask[sampled_indices] = color_variation[sampled_indices] <= color_std_threshold
                large_mask &= low_variance_mask

            candidate_mask = (floater_mask | far_candidate_mask | large_mask) & ~detail_protect_mask
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
            if candidate_indices.numel() == 0:
                return

            max_fraction = max_fraction if max_fraction > 0 else 0.05
            prune_cap = max(1, int(num_points * max_fraction))
            if candidate_indices.numel() > prune_cap:
                iso_denom = isolation_threshold if isolation_threshold > 1e-6 else (mean_knn[candidate_indices].mean().abs() + 1e-6)
                normalized_isolation = torch.where(
                    mean_knn[candidate_indices] > 0,
                    mean_knn[candidate_indices] / iso_denom,
                    torch.zeros_like(mean_knn[candidate_indices])
                )
                scores = (
                    0.45 * importance[candidate_indices] +
                    0.30 * opacity[candidate_indices] +
                    0.20 * grad_norm[candidate_indices] -
                    0.35 * normalized_isolation
                )
                _, order = torch.topk(scores, k=prune_cap, largest=False)
                candidate_indices = candidate_indices[order]

            prune_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
            prune_mask[candidate_indices] = True
            self.prune_points(prune_mask, source='densify')
            if hasattr(self, "_knn_cache"):
                self._knn_cache.clear()

    def cap_gaussian_count(self, max_gaussians):
        if max_gaussians <= 0 or self.get_xyz.shape[0] <= max_gaussians: return
        with torch.no_grad():
            imp = torch.sigmoid(self.get_importance).squeeze(-1)
            excess = self.get_xyz.shape[0] - max_gaussians
            idx = torch.topk(imp, k=excess, largest=False).indices
            mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
            mask[idx] = True
            self.prune_points(mask, source='densify')

    def final_prune(self, threshold=0.05):
        with torch.no_grad():
            mask = (torch.sigmoid(self.get_importance) < threshold).squeeze()
            if mask.sum().item() > 0:
                self.prune_points(mask, source='final')
                print(f"[FINAL PRUNE] Removed {mask.sum().item()} Gaussians (low importance)")

    def sync_auxiliary_tensors(self, context=""):
        n = self._xyz.shape[0]; dev = self._xyz.device
        def _sync(t, shape): 
            if t.shape[0] == n: return t
            if t.shape[0] < n:
                pad = torch.zeros((n - t.shape[0],) + t.shape[1:], device=dev)
                return torch.cat([t, pad])
            return t[:n]
        self._visibility_count = _sync(self._visibility_count, (n,))
        self._grad_ema_pos     = _sync(self._grad_ema_pos,     (n,3))
        self._grad_ema_scale   = _sync(self._grad_ema_scale,   (n,3))
        self._grad_ema_rot     = _sync(self._grad_ema_rot,     (n,4))

# ----------------------- Losses -----------------------
def compute_multiscale_l1(pred, tgt, scales):
    if not scales: return torch.zeros((), device=pred.device, dtype=pred.dtype)
    uniq = []
    for s in scales:
        if s and s not in uniq: uniq.append(float(s))
    uniq.sort(reverse=True)
    if 1.0 not in uniq: uniq.insert(0, 1.0)
    loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    wsum = 0.0
    for i, s in enumerate(uniq):
        w = 1.0 / (2 ** i)
        if s == 1.0:
            p, q = pred, tgt
        else:
            p = F.interpolate(pred, scale_factor=s, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            q = F.interpolate(tgt,  scale_factor=s, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        loss = loss + w * F.l1_loss(p, q)
        wsum += w
    return loss / max(wsum, 1e-6)

def gradient_consistency_loss(pred, tgt):
    def g(img):
        dx = img[..., :, 1:] - img[..., :, :-1]
        dy = img[..., 1:, :] - img[..., :-1, :]
        return dx, dy
    pdx, pdy = g(pred); tdx, tdy = g(tgt)
    return 0.5 * (F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy))

def color_moment_loss(pred, tgt, eps=1e-6):
    pf = pred.view(pred.shape[0], pred.shape[1], -1); qf = tgt.view(tgt.shape[0], tgt.shape[1], -1)
    pm, qm = pf.mean(-1), qf.mean(-1)
    ps, qs = torch.sqrt(pf.var(-1, unbiased=False)+eps), torch.sqrt(qf.var(-1, unbiased=False)+eps)
    return (pm-qm).abs().mean() + (ps-qs).abs().mean()

# ----------------------- Train/Val -----------------------
def validation_loop(iteration, scene, gauss, pipe, bg, depth_est):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    psnr_t = ssim_t = lpips_t = l1_t = 0.0
    d_absrel, d_delta1 = [], []
    total_t = 0.0

    cams = scene.getTestCameras()
    if not cams:
        tr = scene.getTrainCameras()
        if not tr: 
            return {'Val/PSNR':0.0,'Val/SSIM':0.0,'Val/LPIPS':0.0,'Val/L1':0.0,'Val/DepthAbsRel':0.0,'Val/DepthDelta1.25':0.0,'RenderFPS':0.0}
        step = max(1, len(tr)//10); cams = [tr[i] for i in range(0,len(tr),step)]

    for vcam in tqdm(cams, desc="[Val]"):
        gt = vcam.original_image.cuda()
        start = time.time()
        with torch.no_grad():
            pkg = render(vcam, gauss, pipe, bg, importance_scores=torch.sigmoid(gauss.get_importance))
        total_t += time.time() - start
        img = torch.clamp(pkg["render"], 0, 1)
        psnr_t += psnr(_ensure_batch_dim(img), _ensure_batch_dim(gt)).mean().item()
        ssim_t += ssim(_ensure_batch_dim(img), _ensure_batch_dim(gt)).double().item()
        lpips_t += lpips_fn(_ensure_batch_dim(img), _ensure_batch_dim(gt)).mean().double().item()
        l1_t += l1_loss(img, gt).mean().double().item()

        gtd, _ = depth_est.load_depth_and_uncertainty(vcam)
        if gtd is not None:
            rd = pkg["depth"].squeeze(0)
            if gtd.shape != rd.shape:
                gtd = F.interpolate(gtd.unsqueeze(0).unsqueeze(0), size=rd.shape, mode='bilinear', align_corners=False).squeeze()
            mask = (gtd > 1e-6) & (rd > 1e-6)
            if mask.sum() > 0:
                gtv, rdv = gtd[mask], rd[mask]
                t1, t2 = torch.median(gtv), torch.median(rdv)
                s1 = torch.median((gtv - t1).abs()); s2 = torch.median((rdv - t2).abs())
                rd_al = (rdv - t2) * (s1 / (s2 + 1e-6)) + t1
                absrel = torch.mean((gtv - rd_al).abs() / gtv).item()
                thresh = torch.max((gtv / rd_al), (rd_al / gtv))
                delta1 = (thresh < 1.25).float().mean().item()
                _append_if_finite(d_absrel, absrel)
                _append_if_finite(d_delta1, delta1)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    n = len(cams)
    return {
        'Val/PSNR': psnr_t/n if n else 0.0,
        'Val/SSIM': ssim_t/n if n else 0.0,
        'Val/LPIPS': lpips_t/n if n else 0.0,
        'Val/L1': l1_t/n if n else 0.0,
        'Val/DepthAbsRel': float(np.mean(d_absrel)) if d_absrel else 0.0,
        'Val/DepthDelta1.25': float(np.mean(d_delta1)) if d_delta1 else 0.0,
        'RenderFPS': (n/total_t) if total_t > 0 else 0.0
    }

def training(dataset, opt, pipe, test_iters, save_iters, ckpt_iters, checkpoint, debug_from):
    seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    first_iter = 0
    tb_writer, csv_writer, csv_file = prepare_output_and_logger(dataset)
    gauss = RefinedGaussianModel(dataset.sh_degree)
    dataset.test_iterations = test_iters
    scene = Scene(dataset, gauss)
    gauss.register_scene_extent(scene.cameras_extent)
    gauss.training_setup(opt)

    depth_est = DepthEstimator(cache_dir=os.path.join(scene.model_path, "depth_cache"))
    if not depth_est.available and getattr(opt, "lambda_depth", 0.0) > 0:
        print("[Depth] disabling lambda_depth (no estimator)"); opt.lambda_depth = 0.0
    depth_est.precompute_depths_with_uncertainty(scene.getTrainCameras() + scene.getTestCameras())

    # multiscale list
    scales = []
    if hasattr(opt, "multiscale_scales"):
        raw = getattr(opt, "multiscale_scales", "")
        if isinstance(raw, str):
            try: scales = [float(s.strip()) for s in raw.split(",") if s.strip()]
            except ValueError: scales = [1.0, 0.5, 0.25]
        elif isinstance(raw, (list, tuple)):
            scales = [float(s) for s in raw if s]
    if not scales: scales = [1.0, 0.5, 0.25]

    # depth alignment per training camera
    cams = scene.getTrainCameras(); cam_uid_to_idx = {c.uid: i for i, c in enumerate(cams)}
    depth_align = nn.Embedding(len(cams), 2).cuda(); torch.nn.init.normal_(depth_align.weight, mean=0.0, std=1e-4)
    depth_align_opt = torch.optim.Adam(depth_align.parameters(), lr=1e-3)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gauss.restore(model_params, opt)

    bg = torch.tensor(([1,1,1] if dataset.white_background else [0,0,0]), dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)
    bar = tqdm(range(first_iter, opt.iterations + 1), desc="[Train]")

    for it in range(first_iter + 1, opt.iterations + 1):
        iter_start.record()
        gauss.update_learning_rate(it)
        if it % 1000 == 0: gauss.oneupSHdegree()
        gauss.optimizer.zero_grad(set_to_none=True)
        depth_align_opt.zero_grad(set_to_none=True)

        vcam = scene.getTrainCameras().copy().pop(randint(0, len(scene.getTrainCameras()) - 1))
        pkg = render(vcam, gauss, pipe, bg, importance_scores=torch.sigmoid(gauss.get_importance))
        img, depth_r, vis = pkg["render"], pkg["depth"], pkg["visibility_filter"]
        gt = vcam.original_image.cuda()
        ib, gb = _ensure_batch_dim(img), _ensure_batch_dim(gt)

        # photometric loss
        l1v = l1_loss(img, gt)
        dssim = 1.0 - ssim(ib, gb)
        color_loss = (1.0 - opt.lambda_dssim) * l1v + opt.lambda_dssim * dssim

        multiscale_loss = compute_multiscale_l1(ib, gb, scales) if getattr(opt, "lambda_multiscale", 0.0) > 0 else torch.zeros((), device=img.device)
        grad_loss       = gradient_consistency_loss(ib, gb)      if getattr(opt, "lambda_grad", 0.0)        > 0 else torch.zeros((), device=img.device)
        moment_loss     = color_moment_loss(ib, gb)              if getattr(opt, "lambda_color_moment", 0.0)> 0 else torch.zeros((), device=img.device)

        # depth loss (uncertainty-weighted + per-camera scale/shift)
        depth_loss = torch.zeros((), device=img.device)
        gtd, unc = depth_est.load_depth_and_uncertainty(vcam)
        if gtd is not None:
            rd = depth_r.squeeze(0)
            if gtd.shape != rd.shape:
                gtd = F.interpolate(gtd.unsqueeze(0).unsqueeze(0), size=rd.shape, mode='bilinear', align_corners=False).squeeze()
            cam_idx = cam_uid_to_idx[vcam.uid]
            s, t = depth_align(torch.tensor([cam_idx], device="cuda"))[0]
            rd_aligned = rd * torch.exp(s) + t
            w = torch.exp(-unc / opt.uncertainty_tau); w = w / (w.mean() + 1e-6)
            m = (rd > 0) & (gtd > 0)
            if m.sum() > 0:
                depth_loss = (w[m] * F.huber_loss(rd_aligned[m], gtd[m], reduction='none')).mean()

        importance_loss = opt.lambda_importance * torch.sigmoid(gauss.get_importance).mean()
        lpips_metric = lpips_fn(ib, gb).mean()
        lpips_loss = opt.lambda_lpips * lpips_metric

        total = color_loss + opt.lambda_multiscale*multiscale_loss + opt.lambda_grad*grad_loss \
              + opt.lambda_color_moment*moment_loss + opt.lambda_depth*depth_loss \
              + importance_loss + lpips_loss
        total.backward()
        gauss.optimizer.step()
        depth_align_opt.step()

        iter_end.record(); torch.cuda.synchronize()
        elapsed = iter_start.elapsed_time(iter_end)

        with torch.no_grad():
            if it < opt.refinement_start_iter:
                if it < opt.densify_until_iter:
                    viewspace, radii = pkg["viewspace_points"], pkg["radii"]
                    gauss.add_densification_stats(viewspace, vis)
                    if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                        gauss.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, 20, radii)
                    if it % opt.opacity_reset_interval == 0: gauss.reset_opacity()
            else:
                if it % opt.cleanup_interval == 0: gauss.final_prune()

            if it >= opt.floater_cleanup_start and it % opt.floater_cleanup_interval == 0:
                gauss.floater_cleanup(
                    visibility_threshold=opt.floater_visibility_threshold,
                    grad_threshold=opt.floater_grad_threshold,
                    opacity_threshold=opt.floater_opacity_threshold,
                    importance_threshold=opt.floater_importance_threshold,
                    max_fraction=opt.floater_cleanup_max_ratio,
                    distance_factor=opt.floater_distance_factor,
                    knn_k=opt.floater_knn_k,
                    isolation_scale=opt.floater_isolation_scale,
                    detail_scale=opt.floater_detail_scale,
                    color_std_threshold=opt.floater_color_std,
                    sh_energy_threshold=opt.floater_sh_energy_threshold,
                    sample_ratio=opt.floater_knn_sample_ratio
                )
                # cap only if >0 (0 = unlimited/adaptive)
                if getattr(opt, "max_gaussians", 0) and opt.max_gaussians > 0:
                    gauss.cap_gaussian_count(opt.max_gaussians)

        # save/eval
        if it in save_iters:
            scene.save(it)
            dump_gaussians_to_npz(scene, gauss, it)
        do_eval = (it in test_iters) or (opt.eval_interval > 0 and it % opt.eval_interval == 0)
        if do_eval:
            row = validation_loop(it, scene, gauss, pipe, bg, depth_est)
            # model size (best-effort)
            size_mb = 0.0
            if it in save_iters:
                ply = os.path.join(scene.model_path, "point_cloud", f"iteration_{it}", "point_cloud.ply")
                if os.path.exists(ply): size_mb = os.path.getsize(ply)/(1024*1024)
            pruned_iter = gauss.event_counters.get('Pruned(Densify)_Iter',0) + gauss.event_counters.get('Pruned(Final)_Iter',0)
            pruned_cum  = gauss.event_counters.get('Pruned(Densify)_Cum',0)  + gauss.event_counters.get('Pruned(Final)_Cum',0)
            row.update({
                'Iter': it,
                'Train/L1': l1v.item(),
                'Train/TotalLoss': total.item(),
                'Train/SSIM': 1.0 - dssim.item(),
                'Train/MultiScale': float(multiscale_loss.item()) if multiscale_loss.numel() else 0.0,
                'Train/Grad': float(grad_loss.item()) if grad_loss.numel() else 0.0,
                'Train/ColorMoment': float(moment_loss.item()) if moment_loss.numel() else 0.0,
                'Train/LPIPS': float(lpips_metric.item()),
                'Train/Depth': float(depth_loss.item()) if depth_loss.numel() else 0.0,
                **gauss.event_counters,
                'Pruned(Total)_Iter': pruned_iter,
                'Pruned(Total)_Cum': pruned_cum,
                'TotalGaussians': gauss.get_xyz.shape[0],
                'Model_Size(MB)': f"{size_mb:.2f}" if size_mb>0 else "N/A"
            })
            csv_writer.writerow({k: row.get(k,'N/A') for k in csv_writer.fieldnames})
            csv_file.flush()
            gauss.reset_iter_counters()
        if it in ckpt_iters:
            torch.save((gauss.capture(), it), scene.model_path + f"/chkpnt{it}.pth")
        bar.update(1)

    bar.close(); csv_file.close(); print("[Train] complete.")

def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", datetime.now().strftime("Model_%Y%m%d_%H%M%S"))
    print("Output folder:", args.model_path)
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))
    csv_path = os.path.join(args.model_path, 'evaluation_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    header = [
        'Iter','Val/PSNR','Val/SSIM','Val/LPIPS','Val/L1','Val/DepthAbsRel','Val/DepthDelta1.25',
        'Train/L1','Train/TotalLoss','Train/SSIM','Train/MultiScale','Train/Grad','Train/ColorMoment','Train/LPIPS','Train/Depth',
        'RenderFPS','IterTime(ms)',
        'Added(Densify)_Iter','Pruned(Densify)_Iter','Pruned(Final)_Iter',
        'Added(Densify)_Cum','Pruned(Densify)_Cum','Pruned(Final)_Cum',
        'Pruned(Total)_Iter','Pruned(Total)_Cum',
        'TotalGaussians','Model_Size(MB)'
    ]
    writer = csv.DictWriter(csv_file, fieldnames=header); writer.writeheader()
    tb = SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None
    return tb, writer, csv_file
    
# ----------------------- NPZ dump helper -----------------------
def dump_gaussians_to_npz(scene, gauss, iteration):
    """
    scene.model_path/npz/iteration_<it>/snapshot_iter<it>.npz 로 저장.
    이후 시각화 스크립트(plot_omega_vs_alpha.py, visualize_pruning_pipeline.py)가 그대로 읽습니다.
    """
    import os
    out_dir = os.path.join(scene.model_path, "npz", f"iteration_{iteration}")
    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, f"snapshot_iter{iteration}.npz")

    with torch.no_grad():
        xyz    = gauss.get_xyz.detach().cpu().numpy()                         # (N,3)
        alpha  = gauss.get_opacity.detach().cpu().squeeze(-1).numpy()         # (N,)
        omega  = gauss.get_importance.detach().cpu().squeeze(-1).numpy()      # (N,)
        scale  = gauss.get_scaling.detach().cpu().numpy()                     # (N,3)
        # grad_pos_ema: (N,3)→노름으로 축약(시각화 스크립트 호환)
        if gauss._grad_ema_pos is not None and gauss._grad_ema_pos.numel() > 0:
            grad_pos_ema = torch.norm(gauss._grad_ema_pos, dim=1).detach().cpu().numpy()   # (N,)
        else:
            grad_pos_ema = np.zeros((xyz.shape[0],), dtype=np.float32)
        # visibility
        vis_count = gauss._visibility_count.detach().cpu().numpy() if gauss._visibility_count.numel() > 0 \
                    else np.zeros((xyz.shape[0],), dtype=np.float32)
        # SH(rest) 전개 (DC 제외, 이미 (C,…) 텐서라 flatten)
        if gauss._features_rest is not None and gauss._features_rest.numel() > 0:
            sh_rest = gauss._features_rest.detach().cpu().reshape(xyz.shape[0], -1).numpy()
        else:
            sh_rest = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        # RGB 평균(선택): DC를 컬러 근사로 사용
        if gauss._features_dc is not None and gauss._features_dc.numel() > 0:
            rgb_mean = gauss._features_dc.detach().cpu().squeeze(1).numpy()   # (N,3)
        else:
            rgb_mean = np.zeros((xyz.shape[0], 3), dtype=np.float32)

        # scene extent
        E_scene = float(gauss.scene_extent) if getattr(gauss, "scene_extent", None) else float(np.max(scale))

        np.savez(
            out_npz,
            xyz=xyz,
            alpha=alpha,
            omega=omega,
            grad_pos_ema=grad_pos_ema,
            vis_count=vis_count,
            sh_rest=sh_rest,
            rgb_mean=rgb_mean,
            scale=scale,
            E_scene=E_scene
        )
    print(f"[NPZ] saved → {out_npz}")


# ======================= MAIN =======================
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")

    # Grouped argument packs
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # General / IO
    safe_add_argument(parser, '--ip', type=str, default="127.0.0.1")
    safe_add_argument(parser, '--port', type=int, default=6009)
    safe_add_argument(parser, '--debug_from', type=int, default=-1)
    safe_add_argument(parser, '--detect_anomaly', action='store_true', default=False)
    safe_add_argument(parser, "--test_iterations", nargs="+", type=int, default=[7000, 15000, 30000])
    safe_add_argument(parser, "--save_iterations", nargs="+", type=int, default=[7000, 15000, 30000])
    safe_add_argument(parser, "--quiet", action="store_true")
    safe_add_argument(parser, "--checkpoint_iterations", nargs="+", type=int, default=[])
    safe_add_argument(parser, "--start_checkpoint", type=str, default=None)

    # Loss / regularization
    safe_add_argument(parser, "--lambda_depth", type=float, default=0.1)
    safe_add_argument(parser, "--lambda_lpips", type=float, default=0.1)
    safe_add_argument(parser, "--lambda_importance", type=float, default=0.01)
    safe_add_argument(parser, "--uncertainty_tau", type=float, default=0.1)
    safe_add_argument(parser, "--cleanup_interval", type=int, default=200)

    # Floater cleanup scheduling
    safe_add_argument(parser, "--floater_cleanup_start", type=int, default=15000)
    safe_add_argument(parser, "--floater_cleanup_interval", type=int, default=400)
    safe_add_argument(parser, "--floater_visibility_threshold", type=int, default=2)
    safe_add_argument(parser, "--floater_grad_threshold", type=float, default=5e-4)
    safe_add_argument(parser, "--floater_opacity_threshold", type=float, default=0.04)
    safe_add_argument(parser, "--floater_importance_threshold", type=float, default=0.35)
    safe_add_argument(parser, "--floater_cleanup_max_ratio", type=float, default=0.05)
    safe_add_argument(parser, "--floater_distance_factor", type=float, default=1.4)

    # NEW: per-iter densification limits (0 = unlimited)
    safe_add_argument(parser, "--max_clone_points_per_iter", type=int, default=0)
    safe_add_argument(parser, "--max_split_points_per_iter", type=int, default=0)
    safe_add_argument(parser, "--log_densify_clamps", action='store_true', default=False)

    # NEW: adaptive cap (0 = disabled)
    safe_add_argument(parser, "--max_gaussians", type=int, default=0)

    # Eval cadence
    safe_add_argument(parser, "--eval_interval", type=int, default=200)

    # SDRL (self-distilled residual loss) — safe added
    safe_add_argument(parser, "--enable_sdrl", action="store_true", default=True)
    safe_add_argument(parser, "--sdrl_lambda", type=float, default=0.05)
    safe_add_argument(parser, "--sdrl_alpha", type=float, default=0.95)
    safe_add_argument(parser, "--sdrl_update_every", type=int, default=500)
    safe_add_argument(parser, "--sdrl_lowres_factor", type=int, default=4)

    args = parser.parse_args(sys.argv[1:])

    # Hardcode your preferred defaults (won’t error if already set)
    args.lambda_lpips = getattr(args, "lambda_lpips", 0.1) or 0.1
    args.enable_sdrl = True
    args.sdrl_lambda = 0.05
    args.sdrl_lowres_factor = 4
    args.sdrl_update_every = 500
    args.sdrl_alpha = 0.95

    # Ensure we evaluate/save at the last iteration
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    print("Optimizing", args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset_params = lp.extract(args)
    opt_params = op.extract(args)
    pipe_params = pp.extract(args)

    # Map CLI → opt
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

    opt_params.max_clone_points_per_iter = args.max_clone_points_per_iter
    opt_params.max_split_points_per_iter = args.max_split_points_per_iter
    opt_params.log_densify_clamps = args.log_densify_clamps
    opt_params.max_gaussians = args.max_gaussians

    opt_params.eval_interval = args.eval_interval

    # Run
    training(dataset_params, opt_params, pipe_params,
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from)
