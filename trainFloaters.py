# === train_unified.py (30k fixed, multi-res, boosted densify, depth-first region refine, floater cleanup, depth eval, FLAT CSV, L2-schedule transplanted, no image save, no per-iter metric print, metrics printed only at 29800) ===
import csv
import glob
import math
import os
import sys
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime
from random import randint, seed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

# ----------------------- LPIPS -----------------------
try:
    import lpips

    _lpips = lpips.LPIPS(net="alex")
    if torch.cuda.is_available():
        _lpips = _lpips.cuda()
    _lpips.eval()
    _HAS_LPIPS = True
except Exception:
    _lpips = None
    _HAS_LPIPS = False

# ----------------------- TensorBoard -----------------------
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# ----------------------- debug-line suppress -----------------------
from contextlib import contextmanager


@contextmanager
def suppress_debug_lines(prefixes=("DEBUG:",)):
    class _Filter:
        def __init__(self, orig):
            self._orig = orig

        def write(self, s):
            text = s
            for prefix in prefixes:
                while True:
                    idx = text.find(prefix)
                    if idx == -1:
                        break
                    if idx > 0 and text[idx - 1] == "\r":
                        idx -= 1
                    end = text.find("\n", idx)
                    if end == -1:
                        text = text[:idx]
                    else:
                        text = text[:idx] + text[end + 1 :]
            if text:
                self._orig.write(text)
            return len(s)

        def flush(self):
            return self._orig.flush()

    _orig = sys.stdout
    sys.stdout = _Filter(_orig)
    try:
        yield
    finally:
        sys.stdout = _orig


# ----------------------- small utils -----------------------
def _ensure_batch_dim(t: torch.Tensor) -> torch.Tensor:
    return t if t.dim() == 4 else t.unsqueeze(0)


def align_tensor_to_length(tensor: torch.Tensor, target_len: int):
    if tensor.shape[0] == target_len:
        return tensor
    if tensor.shape[0] > target_len:
        return tensor[:target_len]
    pad_shape = (target_len - tensor.shape[0],) + tuple(tensor.shape[1:])
    pad = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=0)


def to_chw01(img: torch.Tensor) -> torch.Tensor:
    if img.ndim == 2:
        return img.view(1, 1, *img.shape).repeat(1, 3, 1, 1).float()
    if img.ndim == 3:
        if img.shape[-1] in (3, 4):
            return img[..., :3].permute(2, 0, 1).unsqueeze(0).float().contiguous()
        if img.shape[0] in (3, 4):
            return img[:3].unsqueeze(0).float().contiguous()
    if img.ndim == 4:
        if img.shape[-1] in (3, 4):
            return img[..., :3].permute(0, 3, 1, 2).float().contiguous()
        if img.shape[1] in (3, 4):
            return img[:, :3].float().contiguous()
    raise RuntimeError(f"Unsupported image shape: {tuple(img.shape)}")


def safe_add_argument(parser: ArgumentParser, *flags, **kwargs):
    existing = parser._option_string_actions  # type: ignore[attr-defined]
    if any(f in existing for f in flags):
        return
    parser.add_argument(*flags, **kwargs)


def _append_if_finite(lst, v):
    if isinstance(v, (float, int)) and math.isfinite(v):
        lst.append(float(v))


# ----------------------- Depth (MiDaS) -----------------------
class DepthEstimator:
    def __init__(self, cache_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        self.available = True
        try:
            self.model = (
                torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(self.device).eval()
            )
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.dpt_transform
        except Exception as exc:
            print(f"[Depth] MiDaS unavailable: {exc} (depth loss disabled)")
            self.available, self.model, self.transform = False, None, None

    def _p(self, image_name, suffix):
        return os.path.join(
            self.cache_dir, os.path.basename(image_name) + f"_{suffix}.pt"
        )

    @torch.no_grad()
    def _estimate(self, img_numpy):
        x = self.transform(img_numpy).to(self.device)
        pred = self.model(x)
        return F.interpolate(
            pred.unsqueeze(1),
            size=img_numpy.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    def precompute(self, cams):
        if not self.available:
            return
        from torchvision.transforms.functional import hflip, to_pil_image

        for cam in tqdm(cams, desc="[Depth] Precompute"):
            dp, up = (
                self._p(cam.image_name, "depth"),
                self._p(cam.image_name, "uncertainty"),
            )
            if os.path.exists(dp) and os.path.exists(up):
                continue
            pil = to_pil_image(cam.original_image.cpu())
            img = np.array(pil)
            img_f = np.array(hflip(pil))
            d1 = self._estimate(img)
            d2 = torch.fliplr(self._estimate(img_f))
            mean = torch.mean(torch.stack([d1, d2]), dim=0)
            var = torch.var(torch.stack([d1, d2]), dim=0)
            mean = (mean - mean.min()) / (mean.max() - mean.min() + 1e-6)
            var = (var - var.min()) / (var.max() - var.min() + 1e-6)
            torch.save(mean.cpu(), dp)
            torch.save(var.cpu(), up)

    def load(self, cam):
        if not self.available:
            return None, None
        dp, up = (
            self._p(cam.image_name, "depth"),
            self._p(cam.image_name, "uncertainty"),
        )
        if os.path.exists(dp) and os.path.exists(up):
            return torch.load(dp).to(self.device), torch.load(up).to(self.device)
        return None, None


# ----------------------- Refined Gaussian Model -----------------------
class RefinedGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int, verbose: bool = True):
        super().__init__(sh_degree)
        # region-aware params
        self.grid_size = 20
        self.std_threshold_factor = 0.125
        self.opacity_threshold = 0.015
        self.merge_weights = None
        self.verbose = verbose
        self.sparse_quantile = 0.30
        self.dense_quantile = 0.80
        self.prune_percent_sparse = 0.003
        self.prune_percent_mid = 0.006
        self.prune_percent_dense = 0.02
        self.merge_scale_sparse = 0.8
        self.merge_scale_mid = 1.0
        self.merge_scale_dense = 1.5
        self.min_points_per_region = 256
        self.max_refine_prune_fraction = 0.02

        # depth-first region clustering용 평균 카메라 중심
        self._mean_cam_center = None

        # densify/floater/adaptive
        self.tmp_radii = None
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
        self._visibility_count = torch.empty(0)
        self._grad_ema_pos = torch.empty(0)
        self._grad_ema_scale = torch.empty(0)
        self._grad_ema_rot = torch.empty(0)

    def register_scene_extent(self, extent):
        self.scene_extent = extent

    def reset_iter_counters(self):
        for k in list(self.event_counters.keys()):
            if k.endswith("_Iter"):
                self.event_counters[k] = 0

    # ------------------------ camera stats for depth-first clustering ------------------------
    def register_cameras(self, cams):
        """
        트레이닝/테스트 카메라들을 받아서 평균 카메라 중심(self._mean_cam_center)을 저장.
        """
        if not cams:
            return
        dev = self._xyz.device if self._xyz.numel() > 0 else torch.device("cuda")
        centers = []
        for c in cams:
            try:
                centers.append(c.camera_center.detach().to(dev))
            except Exception:
                pass
        if len(centers) == 0:
            return
        self._mean_cam_center = torch.stack(centers, dim=0).mean(dim=0)  # [3]

    def training_setup(self, training_args):
        self.training_args = training_args
        super().training_setup(training_args)
        n = self.get_xyz.shape[0]
        dev = self.get_xyz.device
        if self._importance is None or self._importance.shape[0] != n:
            self._importance = nn.Parameter(torch.ones(n, 1, device=dev) * 2.0)
        if self._visibility_count.numel() == 0 or self._visibility_count.shape[0] != n:
            self._visibility_count = torch.zeros(n, device=dev)
            self._grad_ema_pos = torch.zeros((n, 3), device=dev)
            self._grad_ema_scale = torch.zeros((n, 3), device=dev)
            self._grad_ema_rot = torch.zeros((n, 4), device=dev)
        found = False
        for g in self.optimizer.param_groups:
            if g.get("name") == "importance":
                g["params"] = [self._importance]
                found = True
        if not found:
            self.optimizer.add_param_group(
                {
                    "params": [self._importance],
                    "lr": training_args.feature_lr * 0.1,
                    "name": "importance",
                }
            )

    @property
    def get_importance(self):
        return self._importance

    # ---- helper: ensure buffer sizes ----
    def _ensure_point_buffers(self):
        n = self.get_xyz.shape[0]
        dev = self.get_xyz.device

        def _align(t, fill=0.0):
            if t.shape[0] == n:
                return t
            if t.shape[0] > n:
                return t[:n]
            pad = torch.zeros(
                (n - t.shape[0],) + t.shape[1:], device=dev, dtype=t.dtype
            )
            return torch.cat([t, pad], dim=0)

        self.xyz_gradient_accum = _align(self.xyz_gradient_accum)
        self.denom = _align(self.denom)
        self.max_radii2D = _align(self.max_radii2D)
        if self.tmp_radii is None:
            self.tmp_radii = torch.zeros(n, device=dev)
        else:
            if self.tmp_radii.device != dev:
                self.tmp_radii = self.tmp_radii.to(dev)
            if self.tmp_radii.shape[0] != n:
                new_tmp = torch.zeros(n, device=dev)
                c = min(self.tmp_radii.shape[0], n)
                if c > 0:
                    new_tmp[:c] = self.tmp_radii[:c]
                self.tmp_radii = new_tmp

    # ---- densify caps ----
    def _limit_selected_points(self, selected_mask, scores, max_points):
        if max_points <= 0:
            return selected_mask
        cnt = int(selected_mask.sum().item())
        if cnt <= max_points:
            return selected_mask
        idx = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
        k = min(max_points, idx.numel())
        if scores is None:
            chosen = idx[torch.randperm(idx.numel(), device=idx.device)[:k]]
        else:
            s = torch.nan_to_num(scores[idx], nan=0.0, posinf=0.0, neginf=0.0)
            _, order = torch.topk(s, k=k, largest=True)
            chosen = idx[order]
        m = torch.zeros_like(selected_mask)
        m[chosen] = True
        return m

    # ---- counters on add/prune ----
    def densify_and_add(self, props):
        n = props["xyz"].shape[0]
        self.event_counters["Added(Densify)_Iter"] += n
        self.event_counters["Added(Densify)_Cum"] += n
        super().densify_and_add(props)

    def _ensure_grad_length(self, grads: torch.Tensor) -> torch.Tensor:
        target = self.get_xyz.shape[0]
        if grads.shape[0] == target:
            return grads
        if grads.shape[0] > target:
            return grads[:target]
        pad_shape = (target - grads.shape[0],) + grads.shape[1:]
        pad = grads.new_zeros(pad_shape)
        return torch.cat([grads, pad], dim=0)

    def _ensure_aux_lengths(self):
        n = self.get_xyz.shape[0]
        dev = self._xyz.device if self._xyz.numel() else torch.device("cuda")

        def _match(t, shape_tail, fill_value=0.0):
            if t is None:
                base = torch.zeros((n,) + shape_tail, device=dev)
                return base if isinstance(fill_value, float) else fill_value
            if t.shape[0] == n:
                return t
            if t.shape[0] > n:
                return t[:n]
            pad_shape = (n - t.shape[0],) + t.shape[1:]
            pad = torch.zeros(pad_shape, device=dev, dtype=t.dtype)
            return torch.cat([t, pad], dim=0)

        self.max_radii2D = _match(self.max_radii2D, (), 0.0)
        self.tmp_radii = _match(self.tmp_radii, (), 0.0)
        self.xyz_gradient_accum = _match(
            self.xyz_gradient_accum,
            (self.xyz_gradient_accum.shape[1],)
            if self.xyz_gradient_accum is not None
            else (1,),
        )
        self.denom = _match(
            self.denom,
            (self.denom.shape[1],) if self.denom is not None else (1,),
        )
        self._visibility_count = _match(self._visibility_count, (), 0.0)
        self._grad_ema_pos = _match(self._grad_ema_pos, (3,), 0.0)
        self._grad_ema_scale = _match(self._grad_ema_scale, (3,), 0.0)
        self._grad_ema_rot = _match(self._grad_ema_rot, (4,), 0.0)
        if self._importance is not None:
            self._ensure_importance_length(n)

    def _ensure_importance_length(self, target: int):
        dev = self._xyz.device if self._xyz.numel() else torch.device("cuda")
        if target <= 0:
            data = torch.ones(0, 1, device=dev)
        else:
            if self._importance is None:
                data = torch.ones(target, 1, device=dev) * 2.0
            else:
                cur = self._importance.detach()
                if cur.shape[0] == target:
                    return
                if cur.shape[0] > target:
                    data = cur[:target]
                else:
                    pad = (
                        torch.ones(
                            target - cur.shape[0], 1, device=dev, dtype=cur.dtype
                        )
                        * 2.0
                    )
                    data = torch.cat([cur, pad], dim=0)

        new_param = nn.Parameter(data, requires_grad=True)

        def _resize_state(tensor, shape):
            if tensor is None:
                return torch.zeros(shape, device=dev, dtype=data.dtype)
            if tensor.shape == shape:
                return tensor.detach().clone()
            if tensor.shape[0] >= shape[0]:
                return tensor.detach().clone()[: shape[0]]
            pad_shape = (shape[0] - tensor.shape[0],) + tuple(tensor.shape[1:])
            pad = torch.zeros(pad_shape, device=dev, dtype=tensor.dtype)
            return torch.cat([tensor.detach().clone(), pad], dim=0)

        for group in self.optimizer.param_groups:
            if group.get("name") == "importance":
                old_param = group["params"][0]
                old_state = self.optimizer.state.pop(old_param, None)
                group["params"][0] = new_param

                exp_avg = None
                exp_avg_sq = None
                step_val = 0.0
                if old_state is not None:
                    exp_avg = old_state.get("exp_avg")
                    exp_avg_sq = old_state.get("exp_avg_sq")
                    step_old = old_state.get("step", 0.0)
                    if isinstance(step_old, torch.Tensor):
                        step_val = step_old.detach().item()
                    else:
                        step_val = float(step_old)

                exp_avg = _resize_state(exp_avg, new_param.shape)
                exp_avg_sq = _resize_state(exp_avg_sq, new_param.shape)
                self.optimizer.state[new_param] = {
                    "exp_avg": exp_avg,
                    "exp_avg_sq": exp_avg_sq,
                    "step": torch.tensor(step_val, device=dev),
                }
                break
        self._importance = new_param

    def prune_points(self, mask, source="densify"):
        self._ensure_aux_lengths()
        if mask.shape[0] != self.get_xyz.shape[0]:
            mask = mask[: self.get_xyz.shape[0]]
        num = mask.sum().item()
        if source == "densify":
            self.event_counters["Pruned(Densify)_Iter"] += num
            self.event_counters["Pruned(Densify)_Cum"] += num
        else:
            self.event_counters["Pruned(Final)_Iter"] += num
            self.event_counters["Pruned(Final)_Cum"] += num
        # shrink aux arrays
        if self._visibility_count.numel() > 0:
            self._visibility_count = self._visibility_count[~mask]
        if self._grad_ema_pos.numel() > 0:
            self._grad_ema_pos = self._grad_ema_pos[~mask]
        if self._grad_ema_scale.numel() > 0:
            self._grad_ema_scale = self._grad_ema_scale[~mask]
        if self._grad_ema_rot.numel() > 0:
            self._grad_ema_rot = self._grad_ema_rot[~mask]
        # tmp_radii safety
        if self.tmp_radii is None:
            self.tmp_radii = torch.zeros(mask.shape[0], device=self.get_xyz.device)
        elif self.tmp_radii.shape[0] != mask.shape[0]:
            need = mask.shape[0] - self.tmp_radii.shape[0]
            if need > 0:
                self.tmp_radii = torch.cat(
                    [self.tmp_radii, torch.zeros(need, device=self.tmp_radii.device)]
                )
            else:
                self.tmp_radii = self.tmp_radii[: mask.shape[0]]
        super().prune_points(mask)
        self._ensure_importance_length(self.get_xyz.shape[0])

    # ---- clone/split with caps ----
    def densify_and_clone(self, grads, th, extent):
        grads = self._ensure_grad_length(grads)
        self._ensure_importance_length(self.get_xyz.shape[0])
        gnorm = torch.norm(grads, dim=-1)
        scaling_ok = (
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * extent
        )
        candidate = torch.logical_and(gnorm >= th, scaling_ok)
        sel = candidate
        cap = int(getattr(self.training_args, "max_clone_points_per_iter", 0) or 0)
        if cap > 0:
            sel = self._limit_selected_points(candidate, gnorm, cap)
        disallowed = candidate & ~sel
        if disallowed.any():
            grads = grads.clone()
            idx = torch.nonzero(disallowed, as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                grads[idx[idx < grads.shape[0]]] = 0.0
        if int(sel.sum().item()) > 0:
            self._pending_new_importance = self._importance[sel].detach().clone()
        super().densify_and_clone(grads, th, extent)
        self._pending_new_importance = None

    def densify_and_split(self, grads, th, extent, N=2):
        grads = self._ensure_grad_length(grads)
        n0 = self.get_xyz.shape[0]
        self._ensure_importance_length(n0)
        pad = torch.zeros((n0), device=self.get_xyz.device)
        pad[: grads.shape[0]] = grads.squeeze()
        scale_mask = (
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * extent
        )
        candidate = torch.logical_and(pad >= th, scale_mask)
        sel = candidate
        cap = int(getattr(self.training_args, "max_split_points_per_iter", 0) or 0)
        if cap > 0:
            seeds = max(1, cap // max(N, 1))
            sel = self._limit_selected_points(candidate, pad, seeds)
        disallowed = candidate & ~sel
        if disallowed.any():
            grads = grads.clone()
            idx = torch.nonzero(disallowed, as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                grads[idx[idx < grads.shape[0]]] = 0.0
        base = int(sel.sum().item())
        if base > 0:
            self._pending_new_importance = (
                self._importance[sel].detach().clone().repeat(N, 1)
            )
        super().densify_and_split(grads, th, extent, N=N)
        self._pending_new_importance = None

    def densification_postfix(
        self, new_xyz, f_dc, f_rest, opac, scaling, rot, new_tmp_radii
    ):
        new_importance = (
            self._pending_new_importance.to(new_xyz.device)
            if self._pending_new_importance is not None
            else torch.ones(new_xyz.shape[0], 1, device=new_xyz.device) * 2.0
        )
        d = {
            "xyz": new_xyz,
            "f_dc": f_dc,
            "f_rest": f_rest,
            "opacity": opac,
            "scaling": scaling,
            "rotation": rot,
            "importance": new_importance,
        }
        optim = self.cat_tensors_to_optimizer(d)
        self._xyz, self._features_dc, self._features_rest = (
            optim["xyz"],
            optim["f_dc"],
            optim["f_rest"],
        )
        self._opacity, self._scaling, self._rotation = (
            optim["opacity"],
            optim["scaling"],
            optim["rotation"],
        )
        self._importance = optim["importance"]
        if self.tmp_radii is None:
            self.tmp_radii = new_tmp_radii
        else:
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
            if self.tmp_radii.shape[0] != self.get_xyz.shape[0]:
                need = self.get_xyz.shape[0] - self.tmp_radii.shape[0]
                if need > 0:
                    self.tmp_radii = torch.cat(
                        [
                            self.tmp_radii,
                            torch.zeros(need, device=self.tmp_radii.device),
                        ]
                    )
                else:
                    self.tmp_radii = self.tmp_radii[: self.get_xyz.shape[0]]
        dev = self._xyz.device
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=dev)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=dev)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=dev)
        n_new = new_xyz.shape[0]
        if n_new > 0:
            self._visibility_count = torch.cat(
                (self._visibility_count, torch.zeros(n_new, device=dev))
            )
            self._grad_ema_pos = torch.cat(
                (self._grad_ema_pos, torch.zeros((n_new, 3), device=dev))
            )
            self._grad_ema_scale = torch.cat(
                (self._grad_ema_scale, torch.zeros((n_new, 3), device=dev))
            )
            self._grad_ema_rot = torch.cat(
                (self._grad_ema_rot, torch.zeros((n_new, 4), device=dev))
            )

    # ---- densify_and_prune (boosted, + HF-protect) ----
    def _identify_high_frequency_regions(self, grads):
        n = self.get_xyz.shape[0]
        if n == 0:
            d = self.get_xyz.device if self.get_xyz.numel() else "cuda"
            return torch.zeros(0, dtype=torch.bool, device=d), torch.zeros(0, device=d)
        dev = self.get_xyz.device
        grad_ema = (
            self._grad_ema_pos
            if self._grad_ema_pos.shape[0] == n
            else torch.zeros((n, 3), device=dev)
        )
        grad_norm = torch.norm(grad_ema, dim=1)
        sh_energy = torch.zeros(n, device=dev)
        if self._features_rest is not None and self._features_rest.numel() > 0:
            sh_energy = torch.norm(self._features_rest.reshape(n, -1), dim=1)
        min_scale = self.get_scaling.min(dim=1).values
        grad_norm_norm = grad_norm / (torch.max(grad_norm) + 1e-6)
        sh_norm = sh_energy / (torch.max(sh_energy) + 1e-6)
        target_min_scale = getattr(self.training_args, "high_freq_min_scale", 0.02)
        scale_score = torch.clamp(
            (target_min_scale - min_scale) / (target_min_scale + 1e-6), 0.0, 1.0
        )
        score = 0.5 * sh_norm + 0.3 * grad_norm_norm + 0.2 * scale_score
        hf_mask = (
            (sh_norm >= getattr(self.training_args, "high_freq_sh_threshold", 0.2))
            | (
                grad_norm_norm
                >= getattr(self.training_args, "high_freq_grad_threshold", 0.35)
            )
            | (scale_score > 0.0)
        )
        return hf_mask, score

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        self._ensure_aux_lengths()
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self._ensure_importance_length(self.get_xyz.shape[0])
        hf_mask, hf_score = self._identify_high_frequency_regions(grads)
        boosted = grads.clone()
        b = getattr(self.training_args, "high_freq_grad_boost", 1.5)
        if b > 1.0 and hf_mask.any():
            boosted[hf_mask] = boosted[hf_mask] * b
        split_factor = max(
            2, int(getattr(self.training_args, "high_freq_split_factor", 2))
        )
        split_th = max_grad * 0.85
        self.tmp_radii = radii

        # 2× 호출 (1.0, 0.9)
        self.densify_and_clone(boosted * 1.0, max_grad, extent)
        self.densify_and_split(boosted * 1.0, split_th, extent, N=split_factor)
        self.densify_and_clone(boosted * 0.9, max_grad, extent)
        self.densify_and_split(boosted * 0.9, split_th, extent, N=split_factor)

        post_grads = self.xyz_gradient_accum / self.denom
        post_grads[post_grads.isnan()] = 0.0
        hf_mask, hf_score = self._identify_high_frequency_regions(post_grads)

        prune_mask = self.get_opacity.squeeze(-1) < min_opacity
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

    # ------------------------ Region labeling & refine (depth-first) ------------------------
    def set_dynamic_thresholds(self):
        """
        가능하면 카메라 중심 기준 depth 통계,
        아니면 XYZ grid 기반 fallback.
        """
        if self.get_xyz.shape[0] == 0:
            return

        # depth 기반
        if self._mean_cam_center is not None:
            dev = self.get_xyz.device
            center = self._mean_cam_center.to(dev)
            depths = torch.norm(self.get_xyz - center, dim=1)
            if depths.numel() == 0:
                return
            m = depths.mean()
            s = depths.std()
            self.std_threshold_factor = max(0.2, (2.0 * (s / (m + 1e-6))).item())
            if self.verbose:
                dmin, dmax = depths.min().item(), depths.max().item()
                print(
                    f"[DepthThresh] mean={m.item():.3f}, std={s.item():.3f}, "
                    f"min={dmin:.3f}, max={dmax:.3f}, "
                    f"std_factor={self.std_threshold_factor:.3f}"
                )
            return

        # fallback: XYZ grid
        num_points = self.get_xyz.shape[0]
        min_xyz = self.get_xyz.min(dim=0).values
        max_xyz = self.get_xyz.max(dim=0).values
        spatial_extent = torch.max(max_xyz - min_xyz).item()
        target_points_per_cell = 1000
        initial_grid_size = max(
            4,
            min(
                64,
                int((num_points / max(target_points_per_cell, 1)) ** (1.0 / 3.0)),
            ),
        )
        denom = max(float(self.scene_extent or 1.0), 1e-6)
        extent_norm = spatial_extent / denom
        self.grid_size = max(
            4,
            min(64, int(initial_grid_size * max(extent_norm, 1e-6))),
        )
        G = int(self.grid_size)
        step = torch.clamp((max_xyz - min_xyz) / max(G, 1), min=1e-6)
        grid_idx = ((self.get_xyz - min_xyz) / step).floor().long().clamp(0, G - 1)
        lin = grid_idx[:, 0] * (G * G) + grid_idx[:, 1] * G + grid_idx[:, 2]
        counts = torch.zeros(G * G * G, device=self.get_xyz.device, dtype=torch.float32)
        w = (
            self.merge_weights
            if self.merge_weights is not None
            else torch.ones_like(lin, dtype=torch.float32, device=self.get_xyz.device)
        )
        counts.scatter_add_(0, lin, w)
        nz = counts.float()
        m, s = nz.mean(), nz.std()
        self.std_threshold_factor = max(0.2, (2.0 * (s / max(m, 1e-6))).item())

    def _region_labels(self):
        """
        depth-first region labeling.
        """
        N = self.get_xyz.shape[0]
        if N == 0:
            return None, None, None

        dev = self.get_xyz.device

        # depth 기반
        if self._mean_cam_center is not None:
            center = self._mean_cam_center.to(dev)
            depths = torch.norm(self.get_xyz - center, dim=1)  # [N]

            dmin, dmax = depths.min(), depths.max()
            if (dmax - dmin) < 1e-6:
                mid = torch.ones(N, dtype=torch.int8, device=dev)
                lin = torch.zeros(N, dtype=torch.long, device=dev)
                counts = torch.ones(1, dtype=torch.float32, device=dev) * float(N)
                return mid, lin, counts

            num_bins = 16
            norm = (depths - dmin) / (dmax - dmin + 1e-6)
            bins = torch.clamp((norm * num_bins).long(), 0, num_bins - 1)

            counts = torch.zeros(num_bins, device=dev, dtype=torch.float32)
            w = (
                self.merge_weights
                if self.merge_weights is not None
                else torch.ones_like(bins, dtype=torch.float32, device=dev)
            )
            counts.scatter_add_(0, bins, w)

            nz = counts[counts > 0]
            if nz.numel() == 0:
                mid = torch.ones(N, dtype=torch.int8, device=dev)
                lin = bins
                return mid, lin, counts

            q_sparse = torch.quantile(nz, self.sparse_quantile)
            q_dense = torch.quantile(nz, self.dense_quantile)

            cell_class = torch.ones_like(counts, dtype=torch.int8)
            cell_class[counts <= q_sparse] = 0
            cell_class[counts >= q_dense] = 2

            region_labels = cell_class[bins]
            lin = bins
            return region_labels, lin, counts

        # fallback: XYZ grid
        min_xyz = self.get_xyz.min(dim=0).values
        max_xyz = self.get_xyz.max(dim=0).values
        G = int(self.grid_size)
        step = torch.clamp((max_xyz - min_xyz) / max(G, 1), min=1e-6)
        grid_idx = ((self.get_xyz - min_xyz) / step).floor().long().clamp(0, G - 1)
        lin = grid_idx[:, 0] * (G * G) + grid_idx[:, 1] * G + grid_idx[:, 2]

        counts = torch.zeros(G * G * G, device=dev, dtype=torch.float32)
        w = (
            self.merge_weights
            if self.merge_weights is not None
            else torch.ones_like(lin, dtype=torch.float32, device=self.get_xyz.device)
        )
        counts.scatter_add_(0, lin, w)

        nz = counts[counts > 0]
        if nz.numel() == 0:
            mid = torch.ones(N, dtype=torch.int8, device=dev)
            return mid, lin, counts

        q_sparse = torch.quantile(nz, self.sparse_quantile)
        q_dense = torch.quantile(nz, self.dense_quantile)

        cell_class = torch.ones_like(counts, dtype=torch.int8)
        cell_class[counts <= q_sparse] = 0
        cell_class[counts >= q_dense] = 2

        region_labels = cell_class[lin]
        return region_labels, lin, counts

    @torch.no_grad()
    def _fast_merge_and_apply(
        self,
        region_labels,
        lin,
        counts,
        merge_threshold_xyz,
        merge_threshold_color,
        merge_threshold_scale,
        merge_cap_ratio=0.02,
    ):
        N = self.get_xyz.shape[0]
        if N < 2:
            return
        device = self.get_xyz.device
        M = 1 << 20
        rand_key = torch.randint(0, M, (N,), device=device, dtype=torch.long)
        key = lin.to(torch.long) * M + rand_key
        order = torch.argsort(key)
        lin_sorted = lin[order]
        unique_lin, counts_per_cell = torch.unique_consecutive(
            lin_sorted, return_counts=True
        )
        seg_ids = torch.repeat_interleave(
            torch.arange(unique_lin.numel(), device=device), counts_per_cell
        )
        idx_within = torch.arange(N, device=device) - torch.repeat_interleave(
            torch.cumsum(
                torch.cat([torch.tensor([0], device=device), counts_per_cell[:-1]]),
                dim=0,
            ),
            counts_per_cell,
        )
        posA_sorted = torch.nonzero(
            (idx_within % 2 == 0) & (idx_within + 1 < counts_per_cell[seg_ids]),
            as_tuple=False,
        ).squeeze(-1)
        if posA_sorted.numel() == 0:
            return
        A = order[posA_sorted]
        B = order[posA_sorted + 1]
        rA = region_labels[A]
        scale = torch.ones_like(rA, dtype=torch.float32)
        scale = torch.where(
            rA == 0,
            torch.full_like(scale, self.merge_scale_sparse),
            scale,
        )
        scale = torch.where(
            rA == 2,
            torch.full_like(scale, self.merge_scale_dense),
            scale,
        )
        dist = torch.norm(self.get_xyz[A] - self.get_xyz[B], dim=1)
        col = torch.norm(self._features_dc[A] - self._features_dc[B], dim=2).squeeze()
        sca = torch.norm(self._scaling[A] - self._scaling[B], dim=1)
        m = (
            (dist < (merge_threshold_xyz * scale))
            & (col < (merge_threshold_color * scale))
            & (sca < (merge_threshold_scale * scale))
        )
        if not torch.any(m):
            return
        merge_cap = max(1, int(merge_cap_ratio * N))
        idx_accepted = torch.nonzero(m, as_tuple=False).squeeze(-1)
        if idx_accepted.numel() > merge_cap:
            dsel = dist[idx_accepted]
            keep = torch.topk(-dsel, k=merge_cap, largest=True).indices
            idx_accepted = idx_accepted[keep]
        a = A[idx_accepted]
        b = B[idx_accepted]
        if a.numel() == 0:
            return
        new_xyz = 0.5 * (self._xyz[a] + self._xyz[b])
        new_f_dc = 0.5 * (self._features_dc[a] + self._features_dc[b])
        new_f_rest = 0.5 * (self._features_rest[a] + self._features_rest[b])
        new_opacity = torch.max(self._opacity[a], self._opacity[b])
        new_scaling = torch.max(self._scaling[a], self._scaling[b])
        new_rotation = self._rotation[a]
        if self._importance is not None and self._importance.shape[0] == N:
            new_importance = 0.5 * (self._importance[a] + self._importance[b])
        else:
            new_importance = torch.ones(new_xyz.shape[0], 1, device=device) * 2.0
        merged_mask = torch.zeros(N, dtype=torch.bool, device=device)
        merged_mask[a] = True
        merged_mask[b] = True
        old_r2d = self.max_radii2D.clone()
        self.prune_points(merged_mask)
        if self.max_radii2D.shape[0] != self._xyz.shape[0]:
            self.max_radii2D = old_r2d[~merged_mask][: self._xyz.shape[0]]
        new_max_r2d = torch.max(old_r2d[a], old_r2d[b])
        optim = self.cat_tensors_to_optimizer(
            {
                "xyz": new_xyz,
                "f_dc": new_f_dc,
                "f_rest": new_f_rest,
                "opacity": new_opacity,
                "scaling": new_scaling,
                "rotation": new_rotation,
                "importance": new_importance,
            }
        )
        self._xyz, self._features_dc, self._features_rest = (
            optim["xyz"],
            optim["f_dc"],
            optim["f_rest"],
        )
        self._opacity, self._scaling, self._rotation = (
            optim["opacity"],
            optim["scaling"],
            optim["rotation"],
        )
        if "importance" in optim:
            self._importance = optim["importance"]
        if self.max_radii2D.shape[0] < self._xyz.shape[0]:
            self.max_radii2D = torch.cat(
                [self.max_radii2D, torch.zeros(new_xyz.shape[0], device=device)],
                dim=0,
            )
        new_idx = torch.arange(
            self._xyz.shape[0] - new_xyz.shape[0],
            self._xyz.shape[0],
            device=device,
        )
        self.max_radii2D[new_idx] = new_max_r2d
        torch.cuda.empty_cache()

    def refine_and_prune(
        self,
        merge_threshold_xyz,
        merge_threshold_color,
        merge_threshold_scale,
        prune_percent,
        radii,
    ):
        if self.get_xyz.shape[0] == 0:
            return
        N = self.get_xyz.shape[0]
        if radii.shape[0] > N:
            radii = radii[:N]
        if self.max_radii2D.shape[0] != N:
            self.max_radii2D = torch.zeros(N, device=self.get_xyz.device)
        self.set_dynamic_thresholds()
        region_labels, lin, counts = self._region_labels()
        if region_labels is None:
            return
        with torch.no_grad():
            opacity_mask = self.get_opacity.squeeze(-1) < self.opacity_threshold
        if opacity_mask.any():
            opacity_mask = opacity_mask[: self.get_xyz.shape[0]]
            old = self.max_radii2D.clone()
            self.prune_points(opacity_mask)
            if self.max_radii2D.shape[0] != self.get_xyz.shape[0]:
                self.max_radii2D = old[~opacity_mask][: self.get_xyz.shape[0]]
            region_labels, lin, counts = self._region_labels()
            if region_labels is None:
                return
        importance = self.get_opacity.squeeze(-1) * self.get_scaling.mean(dim=1)
        prune_cfg = {
            0: self.prune_percent_sparse,
            1: prune_percent if prune_percent is not None else self.prune_percent_mid,
            2: self.prune_percent_dense,
        }
        global_prune_mask = torch.zeros(
            self.get_xyz.shape[0],
            dtype=torch.bool,
            device=self.get_xyz.device,
        )
        for cls in (0, 1, 2):
            idx = torch.nonzero(region_labels == cls, as_tuple=False).squeeze(-1)
            if idx.numel() < self.min_points_per_region:
                continue
            p = float(prune_cfg[cls])
            if p <= 0:
                continue
            k = int(idx.numel() * min(max(p, 0.0), 0.5))
            if k <= 0:
                continue
            local = torch.topk(importance[idx], k=k, largest=False).indices
            global_prune_mask[idx[local]] = True
        if global_prune_mask.any():
            global_prune_mask = global_prune_mask[: self.get_xyz.shape[0]]
            prune_candidates = torch.nonzero(global_prune_mask, as_tuple=False).squeeze(
                -1
            )
            cap = int(self.get_xyz.shape[0] * self.max_refine_prune_fraction)
            if cap > 0 and prune_candidates.numel() > cap:
                cand_scores = importance[prune_candidates]
                keep_idx = torch.topk(cand_scores, k=cap, largest=False).indices
                m = torch.zeros_like(global_prune_mask)
                m[prune_candidates[keep_idx]] = True
                global_prune_mask = m
            old = self.max_radii2D.clone()
            self.prune_points(global_prune_mask)
            if self.max_radii2D.shape[0] != self.get_xyz.shape[0]:
                self.max_radii2D = old[~global_prune_mask][: self.get_xyz.shape[0]]
            region_labels, lin, counts = self._region_labels()
            if region_labels is None:
                return
        self._fast_merge_and_apply(
            region_labels,
            lin,
            counts,
            merge_threshold_xyz,
            merge_threshold_color,
            merge_threshold_scale,
            merge_cap_ratio=0.02,
        )

    # ---- floater cleanup & adaptive cap ----
    def floater_cleanup(
        self,
        visibility_threshold,
        grad_threshold,
        opacity_threshold,
        importance_threshold,
        max_fraction,
        distance_factor=1.5,
    ):
        num_points = self.get_xyz.shape[0]
        if num_points == 0:
            return
        device = self.get_xyz.device
        with torch.no_grad():
            self._ensure_aux_lengths()
            visibility = self._visibility_count[:num_points]
            importance = torch.sigmoid(self.get_importance).squeeze(-1)
            if importance.shape[0] != num_points:
                importance = importance[:num_points]
            opacity = self.get_opacity.squeeze(-1)
            if opacity.shape[0] != num_points:
                opacity = opacity[:num_points]
            grad_norm = torch.norm(self._grad_ema_pos[:num_points], dim=1)
            scaling = self.get_scaling
            max_scale = scaling.max(dim=1).values
            min_scale = scaling.min(dim=1).values
            base_mask = visibility <= visibility_threshold
            if opacity_threshold > 0:
                base_mask &= opacity <= opacity_threshold
            if importance_threshold > 0:
                base_mask &= importance <= importance_threshold
            if grad_threshold > 0:
                base_mask &= grad_norm <= grad_threshold
            center = self._xyz.mean(dim=0, keepdim=True)
            distances = torch.norm(self._xyz - center, dim=1)
            far_mask = distances >= (
                max(float(self.scene_extent or 1.0), 1e-6) * distance_factor
            )
            detail_scale_threshold = max(
                0.02 * max(float(self.scene_extent or 1.0), 1e-6), 1e-5
            )
            fine_scale_mask = min_scale <= detail_scale_threshold
            candidate_mask = base_mask & far_mask & ~fine_scale_mask
            idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                return
            prune_cap = max(
                1, int(num_points * (max_fraction if max_fraction > 0 else 0.05))
            )
            if idx.numel() > prune_cap:
                _, order = torch.topk(importance[idx], k=prune_cap, largest=False)
                idx = idx[order]
            mask = torch.zeros(num_points, dtype=torch.bool, device=device)
            mask[idx] = True
            self.prune_points(mask, source="densify")

    def cap_gaussian_count(self, max_gaussians):
        if max_gaussians <= 0 or self.get_xyz.shape[0] <= max_gaussians:
            return
        with torch.no_grad():
            imp = torch.sigmoid(self.get_importance).squeeze(-1)
            excess = self.get_xyz.shape[0] - max_gaussians
            idx = torch.topk(imp, k=excess, largest=False).indices
            mask = torch.zeros(
                self.get_xyz.shape[0],
                dtype=torch.bool,
                device=self._xyz.device,
            )
            mask[idx] = True
            self.prune_points(mask, source="densify")

    def final_prune(self, threshold=0.05):
        with torch.no_grad():
            mask = (torch.sigmoid(self.get_importance) < threshold).squeeze()
            if mask.sum().item() > 0:
                self.prune_points(mask, source="final")


# ----------------------- extra losses -----------------------
def compute_multiscale_l1(pred, tgt, scales):
    if not scales:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    uniq = []
    for s in scales:
        if s and s not in uniq:
            uniq.append(float(s))
    uniq.sort(reverse=True)
    if 1.0 not in uniq:
        uniq.insert(0, 1.0)
    loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    wsum = 0.0
    for i, s in enumerate(uniq):
        w = 1.0 / (2**i)
        if s == 1.0:
            p, q = pred, tgt
        else:
            p = F.interpolate(
                pred,
                scale_factor=s,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )
            q = F.interpolate(
                tgt,
                scale_factor=s,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )
        loss = loss + w * F.l1_loss(p, q)
        wsum += w
    return loss / max(wsum, 1e-6)


def gradient_consistency_loss(pred, tgt):
    def g(img):
        return (
            img[..., :, 1:] - img[..., :, :-1],
            img[..., 1:, :] - img[..., :-1, :],
        )

    pdx, pdy = g(pred)
    tdx, tdy = g(tgt)
    return 0.5 * (F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy))


def color_moment_loss(pred, tgt, eps=1e-6):
    pf = pred.view(pred.shape[0], pred.shape[1], -1)
    qf = tgt.view(pred.shape[0], pred.shape[1], -1)
    pm, qm = pf.mean(-1), qf.mean(-1)
    ps, qs = (
        torch.sqrt(pf.var(-1, unbiased=False) + eps),
        torch.sqrt(qf.var(-1, unbiased=False) + eps),
    )
    return (pm - qm).abs().mean() + (ps - qs).abs().mean()


# ----------------------- dynamic render scale -----------------------
def _dyn_render_scale(iteration: int):
    if iteration < 4000:
        return 0.6
    if iteration < 10000:
        return 0.75
    return 1.0


def _copy_attr(dst, src, name_list):
    for n in name_list:
        if hasattr(src, n):
            try:
                setattr(dst, n, getattr(src, n))
            except Exception:
                pass


def make_scaled_camera(vp, scale: float):
    if scale >= 0.999:
        return vp

    class _CamProxy:
        pass

    new = _CamProxy()
    _copy_attr(
        new,
        vp,
        [
            "FoVx",
            "FoVy",
            "tan_fovx",
            "tan_fovy",
            "world_view_transform",
            "projection_matrix",
            "full_proj_transform",
            "world_view_proj",
            "camera_center",
            "camera_matrix",
            "viewmatrix",
            "projmatrix",
            "T",
            "fx",
            "fy",
            "cx",
            "cy",
            "intrinsics",
            "distortion_params",
            "background",
            "uid",
            "time",
            "shutter",
        ],
    )
    img = vp.original_image.to("cuda")
    img_bchw = to_chw01(img)
    H, W = img_bchw.shape[-2:]
    newH, newW = max(1, int(H * scale)), max(1, int(W * scale))
    img_ds = F.interpolate(img_bchw, size=(newH, newW), mode="area")
    new.original_image = img_ds[0].permute(1, 2, 0).contiguous()
    new.image_height, new.image_width = newH, newW
    if hasattr(vp, "viewport_size"):
        try:
            new.viewport_size = torch.tensor(
                [newW, newH], dtype=torch.int32, device="cuda"
            )
        except Exception:
            pass
    _copy_attr(
        new,
        vp,
        [
            "colmap_id",
            "aid",
            "sid",
            "exif",
            "path",
            "name",
            "frame_id",
            "dataset_index",
            "white_background",
            "near",
            "far",
            "exposure",
            "tone_mapping",
        ],
    )
    return new


# ----------------------- CSV / TB -----------------------
def prepare_output_and_logger(args):
    # 입력 경로(basename)로 출력 폴더명 지정
    if not getattr(args, "model_path", None) or args.model_path == "":
        src = (
            getattr(args, "source_path", None)
            or getattr(args, "source", None)
            or getattr(args, "data_path", None)
        )
        if src:
            dataname = os.path.basename(os.path.normpath(src))
            args.model_path = os.path.join("./output", dataname)
        else:
            args.model_path = os.path.join(
                "./output/",
                datetime.now().strftime("Model_%Y%m%d_%H%M%S"),
            )

    print("Output folder:", args.model_path)
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))

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
        "Train/MultiScale",
        "Train/Grad",
        "Train/ColorMoment",
        "Train/LPIPS",
        "Train/Depth",
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
    writer = csv.DictWriter(csv_file, fieldnames=header)
    writer.writeheader()

    flat_path = os.path.join(args.model_path, "metrics_log.csv")
    flat_file = open(flat_path, "w", newline="")
    flat_header = [
        "Iteration",
        "SSIM",
        "L1",
        "PSNR",
        "LPIPS",
        "Loss",
        "FPS",
        "IterTime",
        "Gaussians",
    ]
    flat_writer = csv.DictWriter(flat_file, fieldnames=flat_header)
    flat_writer.writeheader()

    print(f"[CSV] Eval log  → {csv_path}")
    print(f"[CSV] Flat log  → {flat_path}")

    tb = SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None
    return tb, writer, csv_file, flat_writer, flat_file


def measure_model_sizes(output_dir):
    model_sizes = {}
    model_dir_pattern = os.path.join(output_dir, "point_cloud", "iteration_*")
    for model_dir in glob.glob(model_dir_pattern):
        iteration = os.path.basename(model_dir).split("_")[-1]
        total_size = sum(
            os.path.getsize(p) for p in glob.glob(os.path.join(model_dir, "*.ply"))
        )
        model_sizes[iteration] = total_size / (1024 * 1024)
    return model_sizes


# ----------------------- Validation -----------------------
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
            return {
                "Val/PSNR": 0.0,
                "Val/SSIM": 0.0,
                "Val/LPIPS": 0.0,
                "Val/L1": 0.0,
                "Val/DepthAbsRel": 0.0,
                "Val/DepthDelta1.25": 0.0,
                "RenderFPS": 0.0,
            }
        step = max(1, len(tr) // 10)
        cams = [tr[i] for i in range(0, len(tr), step)]
    for vcam in tqdm(cams, desc="[Val]"):
        gt = vcam.original_image.cuda()
        start = time.time()
        with (
            torch.no_grad(),
            suppress_debug_lines(("DEBUG:", "EARLY DEBUG:", "RASTERIZER DEBUG:")),
        ):
            pkg = render(
                vcam,
                gauss,
                pipe,
                bg,
                importance_scores=torch.sigmoid(gauss.get_importance),
            )
        total_t += time.time() - start
        img = torch.clamp(pkg["render"], 0, 1)
        psnr_t += psnr(_ensure_batch_dim(img), _ensure_batch_dim(gt)).mean().item()
        ssim_t += ssim(_ensure_batch_dim(img), _ensure_batch_dim(gt)).double().item()
        if _HAS_LPIPS:
            lpips_t += (
                _lpips(_ensure_batch_dim(img), _ensure_batch_dim(gt))
                .mean()
                .double()
                .item()
            )
        l1_t += l1_loss(img, gt).mean().double().item()
        gtd, _ = depth_est.load(vcam)
        if gtd is not None:
            rd = pkg["depth"].squeeze(0)
            if gtd.shape != rd.shape:
                gtd = F.interpolate(
                    gtd.unsqueeze(0).unsqueeze(0),
                    size=rd.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            mask = (gtd > 1e-6) & (rd > 1e-6)
            if mask.sum() > 0:
                gtv, rdv = gtd[mask], rd[mask]
                t1, t2 = torch.median(gtv), torch.median(rdv)
                s1 = torch.median((gtv - t1).abs())
                s2 = torch.median((rdv - t2).abs())
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
        "Val/PSNR": psnr_t / n if n else 0.0,
        "Val/SSIM": ssim_t / n if n else 0.0,
        "Val/LPIPS": (lpips_t / n if (n and _HAS_LPIPS) else 0.0),
        "Val/L1": l1_t / n if n else 0.0,
        "Val/DepthAbsRel": float(np.mean(d_absrel)) if d_absrel else 0.0,
        "Val/DepthDelta1.25": float(np.mean(d_delta1)) if d_delta1 else 0.0,
        "RenderFPS": (n / total_t) if total_t > 0 else 0.0,
    }


# ----------------------- Training -----------------------
DEFAULT_FINAL_ITERS = 30000  # 🔺 20k → 30k
WARMUP_L2_ONLY_ITERS = 5000
POST_DSSIM = 0.2
POST_L1 = 0.3
POST_L2 = 0.7


def get_l2_scale(iteration: int, total_iters: int = DEFAULT_FINAL_ITERS) -> float:
    """
    두 번째 스크립트 스타일의 '초기 L2 스케일링' 스케줄.
    """
    if total_iters <= 0:
        total_iters = DEFAULT_FINAL_ITERS
    t = float(iteration) / float(total_iters)
    if iteration < 500:
        return 0.05
    elif iteration < 2000:
        return 0.2
    elif iteration < 5000:
        return 0.5
    elif iteration < 12000:
        return 0.8
    else:
        return 1.0


def training(
    dataset, opt, pipe, test_iters, save_iters, ckpt_iters, checkpoint, debug_from
):
    seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    first_iter = 0
    tb_writer, csv_writer, csv_file, flat_writer, flat_file = prepare_output_and_logger(
        dataset
    )
    gauss = RefinedGaussianModel(dataset.sh_degree, verbose=True)
    dataset.test_iterations = test_iters
    scene = Scene(dataset, gauss)

    # scene 정보 등록
    gauss.register_scene_extent(scene.cameras_extent)

    # depth-first 클러스터링용 카메라 등록
    try:
        gauss.register_cameras(scene.getTrainCameras() + scene.getTestCameras())
    except Exception:
        try:
            gauss.register_cameras(scene.getTrainCameras())
        except Exception:
            pass

    gauss.training_setup(opt)

    depth_est = DepthEstimator(cache_dir=os.path.join(scene.model_path, "depth_cache"))
    if not depth_est.available and getattr(opt, "lambda_depth", 0.0) > 0:
        print("[Depth] disabling lambda_depth (no estimator)")
        opt.lambda_depth = 0.0
    depth_est.precompute(scene.getTrainCameras() + scene.getTestCameras())

    bg = torch.tensor(
        ([1, 1, 1] if dataset.white_background else [0, 0, 0]),
        dtype=torch.float32,
        device="cuda",
    )

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    bar = tqdm(range(first_iter, opt.iterations + 1), desc="[Train]")

    # multiscale list (optional)
    scales = [1.0, 0.5, 0.25]
    if hasattr(opt, "multiscale_scales"):
        raw = getattr(opt, "multiscale_scales", "")
        try:
            if isinstance(raw, str):
                scales = [float(s.strip()) for s in raw.split(",") if s.strip()]
        except Exception:
            pass

    flat_interval = max(1, int(getattr(opt, "flat_log_interval", 50)))

    for it in range(first_iter + 1, opt.iterations + 1):
        iter_start.record()
        gauss.update_learning_rate(it)
        if it % 1000 == 0:
            gauss.oneupSHdegree()
        gauss.optimizer.zero_grad(set_to_none=True)

        # choose camera & dynamic scale
        tr = scene.getTrainCameras()
        vcam = tr.copy().pop(randint(0, len(tr) - 1))
        scale = _dyn_render_scale(it)
        vcam_scaled = make_scaled_camera(vcam, scale)

        with suppress_debug_lines(("DEBUG:", "EARLY DEBUG:", "RASTERIZER DEBUG:")):
            pkg = render(
                vcam_scaled,
                gauss,
                pipe,
                bg,
                importance_scores=torch.sigmoid(gauss.get_importance),
            )
        img, depth_r, vis = pkg["render"], pkg["depth"], pkg["visibility_filter"]
        gt = vcam_scaled.original_image.cuda()

        # sRGB metrics
        pred = to_chw01(img)
        tgt = to_chw01(gt)
        if pred.shape[-2:] != tgt.shape[-2:]:
            tgt = F.interpolate(
                tgt,
                size=pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        l1v = l1_loss(pred, tgt)
        l2v = F.mse_loss(pred, tgt)
        ssim_v = ssim(pred, tgt)

        # L2 warmup 스케일
        l2_scale = get_l2_scale(it, opt.iterations)

        if it < WARMUP_L2_ONLY_ITERS:
            # 초기: L2만 사용 (scaling)
            lam_l1 = 0.0
            lam_dssim = 0.0
            lam_l2 = l2_scale
        else:
            # 이후: L1 + DSSIM + scaled L2
            lam_l1 = POST_L1
            lam_dssim = getattr(opt, "lambda_dssim", POST_DSSIM)
            lam_l2 = POST_L2 * l2_scale

        multiscale_loss = (
            compute_multiscale_l1(pred, tgt, scales)
            if getattr(opt, "lambda_multiscale", 0.0) > 0
            else torch.zeros((), device=pred.device)
        )
        grad_loss = (
            gradient_consistency_loss(pred, tgt)
            if getattr(opt, "lambda_grad", 0.0) > 0
            else torch.zeros((), device=pred.device)
        )
        moment_loss = (
            color_moment_loss(pred, tgt)
            if getattr(opt, "lambda_color_moment", 0.0) > 0
            else torch.zeros((), device=pred.device)
        )
        lpips_metric = (
            _lpips(pred, tgt).mean()
            if _HAS_LPIPS
            else torch.zeros((), device=pred.device)
        )
        lpips_loss = getattr(opt, "lambda_lpips", 0.1) * lpips_metric

        # depth (robust aligned)
        depth_loss = torch.zeros((), device=pred.device)
        gtd, unc = depth_est.load(vcam)
        if gtd is not None:
            rd = depth_r.squeeze(0)
            if gtd.shape != rd.shape:
                gtd = F.interpolate(
                    gtd.unsqueeze(0).unsqueeze(0),
                    size=rd.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            if unc is not None and unc.shape != rd.shape:
                unc = F.interpolate(
                    unc.unsqueeze(0).unsqueeze(0),
                    size=rd.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            mask = (gtd > 1e-6) & (rd > 1e-6)
            if mask.sum() > 0:
                gtv, rdv = gtd[mask], rd[mask]
                t1, t2 = torch.median(gtv), torch.median(rdv)
                s1 = torch.median((gtv - t1).abs())
                s2 = torch.median((rdv - t2).abs())
                rd_al = (rdv - t2) * (s1 / (s2 + 1e-6)) + t1
                if unc is None:
                    w = torch.ones_like(gtv)
                else:
                    tau = max(getattr(opt, "uncertainty_tau", 0.1), 1e-6)
                    w = torch.exp(unc[mask].neg() / tau)
                w = w / (w.mean() + 1e-6)
                depth_loss = (w * F.huber_loss(rd_al, gtv, reduction="none")).mean()

        total = (
            lam_l1 * l1v
            + lam_l2 * l2v
            + lam_dssim * (1.0 - ssim_v)
            + getattr(opt, "lambda_multiscale", 0.0) * multiscale_loss
            + getattr(opt, "lambda_grad", 0.0) * grad_loss
            + getattr(opt, "lambda_color_moment", 0.0) * moment_loss
            + getattr(opt, "lambda_depth", 0.0) * depth_loss
            + getattr(opt, "lambda_importance", 0.01)
            * torch.sigmoid(gauss.get_importance).mean()
            + lpips_loss
        )

        total.backward()
        gauss.optimizer.step()

        iter_end.record()
        torch.cuda.synchronize()
        elapsed = iter_start.elapsed_time(iter_end)  # ms

        # --------- Growth (densify) / Refine ---------
        with torch.no_grad():
            gauss._ensure_point_buffers()
            radii = align_tensor_to_length(pkg["radii"], gauss.get_xyz.shape[0])
            if it < opt.densify_until_iter:
                # Densify phase (until 15000)
                gauss.max_radii2D[vis] = torch.max(gauss.max_radii2D[vis], radii[vis])
                gauss.add_densification_stats(pkg["viewspace_points"], vis)
                fast_interval = max(1, int(getattr(opt, "densification_interval", 100)))
                if it > opt.densify_from_iter and it % fast_interval == 0:
                    gauss.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.003,
                        scene.cameras_extent,
                        None,
                        radii,
                    )
                if it % opt.opacity_reset_interval == 0:
                    gauss.reset_opacity()
            else:
                # Refine phase (from 15000 onwards)
                if it % 200 == 0:
                    gauss._ensure_point_buffers()
                    radii = align_tensor_to_length(pkg["radii"], gauss.get_xyz.shape[0])
                    gauss.refine_and_prune(
                        merge_threshold_xyz=0.01,
                        merge_threshold_color=0.12,
                        merge_threshold_scale=0.12,
                        prune_percent=0.005,
                        radii=radii,
                    )
                if it % 500 == 0:
                    low_imp = (torch.sigmoid(gauss.get_importance) < 0.05).squeeze()
                    if low_imp.any():
                        gauss.prune_points(low_imp, source="final")

            # Floater cleanup: start at 25000 iters
            if (
                it >= getattr(opt, "floater_cleanup_start", 25000)
                and it % getattr(opt, "floater_cleanup_interval", 400) == 0
            ):
                gauss.floater_cleanup(
                    visibility_threshold=getattr(
                        opt, "floater_visibility_threshold", 2
                    ),
                    grad_threshold=getattr(opt, "floater_grad_threshold", 5e-4),
                    opacity_threshold=getattr(opt, "floater_opacity_threshold", 0.04),
                    importance_threshold=getattr(
                        opt, "floater_importance_threshold", 0.35
                    ),
                    max_fraction=getattr(opt, "floater_cleanup_max_ratio", 0.05),
                    distance_factor=getattr(opt, "floater_distance_factor", 1.4),
                )
                if getattr(opt, "max_gaussians", 0) > 0:
                    gauss.cap_gaussian_count(getattr(opt, "max_gaussians", 0))

        # --------- Eval/Save/Log ---------
        # per-iter metric 출력 제거, eval은 지정된 test_iters에서만
        do_eval = it in test_iters

        pruned_iter = gauss.event_counters.get(
            "Pruned(Densify)_Iter", 0
        ) + gauss.event_counters.get("Pruned(Final)_Iter", 0)
        pruned_cum = gauss.event_counters.get(
            "Pruned(Densify)_Cum", 0
        ) + gauss.event_counters.get("Pruned(Final)_Cum", 0)

        base_row = {
            "Iter": it,
            "Train/L1": float(l1v.item()),
            "Train/TotalLoss": float(total.item()),
            "Train/SSIM": float(ssim_v.item()),
            "Train/MultiScale": float(multiscale_loss.item())
            if multiscale_loss.numel()
            else 0.0,
            "Train/Grad": float(grad_loss.item()) if grad_loss.numel() else 0.0,
            "Train/ColorMoment": float(moment_loss.item())
            if moment_loss.numel()
            else 0.0,
            "Train/LPIPS": float(lpips_metric.item()) if _HAS_LPIPS else 0.0,
            "Train/Depth": float(depth_loss.item()) if depth_loss.numel() else 0.0,
            **gauss.event_counters,
            "Pruned(Total)_Iter": pruned_iter,
            "Pruned(Total)_Cum": pruned_cum,
            "TotalGaussians": gauss.get_xyz.shape[0],
            "IterTime(ms)": float(elapsed),
            "Model_Size(MB)": "N/A",
            "Val/PSNR": "N/A",
            "Val/SSIM": "N/A",
            "Val/LPIPS": "N/A",
            "Val/L1": "N/A",
            "Val/DepthAbsRel": "N/A",
            "Val/DepthDelta1.25": "N/A",
            "RenderFPS": "N/A",
        }

        if do_eval:
            val_row = validation_loop(it, scene, gauss, pipe, bg, depth_est)
            base_row.update(val_row)
            pc_dir = os.path.join(scene.model_path, "point_cloud", f"iteration_{it}")
            size_mb = 0.0
            if os.path.isdir(pc_dir):
                size_mb = sum(
                    os.path.getsize(p) for p in glob.glob(os.path.join(pc_dir, "*.ply"))
                ) / (1024 * 1024)
            if size_mb > 0:
                base_row["Model_Size(MB)"] = f"{size_mb:.2f}"

            # 🔔 psnr / ssim / lpips / l1 콘솔 출력은 29800에서만
            if it == 29800:
                print(
                    f"[Eval @ {it}] "
                    f"PSNR={val_row['Val/PSNR']:.3f}, "
                    f"SSIM={val_row['Val/SSIM']:.4f}, "
                    f"LPIPS={val_row['Val/LPIPS']:.4f}, "
                    f"L1={val_row['Val/L1']:.6f}, "
                    f"DepthAbsRel={val_row['Val/DepthAbsRel']:.4f}, "
                    f"DepthΔ1.25={val_row['Val/DepthDelta1.25']:.4f}"
                )

        # write eval/detailed row (CSV는 계속 전체 iter 기록)
        csv_writer.writerow({k: base_row.get(k, "N/A") for k in csv_writer.fieldnames})
        csv_file.flush()
        gauss.reset_iter_counters()

        # write flat row at interval (viewer-friendly)
        if (it % flat_interval) == 0 or do_eval or it == opt.iterations:
            flat_writer.writerow(
                {
                    "Iteration": it,
                    "SSIM": float(ssim_v.item()),
                    "L1": float(l1v.item()),
                    "PSNR": float(psnr(pred, tgt).mean().item()),
                    "LPIPS": float(lpips_metric.item()) if _HAS_LPIPS else 0.0,
                    "Loss": float(total.item()),
                    "FPS": float(1000.0 / max(1e-3, elapsed)),
                    "IterTime": float(elapsed),
                    "Gaussians": int(gauss.get_xyz.shape[0]),
                }
            )
            flat_file.flush()

        if it in save_iters:
            scene.save(it)
        if it in ckpt_iters:
            torch.save((gauss.capture(), it), scene.model_path + f"/chkpnt{it}.pth")

        bar.update(1)

    bar.close()
    csv_file.close()
    flat_file.close()
    print("[Train] complete.")


# ----------------------- Main -----------------------
if __name__ == "__main__":
    parser = ArgumentParser(description="Unified training script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # General / IO
    safe_add_argument(parser, "--ip", type=str, default="127.0.0.1")
    safe_add_argument(parser, "--port", type=int, default=6009)
    safe_add_argument(parser, "--debug_from", type=int, default=-1)
    safe_add_argument(parser, "--detect_anomaly", action="store_true", default=False)
    safe_add_argument(
        parser,
        "--test_iterations",
        nargs="+",
        type=int,
        default=[29800],  # 🔺 메트릭은 29800에서만
    )
    safe_add_argument(
        parser,
        "--save_iterations",
        nargs="+",
        type=int,
        default=[DEFAULT_FINAL_ITERS],
    )
    safe_add_argument(parser, "--quiet", action="store_true")
    safe_add_argument(
        parser, "--checkpoint_iterations", nargs="+", type=int, default=[]
    )
    safe_add_argument(parser, "--start_checkpoint", type=str, default=None)

    # Loss / regularization
    safe_add_argument(parser, "--lambda_dssim", type=float, default=0.2)
    safe_add_argument(parser, "--lambda_depth", type=float, default=0.1)
    safe_add_argument(parser, "--lambda_lpips", type=float, default=0.1)
    safe_add_argument(parser, "--lambda_importance", type=float, default=0.01)
    safe_add_argument(parser, "--uncertainty_tau", type=float, default=0.1)
    safe_add_argument(parser, "--lambda_multiscale", type=float, default=0.0)
    safe_add_argument(parser, "--lambda_grad", type=float, default=0.0)
    safe_add_argument(parser, "--lambda_color_moment", type=float, default=0.0)

    # per-iter densification limits (0 = unlimited)
    safe_add_argument(parser, "--max_clone_points_per_iter", type=int, default=0)
    safe_add_argument(parser, "--max_split_points_per_iter", type=int, default=0)
    safe_add_argument(
        parser, "--log_densify_clamps", action="store_true", default=False
    )

    # adaptive cap (0 = disabled)
    safe_add_argument(parser, "--max_gaussians", type=int, default=0)

    # floater cleanup scheduling (기본 시작: 25000 iter)
    safe_add_argument(parser, "--floater_cleanup_start", type=int, default=25000)
    safe_add_argument(parser, "--floater_cleanup_interval", type=int, default=400)
    safe_add_argument(parser, "--floater_visibility_threshold", type=int, default=2)
    safe_add_argument(parser, "--floater_grad_threshold", type=float, default=5e-4)
    safe_add_argument(parser, "--floater_opacity_threshold", type=float, default=0.04)
    safe_add_argument(
        parser, "--floater_importance_threshold", type=float, default=0.35
    )
    safe_add_argument(parser, "--floater_cleanup_max_ratio", type=float, default=0.05)
    safe_add_argument(parser, "--floater_distance_factor", type=float, default=1.4)

    # eval cadence & flat log
    safe_add_argument(parser, "--eval_interval", type=int, default=200)
    safe_add_argument(parser, "--flat_log_interval", type=int, default=50)

    args = parser.parse_args(sys.argv[1:])

    # ✅ Hard-fix iterations to 30k
    args.iterations = DEFAULT_FINAL_ITERS
    if DEFAULT_FINAL_ITERS not in args.save_iterations:
        args.save_iterations.append(DEFAULT_FINAL_ITERS)
    # ⚠ test_iterations에는 DEFAULT_FINAL_ITERS를 강제 추가하지 않음
    #    → 기본값이면 29800에서만 eval

    print("Optimizing", args.model_path)
    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset_params = lp.extract(args)
    opt_params = op.extract(args)
    pipe_params = pp.extract(args)
    opt_params.iterations = DEFAULT_FINAL_ITERS  # guard inner defaults

    # 🔒 densify phase stop at 15000 iters (→ refine from 15000)
    setattr(opt_params, "densify_until_iter", 15000)
    print(
        f"[Sched] densify_until_iter set to {getattr(opt_params, 'densify_until_iter', None)}"
    )

    # map a few CLI → opt
    for k in [
        "lambda_depth",
        "lambda_lpips",
        "lambda_importance",
        "uncertainty_tau",
        "eval_interval",  # (사용은 안 하지만 그대로 보존)
        "max_clone_points_per_iter",
        "max_split_points_per_iter",
        "log_densify_clamps",
        "max_gaussians",
        "floater_cleanup_start",
        "floater_cleanup_interval",
        "floater_visibility_threshold",
        "floater_grad_threshold",
        "floater_opacity_threshold",
        "floater_importance_threshold",
        "floater_cleanup_max_ratio",
        "floater_distance_factor",
        "lambda_multiscale",
        "lambda_grad",
        "lambda_color_moment",
        "flat_log_interval",
    ]:
        if hasattr(args, k):
            setattr(opt_params, k, getattr(args, k))

    training(
        dataset_params,
        opt_params,
        pipe_params,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )
