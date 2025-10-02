"""
This script combines depth map generation using Depth Anything V2 and the calculation
of depth scale/offset parameters from a COLMAP sparse model.

It performs two main steps:
1.  Generates 16-bit single-channel depth maps for all images in a dataset.
2.  Reads a COLMAP sparse reconstruction, matches 2D keypoints to 3D points,
    and compares the COLMAP-derived depth with the monocular depth map to find
    a linear scale and offset for each image.
3.  Saves the calculated parameters to a `depth_params.json` file.

This integrated workflow is designed to be a one-stop-shop for preparing custom
datasets for methods that utilize monocular depth priors, like 3D Gaussian Splatting.
"""
import argparse
import cv2
import glob
import numpy as np
import os
import torch
import json
from joblib import delayed, Parallel
from tqdm import tqdm

# --- Dependency Imports ---
# Assumes 'read_write_model.py' and the 'Depth-Anything-V2' repository are accessible.

try:
    # Helper script from COLMAP to read its model structure.
    from read_write_model import read_model, qvec2rotmat
except ImportError:
    print("\n[ERROR] 'read_write_model.py' not found.")
    print("Please download it from the COLMAP or Gaussian Splatting repository and")
    print("place it in the same directory as this script.\n")
    exit(1)

try:
    # The main Depth Anything V2 model class.
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print("\n[ERROR] 'depth_anything_v2' package not found.")
    print("Please ensure you have cloned the DepthAnythingV2 repository and it is")
    print("in your Python path or the current working directory.\n")
    exit(1)


def get_scales(image_key, cameras, images, points3d_ordered, depths_dir):
    """
    Calculates the scale and offset to align a monocular depth map with COLMAP's sparse point cloud.

    Args:
        image_key: The key (ID) of the image in the COLMAP model.
        cameras: Dictionary of COLMAP camera objects.
        images: Dictionary of COLMAP image objects.
        points3d_ordered: A numpy array of 3D points, indexed by point3D_id.
        depths_dir: The directory where monocular depth maps are stored.

    Returns:
        A dictionary containing the image name, scale, and offset, or None on failure.
    """
    image_meta = images[image_key]
    cam_intrinsic = cameras[image_meta.camera_id]
    image_base_name = os.path.splitext(image_meta.name)[0]

    # 1. Get 2D-3D correspondences for this image.
    pts3d_ids = image_meta.point3D_ids
    xys = image_meta.xys

    # 2. Filter out 2D points that don't correspond to a 3D point.
    has_3d_point_mask = (pts3d_ids != -1)
    pts3d_ids = pts3d_ids[has_3d_point_mask]
    xys = xys[has_3d_point_mask]

    # 3. Filter out points with IDs that are out of bounds for our lookup table.
    in_bounds_mask = pts3d_ids < len(points3d_ordered)
    pts3d_ids = pts3d_ids[in_bounds_mask]
    valid_xys = xys[in_bounds_mask]

    # If no valid points remain, return a default value.
    if len(pts3d_ids) == 0:
        return {"image_name": image_base_name, "scale": 0, "offset": 0}

    # 4. Get 3D point coordinates and transform them into the camera's coordinate system.
    pts3d = points3d_ordered[pts3d_ids]
    R = qvec2rotmat(image_meta.qvec)
    t = image_meta.tvec
    pts_camera_frame = pts3d @ R.T + t
    
    # 5. Calculate inverse depth from the COLMAP points.
    depths_colmap = pts_camera_frame[:, 2]
    inv_depths_colmap = 1.0 / depths_colmap

    # 6. Load the corresponding monocular depth map.
    depth_map_path = os.path.join(depths_dir, f"{image_base_name}.png")
    inv_depth_mono_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
    
    if inv_depth_mono_map is None:
        print(f"Warning: Could not read depth map {depth_map_path}")
        return None

    # Ensure it's single channel and normalized to [0, 1].
    if inv_depth_mono_map.ndim == 3:
        inv_depth_mono_map = inv_depth_mono_map[..., 0]
    inv_depth_mono_map = inv_depth_mono_map.astype(np.float32) / 65535.0
    
    # 7. Sample monocular depth at the 2D keypoint locations.
    h_mono, w_mono = inv_depth_mono_map.shape
    h_cam, w_cam = cam_intrinsic.height, cam_intrinsic.width
    
    scale_w, scale_h = w_mono / w_cam, h_mono / h_cam
    x_coords = (valid_xys[:, 0] * scale_w).astype(np.float32)
    y_coords = (valid_xys[:, 1] * scale_h).astype(np.float32)

    # 8. Filter points to ensure they are within the depth map bounds and have positive depth.
    final_valid_mask = (
        (x_coords >= 0) & (x_coords < w_mono) &
        (y_coords >= 0) & (y_coords < h_mono) &
        (inv_depths_colmap > 0)
    )

    num_valid_points = final_valid_mask.sum()
    
    # We need enough points with sufficient depth variation to get a stable estimate.
    if num_valid_points > 10 and (inv_depths_colmap[final_valid_mask].max() - inv_depths_colmap[final_valid_mask].min()) > 1e-3:
        x_coords_valid = x_coords[final_valid_mask]
        y_coords_valid = y_coords[final_valid_mask]
        inv_depths_colmap_valid = inv_depths_colmap[final_valid_mask]

        # Use cv2.remap to sample depths with bilinear interpolation.
        inv_depths_mono_sampled = cv2.remap(
            inv_depth_mono_map, x_coords_valid, y_coords_valid, interpolation=cv2.INTER_LINEAR
        ).flatten()

        # 9. Compute scale and offset using median and mean absolute deviation for robustness.
        t_colmap = np.median(inv_depths_colmap_valid)
        s_colmap = np.mean(np.abs(inv_depths_colmap_valid - t_colmap))

        t_mono = np.median(inv_depths_mono_sampled)
        s_mono = np.mean(np.abs(inv_depths_mono_sampled - t_mono))

        if s_mono > 1e-6:
            scale = s_colmap / s_mono
            offset = t_colmap - t_mono * scale
        else:
            scale, offset = 0, 0
    else:
        scale, offset = 0, 0
    
    return {"image_name": image_base_name, "scale": scale, "offset": offset}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate depth maps and calculate scale/offset for COLMAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--base_dir', required=True, help="Root directory of the dataset (e.g., 'data/room') which contains 'images' and 'sparse/0' subdirectories.")
    parser.add_argument('--depths_dir', help="Directory to save depth maps. Defaults to '{base_dir}/depthmap'.")
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'], help="Depth Anything V2 model encoder.")
    parser.add_argument('--input-size', type=int, default=518, help="Input image size for the depth model.")
    parser.add_argument('--model-type', default="bin", choices=["bin", "txt"], help="COLMAP model file type (binary '.bin' or text '.txt').")
    
    args = parser.parse_args()

    # --- Path Setup ---
    images_dir = os.path.join(args.base_dir, "images")
    depths_dir = args.depths_dir if args.depths_dir else os.path.join(args.base_dir, "depthmap")
    sparse_dir = os.path.join(args.base_dir, "sparse", "0")
    output_json_path = os.path.join(sparse_dir, "depth_params.json")

    os.makedirs(depths_dir, exist_ok=True)
    if not os.path.isdir(images_dir):
        print(f"[ERROR] Image directory not found at: {images_dir}")
        exit(1)
    if not os.path.isdir(sparse_dir):
        print(f"[ERROR] COLMAP sparse model directory not found at: {sparse_dir}")
        exit(1)

    # =========================================================================
    # PART 1: Generate 16-bit Depth Maps
    # =========================================================================
    print("--- Part 1: Generating depth maps ---")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    checkpoint_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        print("Please download the Depth Anything V2 checkpoints and place them in a 'checkpoints' directory.")
        exit(1)

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    filenames = sorted([p for p in glob.glob(os.path.join(images_dir, '**/*'), recursive=True) if os.path.splitext(p)[1].lower() in image_extensions])
    print(f"Found {len(filenames)} images to process.")

    for filename in tqdm(filenames, desc="Generating depths"):
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"Warning: Could not read image {filename}, skipping.")
            continue
        
        depth = depth_anything.infer_image(raw_image, args.input_size)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth_uint16 = (depth * 65535.0).astype(np.uint16)
        
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_path = os.path.join(depths_dir, f'{base_name}.png')
        cv2.imwrite(output_path, depth_uint16)

    print("Depth map generation complete.")

    # =========================================================================
    # PART 2: Calculate Scale and Offset from COLMAP Model
    # =========================================================================
    print("\n--- Part 2: Calculating depth scales and offsets ---")
    
    try:
        cameras, images, points3d = read_model(sparse_dir, ext=f".{args.model_type}")
    except FileNotFoundError:
        print(f"[ERROR] Could not find COLMAP model files in {sparse_dir} with extension '.{args.model_type}'")
        exit(1)
        
    print(f"Loaded {len(cameras)} cameras, {len(images)} images, and {len(points3d)} 3D points from COLMAP model.")

    max_point_id = max(p3d.id for p3d in points3d.values()) if len(points3d) > 0 else 0
    points3d_ordered = np.zeros([max_point_id + 1, 3])
    for point3d_id, point3d in points3d.items():
        points3d_ordered[point3d.id] = point3d.xyz

    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(key, cameras, images, points3d_ordered, depths_dir) for key in tqdm(images.keys(), desc="Calculating scales")
    )
    
    depth_params = {
        param["image_name"]: {"scale": param["scale"], "offset": param["offset"]}
        for param in depth_param_list if param is not None
    }

    with open(output_json_path, "w") as f:
        json.dump(depth_params, f, indent=4)

    print(f"\nSuccessfully saved depth parameters to: {output_json_path}")
    print("Process finished successfully.")