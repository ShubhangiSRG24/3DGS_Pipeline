"""
Image loading and preprocessing utilities
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import torch
from PIL import Image
import os


def load_images(input_dir: str) -> List[str]:
    """Load image paths from directory"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_path.glob(f"*{ext}"))
        image_paths.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")
    
    # Convert to strings and sort
    image_paths = [str(p) for p in image_paths]
    image_paths.sort()
    
    print(f"Found {len(image_paths)} images")
    return image_paths


def resize_images(image_paths: List[str], max_size: int = 1600) -> List[Dict]:
    """Resize images and return processed image data"""
    processed_images = []
    
    for img_path in image_paths:
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            
            # Get original dimensions
            height, width = image.shape[:2]
            
            # Resize if necessary
            if max(width, height) > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                print(f"Resized {img_path} from {width}x{height} to {new_width}x{new_height}")
            
            processed_images.append({
                'path': img_path,
                'image': image,
                'original_shape': (height, width),
                'processed_shape': image.shape[:2]
            })
            
        except Exception as e:
            print(f"Warning: Error processing {img_path}: {e}")
            continue
    
    return processed_images


def load_image_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """Load image as tensor for GPU processing"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image_tensor.to(device)


def batch_load_images(image_paths: List[str], device: torch.device, 
                     batch_size: int = 8) -> List[torch.Tensor]:
    """Load images in batches for GPU processing"""
    image_tensors = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        
        for img_path in batch_paths:
            try:
                tensor = load_image_tensor(img_path, device)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
                continue
        
        if batch_tensors:
            # Stack batch tensors
            batch_tensor = torch.cat(batch_tensors, dim=0)
            image_tensors.append(batch_tensor)
    
    return image_tensors


def save_processed_image(image: np.ndarray, output_path: str):
    """Save processed image"""
    cv2.imwrite(output_path, image)


def create_image_pyramid(image: np.ndarray, levels: int = 3) -> List[np.ndarray]:
    """Create image pyramid for multi-scale processing"""
    pyramid = [image]
    
    for i in range(1, levels):
        # Downsample by factor of 2
        height, width = pyramid[-1].shape[:2]
        new_height, new_width = height // 2, width // 2
        
        if new_height > 0 and new_width > 0:
            downsampled = cv2.resize(pyramid[-1], (new_width, new_height))
            pyramid.append(downsampled)
        else:
            break
    
    return pyramid


def extract_patches(image: np.ndarray, patch_size: int = 64, stride: int = 32) -> List[np.ndarray]:
    """Extract patches from image for local processing"""
    patches = []
    height, width = image.shape[:2]
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    
    return patches


def compute_image_statistics(image: np.ndarray) -> Dict:
    """Compute basic image statistics"""
    stats = {
        'mean': np.mean(image),
        'std': np.std(image),
        'min': np.min(image),
        'max': np.max(image),
        'shape': image.shape
    }
    
    if len(image.shape) == 3:
        # Color image
        stats['mean_rgb'] = np.mean(image, axis=(0, 1))
        stats['std_rgb'] = np.std(image, axis=(0, 1))
    
    return stats


def normalize_image(image: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """Normalize image to specified mean and std"""
    image_norm = (image - image.mean()) / (image.std() + 1e-8)
    image_norm = image_norm * std + mean
    return np.clip(image_norm, 0, 1)


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to improve contrast"""
    if len(image.shape) == 3:
        # Color image - apply to each channel
        image_eq = np.zeros_like(image)
        for i in range(3):
            image_eq[:, :, i] = cv2.equalizeHist(image[:, :, i])
    else:
        # Grayscale image
        image_eq = cv2.equalizeHist(image)
    
    return image_eq


def resize_image(image_path: str, max_size: int = 1600) -> np.ndarray:
    """Resize a single image to maximum size while maintaining aspect ratio"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image {image_path}")
        
        # Get original dimensions
        height, width = image.shape[:2]
        
        # Resize if necessary
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        return image
        
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}") 